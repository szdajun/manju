"""Wan2.2-5B I2V 逐镜入口(单模型)。每镜: free VRAM → 上传关键帧 → build_wan → queue → 取 mp4 → 命门卡2。

用法:
  uv run python scripts/gen_wan.py --smoke                 # 烟雾: 小配置+占位图, 验管线端到端
  uv run python scripts/gen_wan.py --shot 4 --length 81    # 单镜(需关键帧, 跑命门卡2)
  uv run python scripts/gen_wan.py                          # 全 10 镜
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.comfyui_client import ComfyUIClient, build_wan_i2v_workflow  # noqa: E402
from lib.wan_shots import WAN_PROMPTS, WAN_NEG  # noqa: E402
from lib.ep02_shots import I2V_MOTION as EP02_MOTION  # noqa: E402

COMFY_OUT = "../ComfyUI/output"
FIT_PY = "../fitness-video-pipeline/.venv/Scripts/python.exe"
FFMPEG = "~/ffmpeg/ffmpeg.exe"
FFPROBE = "~/ffmpeg/ffprobe.exe"
THRESH = 0.45   # Wan clip 命门卡(允许略松于关键帧 0.50)

# 每 episode 配置: 关键帧目录/脸金标/输出/motion prompts/shot 前缀 + Wan 生产配置。
# ep01=主角写实(Seedance 历史已跑; Wan 路径备用); ep02=小雅写实(2026-07-21 WSL Wan i2v)。
# ep02 Wan 生产配置(12GB fit + 全画质, WSL 实测): cfg4/tile192/480×832/len81/steps30/blocks25。
# KF/REF/OUT/pfx 在 main() 里按 --episode 从这里取并 global 注入(run_one 读全局)。
EPISODES = {
    "ep01": dict(
        kf="../manju/output/ep01_real",
        ref="../manju/output/lihui_real/主角立绘写实.png",
        out="../manju/output/ep01_i2v_real",
        prompts=WAN_PROMPTS, shot_prefix="ep01_shot",
        cfg=6.0, blocks=20, vae_tile=272, width=None, height=None),
    "ep02": dict(
        kf="../manju/output/ep02_real",
        ref="../manju/output/xiaoya_real/小雅立绘写实_face.png",
        out="../manju/output/ep02_i2v_real",
        prompts=EP02_MOTION, shot_prefix="ep02_shot",
        cfg=4.0, blocks=25, vae_tile=192, width=480, height=832),
}


def latest(pat):
    fs = glob.glob(pat)
    return max(fs, key=os.path.getmtime) if fs else None


def free_memory(server):
    """让 ComfyUI 释放显存(POST /free)——逐镜间清场, 替代逐镜重启(5B 比 14B 轻)。"""
    try:
        req = urllib.request.Request(
            server + "/free",
            data=b'{"unload_models": true, "free_memory": true}',
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=30).read()
    except Exception as e:
        print(f"  [warn] /free 失败: {e}")


def make_test_image(path, size=(512, 768)):
    from PIL import Image
    Image.new("RGB", size, (60, 70, 90)).save(path)
    return path


def probe_dur(mp4):
    try:
        out = subprocess.check_output(
            [FFPROBE, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", mp4],
            text=True).strip()
        return float(out)
    except Exception:
        return 0.0


def probe_img_dims(path):
    """读图像 w/h。Wan i2v 要求 image dims == gen width/height, 不等则 WanVideoSampler
    注入图像 latent 时 shape mismatch (nodes_sampler.py:775)。竖屏关键帧 512×912 等
    非 768 高度会让写死 512×768 的 run_one 崩, 故自动取图像原生尺寸(已 ÷16 对齐)。"""
    try:
        out = subprocess.check_output(
            [FFPROBE, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "default=nw=1:nk=1", path],
            text=True).strip().split()
        return int(out[0]), int(out[1])
    except Exception:
        return None, None


def extract_mid_frame(mp4, out_png):
    dur = probe_dur(mp4) or 5.0
    subprocess.run([FFMPEG, "-y", "-i", mp4, "-ss", f"{dur / 2:.2f}",
                    "-frames:v", "1", out_png], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def face_cosine_fit(a, b):
    code = ("import sys; sys.path.insert(0,'scripts'); from face_sim import face_cosine as f; "
            "print(f(r'%s', r'%s'))" % (a, b))
    out = subprocess.check_output([FIT_PY, "-c", code], cwd="../manju",
                                  text=True, stderr=subprocess.DEVNULL).strip().splitlines()[-1]
    return None if out == "None" else float(out)


def run_one(c, src_img, shot, prompt, length, steps, cfg, blocks, server, skip_gate,
            width=None, height=None, vae_tile=272):
    upload_img = src_img
    iw, ih = probe_img_dims(src_img)
    if width and height:
        # 显式桶(如 Wan 标准竖屏 480×832): resize 关键帧到精确 W×H。
        # 必要性: 512×912→912÷16=57(奇)触发 transformer addcmul off-by-one
        # (model.py:993, 9408 vs 9576); Wan 此 wrapper 要 H-token 偶数。
        if (iw, ih) != (width, height):
            upload_img = os.path.join(OUT, f"_resized_shot{shot:02d}_{width}x{height}.png")
            subprocess.run([FFMPEG, "-y", "-i", src_img, "-vf", f"scale={width}:{height}",
                            upload_img], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        width, height = iw, ih   # 默认用原生 (image dims==gen dims)
    img = c.upload_image(upload_img)
    prefix = f"manju_wan_shot{shot:02d}" if shot else "manju_wan_smoke"
    wf = build_wan_i2v_workflow(
        img, prompt, WAN_NEG, width=width, height=height,
        length=length, steps=steps, cfg=cfg, blocks_to_swap=blocks,
        vae_tile=vae_tile, filename_prefix=prefix)
    print(f"[wan shot {shot}] {width}×{height} len={length}@16≈{length/16:.1f}s steps={steps} blocks={blocks}",
          flush=True)
    pid = c.queue_prompt(wf)
    c.wait_for_completion(pid, timeout=2400)
    time.sleep(1)
    mp4 = latest(os.path.join(COMFY_OUT, f"{prefix}_*.mp4"))
    if not mp4:
        print(f"  [ERR] ComfyUI 没 mp4: {prefix}")
        return None
    print(f"  mp4={os.path.basename(mp4)} dur={probe_dur(mp4):.2f}s", flush=True)

    dst = os.path.join(OUT, os.path.basename(mp4))
    if skip_gate:
        shutil.copy(mp4, dst)
        print(f"  [skip-gate] -> {dst}")
        return dst
    mid_png = os.path.join(OUT, f"_mid_shot{shot:02d}.png")
    extract_mid_frame(mp4, mid_png)
    sim = face_cosine_fit(REF, mid_png)
    print(f"  face_sim(mid)={sim}", flush=True)
    if sim is None or sim < THRESH:
        print(f"  [WARN] 命门卡2 未过(<{THRESH}), 需重抽")
        return None
    shutil.copy(mp4, dst)
    print(f"  [OK] face_sim={sim:.3f} -> {dst}", flush=True)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="ep01", choices=list(EPISODES),
                    help="选 episode 配置(ep02=小雅写实 WSL Wan i2v); 决定 KF/REF/OUT/prompts+Wan 默认配置")
    ap.add_argument("--shot", type=int, default=None)
    ap.add_argument("--shots", default=None, help="逗号镜头表(如 5,6,7,8,9,10); 优先于 --shot/全量, 便断点续跑")
    ap.add_argument("--length", type=int, default=81)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=None, help="覆盖(默认取 episode 配置)")
    ap.add_argument("--blocks", type=int, default=None, help="覆盖(默认取 episode 配置)")
    ap.add_argument("--smoke", action="store_true", help="烟雾测试: 小配置+占位图, 跳命门")
    ap.add_argument("--skip-gate", action="store_true", help="跳过 face_sim 命门(验证/调试用, 直接存 mp4)")
    ap.add_argument("--width", type=int, default=None, help="覆盖宽(默认 episode; None=关键帧原生)")
    ap.add_argument("--height", type=int, default=None, help="覆盖高(默认 episode; None=关键帧原生)")
    ap.add_argument("--vae-tile", type=int, default=None, help="VAE tile 边长(默认 episode; 解码 OOM 调小)")
    ap.add_argument("--server", default="http://127.0.0.1:8188",
                    help="ComfyUI(ep02 Wan i2v 跑 WSL 用 http://127.0.0.1:8189)")
    args = ap.parse_args()

    global KF, REF, OUT
    ep = EPISODES[args.episode]
    KF, REF, OUT = ep["kf"], ep["ref"], ep["out"]
    prompts, pfx = ep["prompts"], ep["shot_prefix"]
    cfg = args.cfg if args.cfg is not None else ep["cfg"]
    blocks = args.blocks if args.blocks is not None else ep["blocks"]
    vae_tile = args.vae_tile if args.vae_tile is not None else ep["vae_tile"]
    width = args.width if args.width is not None else ep["width"]
    height = args.height if args.height is not None else ep["height"]
    os.makedirs(OUT, exist_ok=True)
    c = ComfyUIClient(args.server)
    print(f"[ep] {args.episode}  KF={KF}\n     REF={REF}\n     OUT={OUT}\n     "
          f"cfg={cfg} blocks={blocks} tile={vae_tile} {width or 'native'}×{height or 'native'} "
          f"len={args.length}@16 steps={args.steps}\n", flush=True)

    if args.smoke:
        test_img = os.path.join(OUT, "_smoke_in.png")
        # 崩点 (model.py:3336): block.to(offload_device) 搬 block 回 CPU, 由 blocks_to_swap>0 触发。
        # Windows+torch 任何版本 .to(cpu) 都崩(access violation 2.7.1 / segfault ≥2.9)。
        # smoke 用 episode blocks——验证 block_swap 搬运路径本身不崩。
        #   Windows: blocks>0 必崩 → 历史传 --blocks 0 绕过(不搬, 5B 全留 GPU 勉强)。
        #   WSL/Linux: .to(cpu) 不崩 → ep02 blocks=25 测生产配置(12GB 真实所需)。
        make_test_image(test_img, size=(480, 640))
        free_memory(args.server)
        r = run_one(c, test_img, shot=0,
                    prompt="a man slowly turns his head, subtle breathing, natural motion",
                    length=9, steps=2, cfg=cfg, blocks=blocks,
                    server=args.server, skip_gate=True, width=480, height=640, vae_tile=vae_tile)
        print("\n[smoke]", "PASS — Wan 管线端到端跑通" if r else "FAIL")
        return 0 if r else 1

    if args.shots:
        shots = [int(x) for x in args.shots.split(",") if x.strip()]
    elif args.shot:
        shots = [args.shot]
    else:
        shots = list(range(1, 11))
    ok = []
    generated = False
    for s in shots:
        # 断点续跑: OUT 里已有该镜 mp4(过命门才落盘) → 跳过, 重跑幂等(reaper kill 后续跑不重做已完成的)。
        existing = glob.glob(os.path.join(OUT, f"manju_wan_shot{s:02d}_*.mp4"))
        if existing:
            print(f"[skip] shot {s:02d} 已存在 → {os.path.basename(max(existing, key=os.path.getmtime))}")
            ok.append(s)
            continue
        if generated:
            free_memory(args.server)
        src = os.path.join(KF, f"{pfx}{s:02d}.png")
        if not os.path.exists(src):
            # 规范名缺 → 取该镜最新 take (epNN_shotMM_0000K_.png)
            src_take = latest(os.path.join(KF, f"{pfx}{s:02d}_*.png"))
            if not src_take:
                print(f"[skip] 关键帧缺: {pfx}{s:02d}(.png/_*)")
                continue
            src = src_take
            print(f"[shot {s}] 用 take: {os.path.basename(src)}")
        r = run_one(c, src, s, prompts[s], args.length, args.steps, cfg,
                    blocks, args.server, skip_gate=args.skip_gate,
                    width=width, height=height, vae_tile=vae_tile)
        generated = True
        if r:
            ok.append(s)
    print(f"\n[done] {len(ok)}/{len(shots)} 镜过命门: {ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
