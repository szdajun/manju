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

KF = "../manju/output/ep01_real"
REF = "../manju/output/lihui_real/主角立绘写实.png"
COMFY_OUT = "../ComfyUI/output"
OUT = "../manju/output/ep01_i2v_real"
FIT_PY = "../fitness-video-pipeline/.venv/Scripts/python.exe"
FFMPEG = "~/ffmpeg/ffmpeg.exe"
FFPROBE = "~/ffmpeg/ffprobe.exe"
THRESH = 0.45   # Wan clip 命门卡(允许略松于关键帧 0.50)


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


def make_test_image(path):
    from PIL import Image
    Image.new("RGB", (512, 768), (60, 70, 90)).save(path)
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


def run_one(c, src_img, shot, prompt, length, steps, cfg, blocks, server, skip_gate):
    img = c.upload_image(src_img)
    prefix = f"manju_wan_shot{shot:02d}" if shot else "manju_wan_smoke"
    wf = build_wan_i2v_workflow(
        img, prompt, WAN_NEG, width=512, height=768,
        length=length, steps=steps, cfg=cfg, blocks_to_swap=blocks,
        filename_prefix=prefix)
    print(f"[wan shot {shot}] len={length}@16≈{length/16:.1f}s steps={steps} blocks={blocks}",
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
    ap.add_argument("--shot", type=int, default=None)
    ap.add_argument("--length", type=int, default=81)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--blocks", type=int, default=20)
    ap.add_argument("--smoke", action="store_true", help="烟雾测试: 小配置+占位图, 跳命门")
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    c = ComfyUIClient(args.server)

    if args.smoke:
        test_img = os.path.join(OUT, "_smoke_in.png")
        make_test_image(test_img)
        free_memory(args.server)
        r = run_one(c, test_img, shot=0,
                    prompt="a man slowly turns his head, subtle breathing, natural motion",
                    length=9, steps=2, cfg=6.0, blocks=25,
                    server=args.server, skip_gate=True)
        print("\n[smoke]", "PASS — Wan 管线端到端跑通" if r else "FAIL")
        return 0 if r else 1

    shots = [args.shot] if args.shot else list(range(1, 11))
    ok = []
    for i, s in enumerate(shots):
        if i:
            free_memory(args.server)
        src = os.path.join(KF, f"ep01_shot{s:02d}.png")
        if not os.path.exists(src):
            print(f"[skip] 关键帧缺: {src}")
            continue
        r = run_one(c, src, s, WAN_PROMPTS[s], args.length, args.steps, args.cfg,
                    args.blocks, args.server, skip_gate=False)
        if r:
            ok.append(s)
    print(f"\n[done] {len(ok)}/{len(shots)} 镜过命门: {ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
