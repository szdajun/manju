"""ep02 漫画版 LTX i2v 全 10 镜(本地, 零 token)。

复用 gen_i2v.py LTX 链路, 换 ep02 剧本(lib/ep02_shots.I2V_MOTION) + ep02 关键帧源 + FACE_STABLE 后缀
(ep01 教训#1 降 i2v 脸漂) + face_sim 门。漫画 LTX 历史 face_sim 持 0.67-0.70; 门 mid≥0.40(ep02 新角首验)。

v2 脸形变修复(2026-07-21, 用户报"脸变形可怕", 诊断 = i2v 层 shot4/10 最差):
  - cfg 3.0→2.5 / steps 30→40 降形变幅度
  - FACE_STABLE 加强(同人同脸/对称/no morph) + FACE_NEG(face morphing/distortion/warp)
  - 多 seed 候选取 face_sim 最高(LTX 脸漂高度 seed 相关, 最有效单招)
  - 旧版备份 _old/ 供 before/after 对比; 候选落 candidates/

用法:
    .venv/Scripts/python.exe scripts/gen_ep02_i2v.py --shots 3       # 单镜
    .venv/Scripts/python.exe scripts/gen_ep02_i2v.py                  # 全 10 镜
    .venv/Scripts/python.exe scripts/gen_ep02_i2v.py --shots 4,10 --seeds 3  # pilot 多 seed 取优
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import time

MANJU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MANJU)
from lib.comfyui_client import ComfyUIClient, build_ltx_i2v_workflow  # noqa: E402
from lib.ep02_shots import I2V_MOTION, NEG  # noqa: E402

KF = "../manju/output/ep02"
REF = "../manju/assets/characters/小雅立绘.png"
COMFY_OUT = "../ComfyUI/output"
OUT = "../manju/output/ep02_i2v"
FIT_PY = "../fitness-video-pipeline/.venv/Scripts/python.exe"
FFMPEG = "~/ffmpeg/ffmpeg.exe"
FFPROBE = "~/ffmpeg/ffprobe.exe"
THRESH = 0.40  # 漫画 LTX 松阈(ep02 新角首验; 漫画历史 0.67-0.70)
FACE_STABLE = (", face stable, identical face throughout, consistent facial features, symmetrical face, "
               "face centered, minimal head movement, face clearly visible, no face morphing, no face distortion")
FACE_NEG = ", face morphing, face distortion, warped face, melting face, asymmetrical eyes, deformed face"
NO_FACE_SHOTS = {6}  # shot6 手部特写无主角脸 → skip face_sim 门 (shot8 改小雅为主 2026-07-21, 有脸不 skip)


def latest(pat):
    fs = glob.glob(pat)
    return max(fs, key=os.path.getmtime) if fs else None


def probe_dur(mp4):
    try:
        return float(subprocess.check_output(
            [FFPROBE, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", mp4],
            text=True).strip())
    except Exception:
        return 5.0


def face_sim_mid(mp4):
    """抽中帧 vs 小雅立绘 face_sim(fitness venv 子进程调 face_sim.py)。"""
    mid_png = mp4.replace(".mp4", "_mid.png")
    dur = probe_dur(mp4)
    subprocess.run([FFMPEG, "-y", "-i", mp4, "-ss", f"{dur / 2:.2f}", "-frames:v", "1", mid_png],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        out = subprocess.check_output(
            [FIT_PY, "../manju/scripts/face_sim.py", REF, mid_png],
            cwd="../manju", text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    m = re.search(r"cosine_similarity\s*=\s*([\d.]+)", out)
    return float(m.group(1)) if m else None


def run_one(c, shot, length, steps, cfg, width, height, seed):
    """跑单个 seed 候选, 返回 (cand_path, sim)。sim 仅 face 镜测, 无主角脸镜返 None。"""
    src = latest(os.path.join(KF, f"ep02_shot{shot:02d}_*.png"))
    if not src:
        return None, None
    positive = I2V_MOTION[shot] + FACE_STABLE
    negative = NEG + FACE_NEG
    cand_prefix = f"manju_ep02_shot{shot:02d}_cand"
    img = c.upload_image(src)
    wf = build_ltx_i2v_workflow(
        img, positive, negative, width=width, height=height,
        length=length, steps=steps, cfg=cfg, seed=seed,
        frame_rate=16, filename_prefix=cand_prefix)
    pid = c.queue_prompt(wf)
    c.wait_for_completion(pid, timeout=900)
    time.sleep(1)
    mp4 = latest(os.path.join(COMFY_OUT, f"{cand_prefix}_*.mp4"))
    if not mp4:
        return None, None
    cand_dir = os.path.join(OUT, "candidates")
    os.makedirs(cand_dir, exist_ok=True)
    cand = os.path.join(cand_dir, f"shot{shot:02d}_seed{seed}.mp4")
    shutil.copy(mp4, cand)
    sim = face_sim_mid(cand) if shot not in NO_FACE_SHOTS else None
    return cand, sim


def run_shot(c, shot, length, steps, cfg, width, height, seeds=1):
    """跑 n=seeds 个候选(face 镜), 取 face_sim 最高留 _best.mp4; 无主角脸镜跑 1 个。
    旧版先备份到 _old/ 供 before/after 对比。"""
    src = latest(os.path.join(KF, f"ep02_shot{shot:02d}_*.png"))
    if not src:
        print(f"[skip] 关键帧缺: ep02_shot{shot:02d}_*.png in {KF}")
        return None
    old_dir = os.path.join(OUT, "_old")
    os.makedirs(old_dir, exist_ok=True)
    for o in glob.glob(os.path.join(OUT, f"manju_ep02_shot{shot:02d}_*.mp4")):
        shutil.copy(o, os.path.join(old_dir, f"shot{shot:02d}_old.mp4"))
        os.remove(o)
    n = 1 if shot in NO_FACE_SHOTS else max(1, seeds)
    print(f"[i2v shot {shot}] len={length}@16≈{length / 16:.1f}s steps={steps} cfg={cfg} seeds={n}", flush=True)
    cands = []
    for i in range(n):
        seed = int.from_bytes(os.urandom(4), "big") % (2**32)
        cand, sim = run_one(c, shot, length, steps, cfg, width, height, seed)
        if not cand:
            continue
        if sim is None:
            print(f"  [cand {i+1}/{n}] seed={seed} -> {os.path.basename(cand)} [skip-gate 无主角脸]", flush=True)
        else:
            flag = "[OK]" if sim >= THRESH else "[WARN]"
            print(f"  [cand {i+1}/{n}] seed={seed} face_sim={sim} {flag}", flush=True)
        cands.append((cand, sim, seed))
    if not cands:
        print(f"  [ERR] shot{shot} 无候选产出", flush=True)
        return None
    best = cands[0] if shot in NO_FACE_SHOTS else max(cands, key=lambda x: x[1] if x[1] is not None else -1)
    os.makedirs(OUT, exist_ok=True)
    best_dst = os.path.join(OUT, f"manju_ep02_shot{shot:02d}_best.mp4")
    shutil.copy(best[0], best_dst)
    os.utime(best_dst, None)  # 最新 mtime → make_ep02 latest() 命中
    print(f"  [BEST] seed={best[2]} sim={best[1]} -> {os.path.basename(best_dst)}", flush=True)
    return best_dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--length", type=int, default=41)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--cfg", type=float, default=2.5)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    args = ap.parse_args()

    shots = [int(x) for x in args.shots.split(",") if x.strip()]
    c = ComfyUIClient(args.server)
    ok = []
    for s in shots:
        try:
            r = run_shot(c, s, args.length, args.steps, args.cfg, args.width, args.height, seeds=args.seeds)
            if r:
                ok.append(s)
        except Exception as e:
            print(f"[shot {s}] FAIL: {e}")
    print(f"\n[done] {len(ok)}/{len(shots)} 镜: {ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
