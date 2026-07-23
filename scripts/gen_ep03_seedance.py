"""ep03 写实版 i2v — Seedance 1.0 pro (云端 doubao-seedance-1-0-pro-250528)。

沿用 ep02 写实路径 (gen_ep02_seedance.py), 适配:
  - KF_DIR: output/ep03_real (ep03 关键帧)
  - OUT: output/ep03_i2v_real/_seedance
  - 5 镜 (vs ep02 10 镜), 文件名 ep03_shotNN.mp4
  - I2V_MOTION 来自 lib/ep03_shots, 末尾必带"the misty-blue top stays misty-blue throughout"
    等 3 句不变式 → Seedance i2v 时也持续锁衣服
  - FACE_POS 沿用 ep02 (降头动脸漂 + 薄眼帘/素颜正向)

跑法:
  .venv/Scripts/python.exe scripts/gen_ep03_seedance.py                  # 全 5 镜
  .venv/Scripts/python.exe scripts/gen_ep03_seedance.py --shots 1,3,5    # 指定
"""
import argparse
import glob
import os
import sys

MANJU = "../manju"
sys.path.insert(0, MANJU)
from lib.video_gen import seedance_i2v  # noqa: E402
from lib.ep03_shots import I2V_MOTION  # noqa: E402

KF_DIR = f"{MANJU}/output/ep03_real"
OUT = f"{MANJU}/output/ep03_i2v_real/_seedance"

# FACE_POS 沿用 ep02 (降 Seedance 头动脸漂 + 薄眼帘/素颜正向, 因正向 > 负向)
FACE_POS = (
    ", face stable and centered, minimal head movement, face clearly visible throughout, "
    "bare fresh natural face, no makeup, nude natural lips, no eyeliner, no eyeshadow, "
    "thin smooth natural eyelids, wide open bright alert eyes, clean smooth under-eye area"
)
# ep03 全 5 镜全有脸 → 不需 NO_FACE_SHOTS 排除
NO_FACE_SHOTS = set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="1,2,3,4,5")
    ap.add_argument("--duration", type=int, default=5)
    ap.add_argument("--resolution", default="480p")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    shots = [int(x) for x in args.shots.split(",") if x.strip()]
    ok, skip, fail = [], [], []
    for s in shots:
        out = os.path.join(OUT, f"ep03_shot{s:02d}.mp4")
        if os.path.exists(out):
            print(f"[skip] shot{s:02d} 已存在: {os.path.basename(out)}", flush=True)
            skip.append(s)
            continue
        cands = sorted(glob.glob(os.path.join(KF_DIR, f"ep03_shot{s:02d}_*.png")))
        if not cands:
            print(f"[ERR] shot{s:02d} 关键帧缺: ep03_shot{s:02d}_*.png in {KF_DIR}", flush=True)
            fail.append(s)
            continue
        src = cands[-1]
        motion = I2V_MOTION.get(s)
        if not motion:
            print(f"[ERR] shot{s:02d} 无 motion 意图 (lib/ep03_shots.I2V_MOTION 缺)", flush=True)
            fail.append(s)
            continue
        prompt = motion if s in NO_FACE_SHOTS else motion + FACE_POS
        print(f"[shot{s:02d}] KF={os.path.basename(src)}", flush=True)
        try:
            seedance_i2v(src, prompt, out, duration=args.duration,
                         ratio="9:16", resolution=args.resolution)
            ok.append(s)
        except Exception as e:
            print(f"[FAIL] shot{s:02d}: {e}", flush=True)
            fail.append(s)
    print(f"\n[done] ok={ok} skip={skip} fail={fail}", flush=True)
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())