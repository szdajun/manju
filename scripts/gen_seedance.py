"""Seedance (火山方舟) i2v 逐镜入口。本地 Wan2.2-5B 死路后转 API。

用法:
  uv run python scripts/gen_seedance.py --smoke          # 占位图验端到端
  uv run python scripts/gen_seedance.py --shot 4         # 单镜(需 ep01_real 关键帧)
  uv run python scripts/gen_seedance.py                  # 全 10 镜
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.video_gen import seedance_i2v  # noqa: E402
from lib.wan_shots import WAN_PROMPTS  # noqa: E402  (prompt 复用)

KF = "../manju/output/ep01_real"
OUT = "../manju/output/ep01_i2v_real"
# face-stable 后缀(ep01 教训#1: Seedance i2v 运镜/头动越大脸漂越凶 — 立绘直吃 i2v 0.76-0.98,
#   经关键帧中转掉到 0.30-0.55; 拼后缀降头动降漂, 对任意分镜 prompt 通用, ep02 起默认带)
FACE_STABLE = ", face stable, face centered, minimal head movement, face clearly visible throughout"


def make_test_image(path, size=(720, 1280)):
    from PIL import Image
    Image.new("RGB", size, (60, 70, 90)).save(path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--shot", type=int, default=None)
    ap.add_argument("--shots", default=None, help="逗号分隔镜号(覆盖 --shot / 全量)")
    ap.add_argument("--duration", type=int, default=5)
    ap.add_argument("--resolution", default="480p")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if args.smoke:
        img = os.path.join(OUT, "_smoke_in.png")
        make_test_image(img)
        out = os.path.join(OUT, "seedance_smoke.mp4")
        try:
            seedance_i2v(
                img,
                "a tired office worker slowly turns his head, subtle breathing, "
                "natural cinematic lighting",
                out, duration=args.duration, ratio="9:16", resolution=args.resolution)
            print("\n[smoke] PASS — Seedance i2v 端到端跑通")
            return 0
        except Exception as e:
            print(f"\n[smoke] FAIL: {e}")
            return 1

    if args.shots:
        shots = [int(x) for x in args.shots.split(",") if x.strip()]
    elif args.shot is not None:
        shots = [args.shot]
    else:
        shots = list(range(1, 11))
    ok = []
    for s in shots:
        # ComfyUI SaveImage 存成 ep01_shotNN_00001_.png(带 batch index 后缀), glob 取最新重抽
        cands = sorted(glob.glob(os.path.join(KF, f"ep01_shot{s:02d}_*.png")))
        if not cands:
            print(f"[skip] 关键帧缺: ep01_shot{s:02d}_*.png in {KF}")
            continue
        src = cands[-1]
        out = os.path.join(OUT, f"ep01_shot{s:02d}.mp4")
        try:
            prompt = WAN_PROMPTS[s] + FACE_STABLE  # 拼 face-stable 后缀降 i2v 脸漂(ep01 教训#1)
            seedance_i2v(src, prompt, out, duration=args.duration,
                         ratio="9:16", resolution=args.resolution)
            ok.append(s)
        except Exception as e:
            print(f"[shot {s}] FAIL: {e}")
    print(f"\n[done] {len(ok)}/{len(shots)} 镜: {ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
