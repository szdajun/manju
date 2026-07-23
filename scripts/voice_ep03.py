"""ep03 单人 TTS — 林夏一人 5 镜。

沿用 ep02 模板 (voice_ep02.py), 单角色:
  - LINES 来自 lib/ep03_shots
  - SPEAKER 5 镜全 LINXIA (单一女声)
  - I2V 路径用 ep03 seedance (output/ep03_i2v_real/_seedance)
  - 台词短 (12-23 字), 默认 speed=1.0 (MiniMax 默认偏快, 不用 ep02 1.35)

用法: .venv/Scripts/python.exe scripts/voice_ep03.py
"""
import argparse
import os
import subprocess
import sys

MANJU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MANJU)

# 强制 MiniMax (沿用 ep02 拍板)
os.environ.setdefault("MANJU_TTS", "minimax")

from lib.tts import synthesize, PROVIDER  # noqa: E402
from lib.ep03_shots import LINES, SPEAKER  # noqa: E402

# I2V 路径用 ep03 seedance (5s mp4)
I2V = "../manju/output/ep03_i2v_real/_seedance"
OUT = "../manju/output/ep03_voice"
FFPROBE = "~/ffmpeg/ffprobe.exe"
MARGIN = 0.5

# MiniMax speech-02-hd 女声 (ep03 单人林夏, 选 female-shaonv 少女活泼)
LINXIA = "female-shaonv"


def dur(path):
    try:
        o = subprocess.check_output(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path]).decode().strip()
        return float(o)
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="1,2,3,4,5")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="MiniMax speed (ep03 台词短, 1.0 自然; ep02 1.35 是给长台词)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    shots = [int(x) for x in args.shots.split(",") if x.strip()]
    print(f"[provider] {PROVIDER}  speed={args.speed}\n")
    print(f"{'shot':>4} {'voice_s':>8} {'seg_s':>7} {'flag':>6}  台词")
    print("-" * 60)
    results = []
    for s in shots:
        line = LINES[s]
        spk = SPEAKER[s]
        out = os.path.join(OUT, f"voice_shot{s:02d}.mp3")
        # seg 时长从 i2v mp4 取 (5s 镜)
        seg = dur(os.path.join(I2V, f"ep03_shot{s:02d}.mp4"))
        try:
            synthesize(line, out, voice=LINXIA, speed=args.speed)
        except Exception as e:
            print(f"{s:>4}  FAIL: {e}")
            results.append((s, "FAIL"))
            continue
        vd = dur(out)
        flag = "OK" if vd <= seg - MARGIN else ("FIT" if vd <= seg else "OVER")
        print(f"{s:>4} {vd:>8.2f} {seg:>7.2f} {flag:>6}  {line}")
        results.append((s, flag))
    overs = [r for r in results if r[1] == "OVER"]
    fails = [r for r in results if r[1] == "FAIL"]
    print("-" * 60)
    print(f"[done] {len(results)-len(fails)}/{len(results)} 镜 -> {OUT}/  "
          f"(OVER={len(overs)} FAIL={len(fails)})")


if __name__ == "__main__":
    main()