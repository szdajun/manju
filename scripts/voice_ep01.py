"""逐镜 TTS 旁白 → output/ep01_voice/voice_shotNN.mp3 + 时长核对。

复用 make_ep01.SHOTS 台词。每镜旁白时长 vs 视频段时长, 超(段长 - 0.6s 余量)标 [OVER] ——
超时意味着旁白溢出该镜窗口, v3 mux 时要么压 pace 要么该镜加长。

用法: python scripts/voice_ep01.py [--shots 1,2,3]
"""
import argparse
import os
import subprocess
import sys

MANJU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MANJU)
sys.path.insert(0, os.path.join(MANJU, "scripts"))

from lib.tts import synthesize, PROVIDER  # noqa: E402
from make_ep01 import SHOTS, FFPROBE  # noqa: E402

OUT = "../manju/output/ep01_voice"
MARGIN = 0.6   # 旁白应 <= 段长 - MARGIN(留 lead-in/tail 静音)


def dur(path):
    o = subprocess.check_output(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path]).decode().strip()
    return float(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default=None, help="逗号分隔镜号(默认全 10 镜)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    shots = ([int(x) for x in args.shots.split(",")] if args.shots
             else [s[0] for s in SHOTS])
    rows = SHOTS  # (shot, dur, kind, src, 台词, kb)

    print(f"[provider] {PROVIDER}\n")
    print(f"{'shot':>4} {'voice_s':>8} {'seg_s':>7} {'margin':>7}  {'flag':>6}  台词")
    print("-" * 78)
    results = []
    for shot, seg_dur, kind, src, line, kb in rows:
        if shot not in shots:
            continue
        out = os.path.join(args.out, f"voice_shot{shot:02d}.mp3")
        synthesize(line, out)
        vd = dur(out)
        margin = seg_dur - vd
        flag = "OK" if vd <= seg_dur - MARGIN else ("FIT" if vd <= seg_dur else "OVER")
        print(f"{shot:>4} {vd:>8.2f} {seg_dur:>7.2f} {margin:>+7.2f}  {flag:>6}  {line}")
        results.append((shot, vd, seg_dur, flag))

    overs = [r for r in results if r[3] == "OVER"]
    fits = [r for r in results if r[3] == "FIT"]
    print("-" * 78)
    print(f"[done] {len(results)} 镜旁白 -> {args.out}/  "
          f"(OVER={len(overs)} FIT={len(fits)})")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
