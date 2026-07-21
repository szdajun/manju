"""ep02 漫画版组装(复用 make_ep01 v4 链路, 换 ep02 i2v 片段 + LINES 台词)。

无旁白 MVP(字幕 + xfade + 静音轨); 旁白 MiniMax TTS 后补(要 output/ep02_voice/voice_shotNN.mp3)。
LTX 片段 512×768 → gen_seg_i2v 上采 720×1280 + 字幕 → xfade 链 → 静音轨。

用法: .venv/Scripts/python.exe scripts/make_ep02.py
      .venv/Scripts/python.exe scripts/make_ep02.py --xfade 0.8   # 转场实验
产物: output/ep02_assemble/ep02_v1.mp4
"""
import argparse
import glob
import os
import sys
import tempfile

MANJU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(MANJU, "scripts"))
import make_ep01 as m  # noqa: E402  复用 gen_seg_i2v/render_subtitle/concat_xfade/add_silent_audio/probe_dur
sys.path.insert(0, MANJU)
from lib.ep02_shots import LINES  # noqa: E402

I2V = "../manju/output/ep02_i2v"
OUT = "../manju/output/ep02_assemble"


def latest(pat):
    fs = glob.glob(pat)
    return max(fs, key=os.path.getmtime) if fs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--out", default=os.path.join(OUT, "ep02_v1.mp4"))
    ap.add_argument("--xfade", type=float, default=0.4)
    args = ap.parse_args()

    shots = [int(x) for x in args.shots.split(",") if x.strip()]
    os.makedirs(OUT, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="ep02_assemble_")

    segs, durs = [], []
    for s in shots:
        src = latest(os.path.join(I2V, f"manju_ep02_shot{s:02d}_*.mp4"))
        if not src:
            print(f"[skip] i2v 片段缺: manju_ep02_shot{s:02d}_*.mp4 in {I2V}")
            continue
        sub_png = os.path.join(tmp, f"sub_{s:02d}.png")
        m.render_subtitle(LINES[s], sub_png)
        dst = os.path.join(tmp, f"seg_{s:02d}.mp4")
        m.gen_seg_i2v(src, dst, sub_png)
        segs.append(dst)
        durs.append(m.probe_dur(dst))
        print(f"[shot {s:2d}] {os.path.basename(src)} -> seg {durs[-1]:.2f}s  「{LINES[s][:18]}」")

    if len(segs) < 2:
        print("[ERR] <2 段 i2v 片段, 无法组装")
        return 1

    novid = os.path.join(tmp, "novid.mp4")
    m.concat_xfade(segs, durs, novid, xfade=args.xfade, fps=m.V4_FPS, ts=m.V4_TS)
    m.add_silent_audio(novid, args.out)
    total = m.probe_dur(args.out)
    print(f"\n[saved] {args.out}  {total:.2f}s  ({len(segs)} 镜, xfade={args.xfade}, 无旁白 MVP)")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
