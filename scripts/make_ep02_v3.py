"""ep02 v3 带声版组装(复用 make_ep01 v4 配音链)。

i2v v3 片段(length=65 ≈4.06s, 对白节奏) → gen_seg_i2v 上采 720×1280 + 字幕
→ mux_seg_audio 逐镜配音(MiniMax 女声, 300ms lead-in + apad 到 seg) → concat_xfade_av(视频xfade+音频acrossfade)。

v2(无旁白 21s) 已存 ep02_v2.mp4; v3 = 同画面风格 + 配音 + 对白节奏(更长镜)。
i2v v3 片段覆盖了 ep02_i2v/manju_ep02_shot\*_best.mp4, v2 备份在 _v2_backup/。

用法: .venv/Scripts/python.exe scripts/make_ep02_v3.py
      .venv/Scripts/python.exe scripts/make_ep02_v3.py --xfade 0.5
产物: output/ep02_assemble/ep02_v3.mp4
"""
import argparse
import glob
import os
import sys
import tempfile

MANJU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(MANJU, "scripts"))
import make_ep01 as m  # noqa: E402  gen_seg_i2v/render_subtitle/mux_seg_audio/concat_xfade_av/probe_dur
sys.path.insert(0, MANJU)
from lib.ep02_shots import LINES  # noqa: E402

I2V = "../manju/output/ep02_i2v"
VOICE = "../manju/output/ep02_voice"
OUT = "../manju/output/ep02_assemble"


def latest(pat):
    fs = glob.glob(pat)
    return max(fs, key=os.path.getmtime) if fs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--out", default=os.path.join(OUT, "ep02_v3.mp4"))
    ap.add_argument("--xfade", type=float, default=0.4)
    args = ap.parse_args()

    shots = [int(x) for x in args.shots.split(",") if x.strip()]
    os.makedirs(OUT, exist_ok=True)
    tmp = os.path.join(OUT, "_v3_tmp")
    os.makedirs(tmp, exist_ok=True)

    segs_av = []
    for s in shots:
        src = latest(os.path.join(I2V, f"manju_ep02_shot{s:02d}_*.mp4"))
        if not src:
            print(f"[ERR] i2v 片段缺: manju_ep02_shot{s:02d}_*.mp4")
            return 1
        seg_v = os.path.join(tmp, f"seg{s:02d}.mp4")
        if not os.path.exists(seg_v):
            sub_png = os.path.join(tmp, f"sub{s:02d}.png")
            m.render_subtitle(LINES[s], sub_png)
            print(f"=== shot {s:2d} i2v -> seg ===", flush=True)
            m.gen_seg_i2v(src, seg_v, sub_png)
        else:
            print(f"=== shot {s:2d} seg cached ===")
        voice = os.path.join(VOICE, f"voice_shot{s:02d}.mp3")
        dst_av = os.path.join(tmp, f"av{s:02d}.mp4")
        tag = os.path.basename(voice) if os.path.exists(voice) else "(silent)"
        if not os.path.exists(dst_av):
            print(f"  mux shot {s:2d}  {tag}")
            m.mux_seg_audio(seg_v, voice, dst_av)
        else:
            print(f"  mux shot {s:2d}  cached")
        segs_av.append(dst_av)

    durs = [m.probe_dur(s) for s in segs_av]
    print(f"[durs] {[round(d,2) for d in durs]}  sum={sum(durs):.2f}")
    m.concat_xfade_av(segs_av, durs, args.out, fps=m.V4_FPS, ts=m.V4_TS, xfade=args.xfade)
    total = m.probe_dur(args.out)
    print(f"\n[saved] {args.out}  {total:.2f}s  ({len(segs_av)} 镜, xfade={args.xfade}, 带声 v3)")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
