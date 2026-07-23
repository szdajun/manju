"""ep03 写实版组装+配音 (2026-07-23, 沿用 make_ep02_real.py 模板)。

5 镜 Seedance i2v (480×864, 5.04s@24fps) → gen_seg_i2v 上采 720×1280 + 字幕
→ mux_seg_audio 逐镜配音 (MiniMax 女声 voice_shotNN.mp3)
→ concat_xfade_av (视频 xfade + 音频 acrossfade)。

vs ep02 差异:
  - 5 镜 (vs 10), 全部单人林夏
  - 文件名前缀 ep03_shotNN.mp4 (ep02_shotNN.mp4)
  - 配音 voice_ep03 跑通, 台词 2.25-3.27s < 5.04s 镜, 静音尾 1.8-2.8s (戏剧停顿)
  - 不含 MuseTalk (本次 ep03 不做唇同步, 沿 ep02 v10 之前的 v3 范式纯配音)

用法: .venv/Scripts/python.exe scripts/make_ep03_real.py
      .venv/Scripts/python.exe scripts/make_ep03_real.py --xfade 0.5
产物: output/ep03_assemble/ep03_real.mp4
"""
import argparse
import glob
import os
import sys

MANJU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(MANJU, "scripts"))
import make_ep01 as m  # noqa: E402  gen_seg_i2v/render_subtitle/mux_seg_audio/concat_xfade_av/probe_dur
sys.path.insert(0, MANJU)
from lib.ep03_shots import LINES  # noqa: E402

I2V = "../manju/output/ep03_i2v_real/_seedance"
VOICE = "../manju/output/ep03_voice"
OUT = "../manju/output/ep03_assemble"


def latest(pat):
    fs = glob.glob(pat)
    return max(fs, key=os.path.getmtime) if fs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="1,2,3,4,5")
    ap.add_argument("--i2v-dir", default=I2V)
    ap.add_argument("--out", default=os.path.join(OUT, "ep03_real.mp4"))
    ap.add_argument("--xfade", type=float, default=0.4)
    ap.add_argument("--transition", default="fade",
                    help="xfade 转场类型: fade/fadeblack/dissolve/…, 默认 fade")
    args = ap.parse_args()

    shots = [int(x) for x in args.shots.split(",") if x.strip()]
    os.makedirs(OUT, exist_ok=True)
    tmp = os.path.join(OUT, "_real_tmp")
    os.makedirs(tmp, exist_ok=True)

    segs_av = []
    for s in shots:
        src = latest(os.path.join(args.i2v_dir, f"ep03_shot{s:02d}.mp4"))
        if not src:
            print(f"[ERR] i2v 片段缺: ep03_shot{s:02d}.mp4 in {args.i2v_dir}")
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
    print(f"[durs] {[round(d, 2) for d in durs]}  sum={sum(durs):.2f}")
    m.concat_xfade_av(segs_av, durs, args.out, fps=m.V4_FPS, ts=m.V4_TS,
                      xfade=args.xfade, transition=args.transition)
    total = m.probe_dur(args.out)
    print(f"\n[saved] {args.out}  {total:.2f}s  ({len(segs_av)} 镜, xfade={args.xfade}, "
          f"{args.transition}, 写实 Seedance + MiniMax 女声)")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)