"""ep02 写实版组装+配音(2026-07-22, 复用 make_ep02_v3 / make_ep01 v4 链)。

写实 Wan i2v 片段(480×832, 5.06s@16fps) → gen_seg_i2v 上采 720×1280 + 字幕
→ mux_seg_audio 逐镜配音(复用 manga 期 MiniMax 女声 voice_shotNN.mp3, LINES 未变, Wan 镜更长更宽裕)
→ concat_xfade_av(视频 xfade + 音频 acrossfade)。

vs manga v3: i2v 源从 LTX 漫画换 Wan 写实(output/ep02_i2v_real), 脸稳不怕长镜 → 配音可行。
voice mp3 沿用 output/ep02_voice/(LINES 同, speed=1.35 当初按 4.06s 压, 5.06s 窗口更松)。

用法: .venv/Scripts/python.exe scripts/make_ep02_real.py
      .venv/Scripts/python.exe scripts/make_ep02_real.py --xfade 0.5
产物: output/ep02_assemble/ep02_real.mp4
"""
import argparse
import glob
import os
import sys

MANJU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(MANJU, "scripts"))
import make_ep01 as m  # noqa: E402  gen_seg_i2v/render_subtitle/mux_seg_audio/concat_xfade_av/probe_dur
sys.path.insert(0, MANJU)
from lib.ep02_shots import LINES  # noqa: E402

I2V = "../manju/output/ep02_i2v_real"
VOICE = "../manju/output/ep02_voice"
OUT = "../manju/output/ep02_assemble"


def latest(pat):
    fs = glob.glob(pat)
    return max(fs, key=os.path.getmtime) if fs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--i2v-dir", default=I2V,
                    help="i2v 源目录(默认原 i2v; 换脸版传 _facelock)")
    ap.add_argument("--out", default=os.path.join(OUT, "ep02_real.mp4"))
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
        # Wan 写实命名(manju_wan_shotNN_*)优先; 兜底 Seedance 命名(ep02_shotNN.mp4)。
        # 2026-07-23: 换模型族 Seedance rollout 产物在 _seedance/ 下用后者命名。
        src = (latest(os.path.join(args.i2v_dir, f"manju_wan_shot{s:02d}_*.mp4"))
               or latest(os.path.join(args.i2v_dir, f"ep02_shot{s:02d}.mp4")))
        if not src:
            print(f"[ERR] i2v 片段缺: manju_wan_shot{s:02d}_*.mp4 或 ep02_shot{s:02d}.mp4 "
                  f"in {args.i2v_dir}")
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
          f"{args.transition}, 写实 Wan + MiniMax 女声)")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
