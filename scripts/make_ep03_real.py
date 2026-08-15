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
from lib.storyboard import load_script, shot  # noqa: E402
from lib import music3  # noqa: E402  # 本地 MiniMax-Music3 BGM 生成 (2026-08-15)


def latest(pat):
    fs = glob.glob(pat)
    return max(fs, key=os.path.getmtime) if fs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", default="03", help="集数 (读剧本数据, 默认 03)")
    ap.add_argument("--shots", default=None, help="指定镜 (如 1,2,3; 默认剧本全部)")
    ap.add_argument("--i2v-dir", default=None,
                    help="i2v 片段目录 (默认 output/ep{ep}_i2v_real/_seedance)")
    ap.add_argument("--out", default=None,
                    help="成片输出 (默认 output/ep{ep}_assemble/ep{ep}_real.mp4)")
    ap.add_argument("--xfade", type=float, default=0.4)
    ap.add_argument("--transition", default="fade",
                    help="xfade 转场类型: fade/fadeblack/dissolve/…, 默认 fade")
    # orchestrator 接入: 配音目录可覆盖 (默认 output/ep{ep}_voice)
    ap.add_argument("--voice-dir", default=None, help="配音 mp3 目录 (默认 output/ep{ep}_voice)")
    # 2026-08-15: MiniMax Music 生成的 BGM 垫底混入 (侧链闪避保语音清晰)
    ap.add_argument("--bgm", default=None,
                    help="BGM 音乐文件路径 (MiniMax Music mmx 生成; 垫底混入 + 侧链闪避)")
    ap.add_argument("--bgm-gen", action="store_true",
                    help="本地 MiniMax-Music3 生成 BGM (ComfyUI 起着, 需 E:/music_models 权重)")
    ap.add_argument("--bgm-volume", type=float, default=0.30,
                    help="BGM 音量系数 0-1 (默认 0.30; 想更燃调 0.5, 更轻调 0.15)")
    ap.add_argument("--bgm-caption", default=None,
                    help="--bgm-gen 时 BGM 描述 (默认 lib/music3 健身曲描述)")
    args = ap.parse_args()

    script = load_script(args.ep)
    ep = script["ep"]
    i2v_dir = args.i2v_dir or os.path.join(MANJU, "output", f"ep{ep}_i2v_real", "_seedance")
    voice_dir = args.voice_dir or os.path.join(MANJU, "output", f"ep{ep}_voice")
    out = args.out or os.path.join(MANJU, "output", f"ep{ep}_assemble", f"ep{ep}_real.mp4")

    shots = [int(x) for x in (args.shots or ",".join(script["shots"].keys())).split(",")
             if x.strip()]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = os.path.join(os.path.dirname(os.path.abspath(out)), "_real_tmp")
    os.makedirs(tmp, exist_ok=True)

    segs_av = []
    prefix = f"ep{ep}_shot"
    for s in shots:
        src = latest(os.path.join(i2v_dir, f"{prefix}{s:02d}.mp4"))
        if not src:
            print(f"[ERR] i2v 片段缺: {prefix}{s:02d}.mp4 in {i2v_dir}")
            return 1
        seg_v = os.path.join(tmp, f"seg{s:02d}.mp4")
        if not os.path.exists(seg_v):
            sub_png = os.path.join(tmp, f"sub{s:02d}.png")
            m.render_subtitle(shot(script, s)["line"], sub_png)
            print(f"=== shot {s:2d} i2v -> seg ===", flush=True)
            m.gen_seg_i2v(src, seg_v, sub_png)
        else:
            print(f"=== shot {s:2d} seg cached ===")
        voice = os.path.join(voice_dir, f"voice_shot{s:02d}.mp3")
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
    m.concat_xfade_av(segs_av, durs, out, fps=m.V4_FPS, ts=m.V4_TS,
                      xfade=args.xfade, transition=args.transition)
    total = m.probe_dur(out)
    bgm_path = args.bgm
    if bgm_path is None and args.bgm_gen:
        # 本地 Music3 生成 BGM (时长=成片总长+余量, 混音 atrim 会截)
        print(f"[bgm] 本地生成 BGM ({total + 5:.0f}s, 健身曲) ...", flush=True)
        saved = music3.generate_bgm(
            caption=args.bgm_caption or music3.DEFAULT_BGM_CAPTION,
            seconds=total + 5, seed=42)
        if saved:
            bgm_path = saved[-1]
            print(f"[bgm] 生成完成: {os.path.basename(bgm_path)}", flush=True)
        else:
            print("[bgm] ⚠ 本地生成无输出 (ComfyUI 起着? E:/music_models 权重在?)", flush=True)
    if bgm_path:
        print(f"[bgm] 混入 {os.path.basename(bgm_path)} (侧链闪避) ...", flush=True)
        out_mixed = out + ".bgm.mp4"
        m.mix_bgm(out, bgm_path, out_mixed, bgm_volume=args.bgm_volume)
        os.replace(out_mixed, out)
        total = m.probe_dur(out)
        print(f"[bgm] 完成: 语音清晰, BGM 垫底", flush=True)
    print(f"\n[saved] {out}  {total:.2f}s  ({len(segs_av)} 镜, xfade={args.xfade}, "
          f"{args.transition}, 写实 Seedance + MiniMax 女声"
          + (f" + BGM={os.path.basename(bgm_path)}" if bgm_path else "") + ")")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)