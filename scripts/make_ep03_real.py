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
import subprocess
import sys

MANJU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(MANJU, "scripts"))
import make_ep01 as m  # noqa: E402  gen_seg_i2v/render_subtitle/mux_seg_audio/concat_xfade_av/probe_dur
sys.path.insert(0, MANJU)
from lib.storyboard import clean_line, load_script, shot  # noqa: E402
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
    # 2026-08-17 后处理风格库 (spec docs/superpowers/specs/2026-08-17-post-enhance-design.md)
    ap.add_argument("--sub-style", default=None,
                    choices=["classic", "clean", "variety", "minimal"],
                    help="字幕风格 (默认剧本 post.sub_style 或 classic 黄字)")
    ap.add_argument("--transition-mode", default=None, choices=["auto", "fade"],
                    help="auto=按镜头节奏查表 / fade=全 fade (默认, 现行为)")
    ap.add_argument("--bgm-preset", default=None,
                    choices=["pump", "funny", "warm", "chill"],
                    help="BGM 调性预设 (--bgm 显式路径优先级更高)")
    ap.add_argument("--cover", action="store_true",
                    help="出封面 ep{ep}_cover.png (关键帧+title+角标)")
    ap.add_argument("--cover-shot", type=int, default=None,
                    help="封面用哪镜关键帧 (默认 shot01)")
    # orchestrator 接入: 配音目录可覆盖 (默认 output/ep{ep}_voice)
    ap.add_argument("--voice-dir", default=None, help="配音 mp3 目录 (默认 output/ep{ep}_voice)")
    # 2026-08-15: MiniMax Music 生成的 BGM 垫底混入 (侧链闪避保语音清晰)
    ap.add_argument("--bgm", default=None,
                    help="BGM 音乐文件路径 (MiniMax Music mmx 生成; 垫底混入 + 侧链闪避)")
    ap.add_argument("--bgm-gen", action="store_true",
                    help="本地 MiniMax-Music3 生成 BGM (ComfyUI 起着, 需 E:/music_models 权重)")
    ap.add_argument("--bgm-volume", type=float, default=None,
                    help="BGM 音量系数 0-1 (默认 preset 音量或 0.30; 想更燃调 0.5, 更轻调 0.15)")
    ap.add_argument("--bgm-caption", default=None,
                    help="--bgm-gen 时 BGM 描述 (默认 lib/music3 健身曲描述)")
    args = ap.parse_args()

    script = load_script(args.ep)
    ep = script["ep"]
    i2v_dir = args.i2v_dir or os.path.join(MANJU, "output", f"ep{ep}_i2v_real", "_seedance")
    voice_dir = args.voice_dir or os.path.join(MANJU, "output", f"ep{ep}_voice")
    out = args.out or os.path.join(MANJU, "output", f"ep{ep}_assemble", f"ep{ep}_real.mp4")

    # 2026-08-17 后处理风格库: 三级优先 (剧本 post 字段 > CLI > 默认)
    from lib.post_styles import (
        pick_transitions, resolve_bgm, resolve_post_opts, bgm_cache_path)
    post = resolve_post_opts(script, sub_style=args.sub_style,
                             transition_mode=args.transition_mode,
                             bgm_preset=args.bgm_preset,
                             cover=args.cover, cover_shot=args.cover_shot)
    print(f"[post] sub={post['sub_style']} transition={post['transition_mode']} "
          f"bgm={post['bgm_preset'] or '-'} cover={post['cover']}")

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
        # 2026-08-18 fix: tmp 缓存名带 sub_style — 否则同目录切风格重跑静默沿用旧字幕
        seg_v = os.path.join(tmp, f"seg{s:02d}_{post['sub_style']}.mp4")
        # 2026-08-20 fix: i2v 重生成后旧 seg 缓存必须失效 — 否则重组装静默沿用旧画面
        # +旧音轨 (ep26 镜2 说话人修复被缓存吞掉, 用户二次反馈才抓到)
        if (not os.path.exists(seg_v)
                or os.path.getmtime(src) > os.path.getmtime(seg_v)):
            sub_png = os.path.join(tmp, f"sub{s:02d}_{post['sub_style']}.png")
            m.render_subtitle(clean_line(shot(script, s)["line"]), sub_png,
                              style=post["sub_style"])
            print(f"=== shot {s:2d} i2v -> seg ===", flush=True)
            m.gen_seg_i2v(src, seg_v, sub_png)
        else:
            print(f"=== shot {s:2d} seg cached ===")
        voice = os.path.join(voice_dir, f"voice_shot{s:02d}.mp3")
        dst_av = os.path.join(tmp, f"av{s:02d}_{post['sub_style']}.mp4")
        tag = os.path.basename(voice) if os.path.exists(voice) else "(silent)"
        # 同理: seg 或 voice 比旧 av 新 → 重 mux
        _av_stale = False
        if os.path.exists(dst_av):
            if os.path.getmtime(seg_v) > os.path.getmtime(dst_av):
                _av_stale = True
            if os.path.exists(voice) and os.path.getmtime(voice) > os.path.getmtime(dst_av):
                _av_stale = True
        if not os.path.exists(dst_av) or _av_stale:
            print(f"  mux shot {s:2d}  {tag}")
            m.mux_seg_audio(seg_v, voice, dst_av)
        else:
            print(f"  mux shot {s:2d}  cached")
        segs_av.append(dst_av)

    durs = [m.probe_dur(s) for s in segs_av]
    print(f"[durs] {[round(d, 2) for d in durs]}  sum={sum(durs):.2f}")
    transitions = None
    if post["transition_mode"] == "auto":
        # 2026-08-18 fix: 按选中镜构造子剧本 — 否则 --shots 非前缀子集时转场按全镜错位
        sub_script = {"shots": {str(s): script["shots"][str(s)] for s in shots}}
        transitions = pick_transitions(sub_script)
    m.concat_xfade_av(segs_av, durs, out, fps=m.V4_FPS, ts=m.V4_TS,
                      xfade=args.xfade, transition=args.transition,
                      transitions=transitions)
    total = m.probe_dur(out)
    bgm_path = args.bgm
    bgm_preset_cfg = (resolve_bgm(post["bgm_preset"])
                      if post["bgm_preset"] is not None else None)
    if bgm_path is None and post["bgm_preset"] is not None:
        p = bgm_preset_cfg
        cache = bgm_cache_path(post["bgm_preset"], 42, os.path.dirname(out))
        if os.path.exists(cache):
            bgm_path = cache
            print(f"[bgm] 缓存命中 {os.path.basename(cache)}")
        else:
            print(f"[bgm] preset={post['bgm_preset']} 生成 ...", flush=True)
            saved = music3.generate_bgm(caption=p["caption"],
                                        seconds=total + 5, seed=42)
            if saved:
                import shutil as _sh
                _sh.copyfile(saved[-1], cache)
                bgm_path = cache
            else:
                print("[bgm] WARN preset 生成无输出 (ComfyUI 起着? E:/music_models 权重在?)",
                      flush=True)
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
        # 音量三级: --bgm-volume 显式 > preset 音量 > 0.30 (旧默认)
        vol = (args.bgm_volume if args.bgm_volume is not None
               else (bgm_preset_cfg["volume"] if bgm_preset_cfg else 0.30))
        print(f"[bgm] 混入 {os.path.basename(bgm_path)} (侧链闪避, vol={vol:.2f}) ...", flush=True)
        out_mixed = out + ".bgm.mp4"
        m.mix_bgm(out, bgm_path, out_mixed, bgm_volume=vol)
        os.replace(out_mixed, out)
        total = m.probe_dur(out)
        print(f"[bgm] 完成: 语音清晰, BGM 垫底", flush=True)
    if post["cover"]:
        from lib.post_styles import render_cover
        # kf 目录 = orchestrator default_out_dirs["kf"] (本脚本此前不 import kf 目录)
        dirs_kf = os.path.join(MANJU, "output", f"ep{ep}_real")
        cover_out = os.path.join(os.path.dirname(out), f"ep{ep}_cover.png")
        # 底图优先 i2v 视频抽帧 (SDXL kf 塑料感重/头身比易崩, 2026-08-18 用户反馈)
        cover_sid = 1 if post["cover_shot"] is None else post["cover_shot"]
        frame_src = os.path.join(i2v_dir, f"ep{ep}_shot{cover_sid:02d}.mp4")
        try:
            render_cover(script, dirs_kf, cover_out, shot_id=post["cover_shot"],
                         frame_src=frame_src)
            print(f"[cover] {cover_out}")
            # 2026-08-18: 封面进片头 — 1.2s 定帧卡 (fade in/out) 前接成片,
            # 平台取首帧即封面; 音频延 cover时长 (BGM 已含, anullsrc 补静音)。
            cover_d = 1.2
            out_cov = out + ".cover.mp4"
            r = subprocess.run(
                [m.FFMPEG, "-y", "-loop", "1", "-t", f"{cover_d}", "-i", cover_out,
                 "-i", out,
                 "-filter_complex",
                 f"[0:v]scale={m.W}:{m.H}:force_original_aspect_ratio=increase,"
                 f"crop={m.W}:{m.H},fps={m.V4_FPS},format=yuv420p,"
                 f"fade=t=in:st=0:d=0.3[cf];"
                 # 两输入都过 fps → timebase 一致 (xfade 要求 1/24=1/24,
                 # 成片原生 1/24576 直接接 xfade 会 Invalid argument)
                 f"[1:v]fps={m.V4_FPS},fade=t=in:st=0:d=0.3[vf];"
                 f"[cf][vf]xfade=transition=fade:duration=0.3:offset={cover_d - 0.3}[v];"
                 f"[1:a]adelay={int(cover_d * 1000)}|{int(cover_d * 1000)},"
                 f"apad[a]",
                 "-map", "[v]", "-map", "[a]",
                 *m.enc_args(fps=m.V4_FPS, ts=m.V4_TS),
                 "-c:a", "aac", "-b:a", "192k",
                 "-t", f"{total + cover_d:.3f}",
                 out_cov],
                capture_output=True, text=True)
            if r.returncode != 0:
                print(f"[cover] WARN 片头失败, 用无封面版: {r.stderr[-300:]}")
            else:
                os.replace(out_cov, out)
                total = m.probe_dur(out)
                print(f"[cover] 片头封面卡 {cover_d}s 已前接 (总长 {total:.2f}s)")
        except Exception as e:  # 2026-08-18 fix: 宽捕获, 封面失败不阻塞成片
            print(f"[cover] WARN 跳过: {e}")
    print(f"\n[saved] {out}  {total:.2f}s  ({len(segs_av)} 镜, xfade={args.xfade}, "
          f"{args.transition}, 写实 Seedance + MiniMax 女声"
          + (f" + BGM={os.path.basename(bgm_path)}" if bgm_path else "") + ")")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)