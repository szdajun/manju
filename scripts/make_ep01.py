"""ep01 成片组装 v1~v4。

v1(rough cut): KB 慢推静态镜 + 3× 慢放动态镜, 硬切, 无字幕无音。
v2: + 字幕(PIL 黄字黑描边, 避 ffmpeg drawtext 8.1 bug)
    + 淡入淡出转场(xfade 0.4s, 链式 offset 精确按各段实测时长)
    + KB 运镜变化(推近 in / 拉远 out / 横摇 pan 交替, 不再全中心推)
    + 静音占位音轨(AAC, TTS 替换时直接换音轨)
v3: = v2 画面 + 真配音(逐镜 TTS 旁白 mux, xfade 视频链 + acrossfade 音频链对齐)。
v4: 写实版(Seedance i2v 8 镜 {1,3,4,5,6,7,8,10})。片段已是动态视频(5s@24fps),
    不 KB 不慢放, 仅 scale+crop 到 720×1280 + 字幕; 旁白复用 v3 MiniMax 配音;
    xfade(视频)+acrossfade(音频) 链。源 480×864→720×1280 lanczos 1.5× 上采。

用法:
    python scripts/make_ep01.py --mode v1        # rough cut 纯画面硬切
    python scripts/make_ep01.py --mode v2        # 字幕+转场+KB变化+静音音轨
    python scripts/make_ep01.py --mode v3        # v2 + 真配音(edge-tts 草版/可切 provider)
    python scripts/make_ep01.py --mode v4        # 写实 Seedance 8 镜成片
"""
import argparse
import os
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont

MANJU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if MANJU not in sys.path:  # CLI 直跑 (python scripts/make_ep01.py) 时 lib/ 可导入
    sys.path.insert(0, MANJU)

FFMPEG = "~/ffmpeg/ffmpeg.exe"
FFPROBE = "~/ffmpeg/ffprobe.exe"
EP01 = "../manju/output/ep01"
I2V = "../manju/output/ep01_i2v"
VOICE = "../manju/output/ep01_voice"
OUT = "../manju/output/ep01_assemble"
FONT = "C:/Windows/Fonts/msyhbd.ttc"
LEAD_IN_MS = 300   # 旁白 lead-in 静音(ms), 避免开口贴镜首

FPS = 16
V4_FPS = 24                      # 写实 Seedance 片段原生 24fps, 保流畅不动
W, H = 720, 1280
XFADE = 0.4
TIMESCALE = str(FPS * 1024)
V4_TS = str(V4_FPS * 1024)
ENC = ["-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p",
       "-r", str(FPS), "-video_track_timescale", TIMESCALE,
       "-frames:v", None]  # frames 占位, 调用时填

# (shot, dur, kind, src, 台词, kb_type)  kb_type: in推近 out拉远 pan横摇 (动态镜用'-')
SHOTS = [
    (1,  5.0, "kb",  f"{EP01}/ep01_shot01_00001_.png", "加班到十二点，老板说……健康也很重要。", "in"),
    (2,  6.0, "kb",  f"{EP01}/ep01_shot02_00001_.png", "于是，一个深陷996的牛马，走进了健身房。", "out"),
    (3,  7.0, "kb",  f"{EP01}/ep01_shot03_00001_.png", "第一次，我觉得……我不属于这里。", "pan"),
    (4,  4.7, "dyn", f"{I2V}/manju_i2v_shot04_00002.mp4", "这玩意儿……比我KPI还重。", "-"),
    (5,  4.7, "dyn", f"{I2V}/manju_i2v_shot05_00002.mp4", "再跑五分钟……啊！！", "-"),
    (6,  7.0, "kb",  f"{EP01}/ep01_shot06_00001_.png", "蛋白粉配冰美式，打工人专属。", "in"),
    (7,  4.7, "dyn", f"{I2V}/manju_i2v_shot07_00001.mp4", "核心……要稳……", "-"),
    (8, 10.0, "kb",  f"{EP01}/ep01_shot08_00001_.png", "兄弟你这体能……但我，加过班。", "out"),
    (9, 10.0, "kb",  f"{EP01}/ep01_shot09_00001_.png", "健身……是为了更好地加班吗？", "in"),
    (10, 10.0, "kb", f"{EP01}/ep01_shot10_00001_.png", "我是牛马，但我要做，一个举过铁的牛马。", "in"),
]

# 写实版 v4: Seedance i2v 8 镜(用户挑掉 2 冗余 + 9 弱)。片段统一 5s@24fps,
# 不分 kb/dyn(全是动态视频), 无 kb_type。台词与漫画版同镜一致。
I2V_REAL = "../manju/output/ep01_i2v_real"
SHOTS_REAL = [
    (1,  f"{I2V_REAL}/ep01_shot01.mp4", "加班到十二点，老板说……健康也很重要。"),
    (3,  f"{I2V_REAL}/ep01_shot03.mp4", "第一次，我觉得……我不属于这里。"),
    (4,  f"{I2V_REAL}/ep01_shot04.mp4", "这玩意儿……比我KPI还重。"),
    (5,  f"{I2V_REAL}/ep01_shot05.mp4", "再跑五分钟……啊！！"),
    (6,  f"{I2V_REAL}/ep01_shot06.mp4", "蛋白粉配冰美式，打工人专属。"),
    (7,  f"{I2V_REAL}/ep01_shot07.mp4", "核心……要稳……"),
    (8,  f"{I2V_REAL}/ep01_shot08.mp4", "兄弟你这体能……但我，加过班。"),
    (10, f"{I2V_REAL}/ep01_shot10.mp4", "我是牛马，但我要做，一个举过铁的牛马。"),
]


def run(cmd):
    print("  $", os.path.basename(str(cmd[0])), "...", str(cmd[-1]))
    subprocess.run([str(c) for c in cmd], check=True)


def probe_dur(path):
    out = subprocess.check_output(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=duration", "-of", "default=noprint_wrappers=1:nokey=1",
         path]).decode().strip()
    return float(out)


# ---------------- 字幕渲染(PIL) ----------------
def wrap_text(text, max_chars=9, hard=15):
    """软上限遇标点优先断, 硬上限强制断(防长句中间无标点一行超画布被裁)。

    两条防丑规则:
    - 不劈连续同类标点(……) —— 下一个字符仍是同一标点则推迟到 run 结束再断。
    - 不留孤标点行 —— 剩余 ≤2 字符(只剩句末标点)时并入当前行, 不另起一行。
    """
    pun = "，。！？、… "
    n = len(text)
    lines, cur = [], ""
    for i, ch in enumerate(text):
        cur += ch
        next_same = (i + 1 < n) and (text[i + 1] == ch)
        would_orphan = (n - (i + 1)) <= 2          # 剩下只剩句末标点 → 并入, 不断
        soft = len(cur) >= max_chars and ch in pun and not next_same
        hard_hit = len(cur) >= hard
        if (soft or hard_hit) and not would_orphan:
            lines.append(cur)
            cur = ""
    if cur:
        lines.append(cur)
    return lines


def render_subtitle(text, out, style="classic"):
    """底部字幕 + 半透明底条。720×1280 透明 PNG。

    style: lib.post_styles.SUBTITLE_STYLES key (默认 classic = 原黄字黑边)。
    """
    from lib.post_styles import resolve_style
    st = resolve_style(style)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    lines = wrap_text(text or "")
    if not lines:
        # 空台词 (ambient 镜): 存透明 PNG, 不加字幕
        img.save(out)
        return
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, st["font_size"])
    line_h = int(st["font_size"] * 4 / 3)
    total_h = len(lines) * line_h
    y0 = H - total_h - 120
    widths = [d.textlength(l, font=font) for l in lines]
    maxw = max(widths)
    if st["box"]:
        pad_x, pad_y = 30, 18
        bx0 = (W - maxw) / 2 - pad_x
        by0 = y0 - pad_y
        bx1 = (W + maxw) / 2 + pad_x
        by1 = y0 + total_h + pad_y - 4
        d.rounded_rectangle([bx0, by0, bx1, by1], radius=st["radius"],
                            fill=st["box"])
    for i, (l, w) in enumerate(zip(lines, widths)):
        x = (W - w) / 2
        y = y0 + i * line_h
        d.text((x, y), l, font=font, fill=tuple(st["fill"]),
               stroke_width=st["stroke_w"], stroke_fill=tuple(st["stroke"]))
    img.save(out)


# ---------------- 段生成 ----------------
def kenburns_vf(dur, kb_type):
    n = int(round(dur * FPS))
    scale = (f"scale={W*2}:{H*2}:force_original_aspect_ratio=increase,"
             f"crop={W*2}:{H*2},")
    if kb_type == "in":
        z, xy = f"1+0.12*on/{n}", "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
    elif kb_type == "out":
        z, xy = f"1.12-0.12*on/{n}", "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
    else:  # pan
        z, xy = "1.1", f"x='(iw-iw/zoom)*on/{n}':y='ih/2-(ih/zoom/2)'"
    return (f"{scale}zoompan=z='min(max({z},1),1.15)':{xy}:"
            f"d={n}:s={W}x{H}:fps={FPS}")


def enc_args(fps=FPS, ts=TIMESCALE):
    return ["-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-r", str(fps), "-video_track_timescale", ts]


def gen_seg_kb(src, dst, dur, kb_type, sub_png):
    vf = kenburns_vf(dur, kb_type)
    if sub_png:
        fc = f"[0:v]{vf}[bg];[bg][1:v]overlay=0:0:shortest=1[v]"
        run([FFMPEG, "-y", "-i", src, "-loop", "1", "-i", sub_png,
             "-filter_complex", fc, "-map", "[v]"] + enc_args() + [dst])
    else:
        run([FFMPEG, "-y", "-i", src, "-vf", vf] + enc_args() + [dst])


def gen_seg_dyn(src, dst, dur, sub_png, factor=3.0):
    base = (f"scale={W}:{H}:force_original_aspect_ratio=increase,"
            f"crop={W}:{H},setpts=PTS*{factor},fps={FPS}")
    if sub_png:
        fc = f"[0:v]{base}[bg];[bg][1:v]overlay=0:0:shortest=1[v]"
        run([FFMPEG, "-y", "-i", src, "-loop", "1", "-i", sub_png,
             "-filter_complex", fc, "-map", "[v]"] + enc_args() + [dst])
    else:
        run([FFMPEG, "-y", "-i", src, "-vf", base] + enc_args() + [dst])


def gen_seg_i2v(src, dst, sub_png, fps=V4_FPS, breathe=None):
    """写实 i2v 片段: 已是动态视频(5s@24fps), 不 KB 不慢放。
    scale+crop 到 720×1280(源 480×864 lanczos 1.5× 上采) + 字幕 overlay。fps 保原生流畅。

    ⚠ 2026-08-06 端到端整合: `-map 0:a?` 保留源音频 (H3 原生音: 对话语音/环境音).
      历史 -map [v] 只映射视频 → H3 自带音到组装全丢 (端到端根因 bug, 已修).
      `0:a?` 可选映射: 无音源 (seedance/ltx) 自动跳过, 旧流零回归.

    breathe=(head, tail) 秒: 2026-08-20 用户反馈"衔接太仓促" — 首帧定格 head 秒
    + 尾帧定格 tail 秒 (tpad), 音轨同步 adelay/apad。H3 台词 ~0.2s 起说,
    head=0.4 给起说前一个呼吸位, xfade 0.6 慢溶叠上后衔接不再抢话。
    """
    base = (f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={W}:{H},fps={fps}")
    if breathe:
        h_ms, t_ms = int(breathe[0] * 1000), int(breathe[1] * 1000)
        base = (f"{base},tpad=start_duration={breathe[0]}:start_mode=clone:"
                f"stop_duration={breathe[1]}:stop_mode=clone")
        # 音轨: 头静音 adelay + 尾静音 apad (pad_dur), 与视频 tpad 对齐
        a_pad = f"adelay={h_ms}:all=1,apad=pad_dur={breathe[1] + 0.5}"
        a_fc = f"[0:a]{a_pad}[a]" if _has_audio_stream(src) else None
    else:
        a_fc = None
    if sub_png:
        chain = f"{base}[bg];[bg][1:v]overlay=0:0:shortest=1[v]"
        if a_fc:
            fc = f"[0:v]{chain};{a_fc}"
            maps = ["-map", "[v]", "-map", "[a]"]
        else:
            fc = f"[0:v]{chain}"
            maps = ["-map", "[v]", "-map", "0:a?"]
        run([FFMPEG, "-y", "-i", src, "-loop", "1", "-i", sub_png,
             "-filter_complex", fc, *maps,
             "-c:a", "aac", "-b:a", "128k", "-shortest"]
            + enc_args(fps=fps, ts=V4_TS) + [dst])
    else:
        if a_fc:
            fc = f"[0:v]{base}[v];{a_fc}"
            run([FFMPEG, "-y", "-i", src, "-filter_complex", fc,
                 "-map", "[v]", "-map", "[a]",
                 "-c:a", "aac", "-b:a", "128k", "-shortest"]
                + enc_args(fps=fps, ts=V4_TS) + [dst])
        else:
            run([FFMPEG, "-y", "-i", src, "-vf", base,
                 "-map", "0:v", "-map", "0:a?", "-c:a", "aac", "-b:a", "128k",
                 "-shortest"]
                 + enc_args(fps=fps, ts=V4_TS) + [dst])


# ---------------- 拼接 ----------------
def concat_hard(segs, dst):
    tmp = os.path.dirname(segs[0])
    lst = os.path.join(tmp, "list.txt")
    with open(lst, "w", encoding="utf-8") as f:
        for s in segs:
            f.write(f"file '{s}'\n")
    run([FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", lst,
         "-c", "copy", dst])


def concat_xfade(segs, durs, dst, xfade=XFADE, fps=FPS, ts=TIMESCALE):
    """链式 xfade。offset_i = sum(durs[:i]) - i*xfade(实测时长, 精确)。"""
    n = len(segs)
    inputs = []
    for s in segs:
        inputs += ["-i", s]
    parts, prev = [], "0:v"
    for i in range(1, n):
        offset = round(sum(durs[:i]) - i * xfade, 3)
        label = "vout" if i == n - 1 else f"x{i}"
        parts.append(f"[{prev}][{i}:v]xfade=transition=fade:"
                     f"duration={xfade}:offset={offset}[{label}]")
        prev = label
    run([FFMPEG, "-y"] + inputs + ["-filter_complex", ";".join(parts),
         "-map", "[vout]"] + enc_args(fps=fps, ts=ts) + [dst])


def add_silent_audio(video, dst):
    dur = probe_dur(video)
    run([FFMPEG, "-y", "-i", video,
         "-f", "lavfi", "-t", f"{dur:.3f}",
         "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
         "-shortest", dst])


def _has_audio_stream(path):
    """探测视频文件是否有音轨 (H3 自带音 vs seedance/ltx 无声)."""
    try:
        out = subprocess.run(
            [FFPROBE, "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", path],
            capture_output=True, text=True)
        return bool(out.stdout.strip())
    except Exception:
        return False


def resolve_mux_mode(has_voice, has_orig):
    """组装音频路由纯函数 (provider 无关): has_voice × has_orig 四象限.

    has_voice=T & has_orig=T → mix     (H3 旁白镜: 环境音 + 叠 TTS 旁白)
    has_voice=T & has_orig=F → tts     (seedance/ltx 无声 provider: TTS 覆盖)
    has_voice=F & has_orig=T → keep    (H3 环境音 / 对话内嵌: 保留原音)
    has_voice=F & has_orig=F → silent  (全无音: 静音轨占位, 保 concat 音频链)

    ⚠ H3 对话镜的 TTS 不进 voice_dir (i2v 阶段内联驱动口型) → has_voice=F,
      走 keep 保留 H3 内嵌语音, 不重复叠音.
    """
    if has_voice and has_orig:
        return "mix"
    if has_voice:
        return "tts"
    if has_orig:
        return "keep"
    return "silent"


def mux_seg_audio(seg_video, voice_mp3, dst, keep_orig_audio=False):
    """视频段 + 本镜音频 → 音画段. 自动按 has_voice × has_orig 路由 (见 resolve_mux_mode).

    mode=mix (H3 旁白镜): 保留 H3 环境音 (0.4) + 叠 TTS 旁白 (1.0) → amix.
      ⚠ 2026-08-06 修音量: amix 默认 normalize=1 压低输出 (实测 max -10dB vs TTS
      -0.2dB) → normalize=0 + 显式音量 + alimiter 防削波.
    mode=tts (seedance): TTS 覆盖 (历史行为), 前 LEAD_IN_MS 静音 + apad 到视频长.
    mode=keep (H3 环境音/对话内嵌): 保留原音轨 (音画直接复制).
    mode=silent: 静音轨 (anullsrc), 保 concat 音频链不断.

    keep_orig_audio 历史显式混音开关, 现被自动路由取代 (H3 有原音即混), 参数保留兼容.
    """
    has_voice = bool(voice_mp3) and os.path.exists(voice_mp3)
    has_orig = _has_audio_stream(seg_video)
    mode = resolve_mux_mode(has_voice, has_orig)
    if mode == "mix":
        # H3 环境音 + TTS 旁白: amix 混音 (normalize=0 + alimiter 治低音量)
        # ⚠ 2026-08-07 修: TTS 单声道上混立体声会被 amix 内部 -3dB 平衡衰减
        # (旁白从 -9.3dB 掉到 -12dB). 显式 pan=stereo 复制单声道到双声道 (不衰减).
        fc = (f"[1:a]aresample=44100,adelay={LEAD_IN_MS},"
              f"pan=stereo|c0=c0|c1=c0,volume=1.0[voice];"
              f"[0:a]volume=0.4[bg];"
              f"[bg][voice]amix=inputs=2:duration=first:dropout_transition=0:normalize=0"
              f"[a0];[a0]alimiter=limit=0.95[a]")
        run([FFMPEG, "-y", "-i", seg_video, "-i", voice_mp3,
             "-filter_complex", fc, "-map", "0:v", "-map", "[a]",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
             "-shortest", dst])
    elif mode == "tts":
        fc = (f"[1:a]aresample=44100,adelay={LEAD_IN_MS}|{LEAD_IN_MS},"
              f"apad,atrim=0:99[a]")
        run([FFMPEG, "-y", "-i", seg_video, "-i", voice_mp3,
             "-filter_complex", fc, "-map", "0:v", "-map", "[a]",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
             "-shortest", dst])
    elif mode == "keep":
        # H3 环境音/对话内嵌: 保留原音轨 (不叠 TTS)。
        # 2026-08-20 v2 自然度: 对话镜内嵌语音 vs 环境镜环境音响度差大,
        # 轻压动态 (loudnorm 单遍 dynamic 模式) 拉平镜间音量跳变; -c copy 改重编码。
        run([FFMPEG, "-y", "-i", seg_video,
             "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", dst])
    else:  # silent
        dur = probe_dur(seg_video)
        run([FFMPEG, "-y", "-i", seg_video,
             "-f", "lavfi", "-t", f"{dur:.3f}",
             "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
             "-shortest", dst])


def concat_xfade_av(segs, durs, dst, xfade=XFADE, fps=FPS, ts=TIMESCALE,
                    transition="fade", transitions=None):
    """链式 xfade(视频) + acrossfade(音频) 对齐。offset_i = sum(durs[:i]) - i*xfade。

    transitions: 逐衔接转场名列表 (len = len(segs)-1, i 项 = seg_i → seg_{i+1});
                 None → 全部用 transition (现行为)。
    """
    n = len(segs)
    if transitions is not None:
        assert len(transitions) == len(segs) - 1, \
            f"transitions len {len(transitions)} != 镜数-1 {len(segs)-1}"
    inputs = []
    for s in segs:
        inputs += ["-i", s]
    pv, pa, prev_v, prev_a = [], [], "0:v", "0:a"
    for i in range(1, n):
        tr = transitions[i - 1] if transitions else transition
        offset = round(sum(durs[:i]) - i * xfade, 3)
        lv = "vout" if i == n - 1 else f"x{i}"
        la = "aout" if i == n - 1 else f"af{i}"
        pv.append(f"[{prev_v}][{i}:v]xfade=transition={tr}:"
                  f"duration={xfade}:offset={offset}[{lv}]")
        # 2026-08-20 v2: 曲线 tri→qsin — 三角淡入淡出机械感消除, 正弦缓入缓出自然
        pa.append(f"[{prev_a}][{i}:a]acrossfade=duration={xfade}"
                  f":curve1=qsin:curve2=qsin[{la}]")
        prev_v, prev_a = lv, la
    fc = ";".join(pv + pa)
    run([FFMPEG, "-y"] + inputs + ["-filter_complex", fc,
         "-map", "[vout]", "-map", "[aout]"] + enc_args(fps=fps, ts=ts)
         + ["-c:a", "aac", "-b:a", "128k", dst])


def mix_bgm(video, bgm, dst, bgm_volume=0.30, duck=True):
    """BGM 垫底混入成片 (2026-08-15, MiniMax Music 生成的健身 BGM).

    video: 已组装成片 (含对白/旁白/环境音)
    bgm:   音乐文件 (mp3/wav)
    dst:   输出 mp4
    bgm_volume: BGM 音量系数 (0-1, 默认 0.30 低调垫底)
    duck:  True=侧链闪避 (对白时 BGM 自动压低, 保语音清晰) / False=纯混音

    ffmpeg 侧链: [bgm][voice]sidechaincompress — 主输入=要压的 BGM, 次输入=侧链键 (语音).
    对白响时 BGM 被压下去, 语音间隙 BGM 回上来.
    """
    dur = probe_dur(video)
    if duck:
        # 2026-08-18 实测修正: threshold 0.03(≈-30dB) 连 H3 环境音(mean -11.8dB)
        # 都触发 → BGM 全程被 8:1 压死 (用户: 音乐几乎听不到)。
        # 提到 0.15(≈-16dB): 只对真语音(>0.15 linear)闪避, 环境音不误压。
        fc = (f"[0:a]asplit=2[va][vc];"
              f"[1:a]atrim=0:{dur},asetpts=PTS-STARTPTS,volume={bgm_volume}[bg];"
              f"[bg][vc]sidechaincompress=threshold=0.15:ratio=8:attack=30:release=400[duck];"
              f"[va][duck]amix=inputs=2:duration=first:normalize=0[outa]")
    else:
        fc = (f"[1:a]atrim=0:{dur},asetpts=PTS-STARTPTS,volume={bgm_volume}[bg];"
              f"[0:a][bg]amix=inputs=2:duration=first:normalize=0[outa]")
    run([FFMPEG, "-y", "-i", video, "-i", bgm,
         "-filter_complex", fc,
         "-map", "0:v", "-map", "[outa]",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", dst])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="v2", choices=["v1", "v2", "v3", "v4"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--xfade-dur", type=float, default=0.4,
                    help="xfade 转场时长(秒), 默认 0.4")
    ap.add_argument("--xfade-trans", default="fade",
                    help="xfade 转场类型: fade/fadeblack/dissolve/…, 默认 fade")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=f"ep01_{args.mode}_")
    print(f"[tmp] {tmp}")

    out = args.out or os.path.join(OUT, f"ep01_{args.mode}.mp4")

    if args.mode == "v4":
        # 写实 Seedance 8 镜: 全 i2v, 不分 kb/dyn, 复用字幕 + 旁白 mux + xfade_av。
        # 固定 tmp + seg/av 增量跳过 → 转场实验(--xfade-*)只重跑 concat, 秒出。
        tmp = os.path.join(OUT, "_v4_tmp")
        os.makedirs(tmp, exist_ok=True)
        print(f"[tmp] {tmp}")
        segs = []
        for shot, src, line in SHOTS_REAL:
            if not os.path.exists(src):
                print(f"[ERR] 源缺失: {src}")
                return 1
            dst = os.path.join(tmp, f"seg{shot:02d}.mp4")
            if not os.path.exists(dst):
                sub_png = os.path.join(tmp, f"sub{shot:02d}.png")
                render_subtitle(line, sub_png)
                print(f"=== shot {shot:2d} i2v ===", flush=True)
                gen_seg_i2v(src, dst, sub_png)
            else:
                print(f"=== shot {shot:2d} cached ===")
            segs.append(dst)
        durs = [probe_dur(s) for s in segs]
        print(f"[durs] {[round(d,2) for d in durs]}  sum={sum(durs):.2f}")
        segs_av = []
        for (shot, src, line), seg_v in zip(SHOTS_REAL, segs):
            voice = os.path.join(VOICE, f"voice_shot{shot:02d}.mp3")
            dst_av = os.path.join(tmp, f"av{shot:02d}.mp4")
            tag = os.path.basename(voice) if os.path.exists(voice) else "(silent)"
            if not os.path.exists(dst_av):
                print(f"  mux shot {shot:2d}  {tag}")
                mux_seg_audio(seg_v, voice, dst_av)
            else:
                print(f"  mux shot {shot:2d}  cached")
            segs_av.append(dst_av)
        durs_av = [probe_dur(s) for s in segs_av]
        concat_xfade_av(segs_av, durs_av, out, fps=V4_FPS, ts=V4_TS,
                        xfade=args.xfade_dur, transition=args.xfade_trans)
        final_dur = probe_dur(out)
        print(f"\n[done] -> {out}  ({final_dur:.1f}s)")
        return 0

    use_subs = args.mode in ("v2", "v3")
    segs = []
    for shot, dur, kind, src, line, kb in SHOTS:
        if not os.path.exists(src):
            print(f"[ERR] 源缺失: {src}")
            return 1
        sub_png = None
        if use_subs:
            sub_png = os.path.join(tmp, f"sub{shot:02d}.png")
            render_subtitle(line, sub_png)
        dst = os.path.join(tmp, f"seg{shot:02d}.mp4")
        print(f"=== shot {shot:2d} {kind} {dur}s {('kb:'+kb) if kind=='kb' else ''} ===")
        if kind == "kb":
            gen_seg_kb(src, dst, dur, kb, sub_png)
        else:
            gen_seg_dyn(src, dst, dur, sub_png)
        segs.append(dst)

    # 实测各段时长(xfade offset 要精确)
    durs = [probe_dur(s) for s in segs]
    print(f"[durs] {[round(d,2) for d in durs]}  sum={sum(durs):.2f}")

    if args.mode == "v3":
        # 每段 mux 本镜旁白 → xfade(视频)+acrossfade(音频) 对齐链
        segs_av = []
        for (shot, dur, kind, src, line, kb), seg_v in zip(SHOTS, segs):
            voice = os.path.join(VOICE, f"voice_shot{shot:02d}.mp3")
            dst_av = os.path.join(tmp, f"av{shot:02d}.mp4")
            tag = os.path.basename(voice) if os.path.exists(voice) else "(silent)"
            print(f"  mux shot {shot:2d}  {tag}")
            mux_seg_audio(seg_v, voice, dst_av)
            segs_av.append(dst_av)
        durs_av = [probe_dur(s) for s in segs_av]
        concat_xfade_av(segs_av, durs_av, out)
    elif args.mode == "v2":
        novid = os.path.join(tmp, "xfade.mp4")
        concat_xfade(segs, durs, novid)
        add_silent_audio(novid, out)
    else:
        concat_hard(segs, out)

    final_dur = probe_dur(out)
    print(f"\n[done] -> {out}  ({final_dur:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
