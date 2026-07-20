"""ep01 成片组装 v1/v2。

v1(rough cut): KB 慢推静态镜 + 3× 慢放动态镜, 硬切, 无字幕无音。
v2: + 字幕(PIL 黄字黑描边, 避 ffmpeg drawtext 8.1 bug)
    + 淡入淡出转场(xfade 0.4s, 链式 offset 精确按各段实测时长)
    + KB 运镜变化(推近 in / 拉远 out / 横摇 pan 交替, 不再全中心推)
    + 静音占位音轨(AAC, TTS 替换时直接换音轨)
v3: = v2 画面 + 真配音(逐镜 TTS 旁白 mux, xfade 视频链 + acrossfade 音频链对齐)。

用法:
    python scripts/make_ep01.py --mode v1        # rough cut 纯画面硬切
    python scripts/make_ep01.py --mode v2        # 字幕+转场+KB变化+静音音轨
    python scripts/make_ep01.py --mode v3        # v2 + 真配音(edge-tts 草版/可切 provider)
"""
import argparse
import os
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont

FFMPEG = "~/ffmpeg/ffmpeg.exe"
FFPROBE = "~/ffmpeg/ffprobe.exe"
EP01 = "../manju/output/ep01"
I2V = "../manju/output/ep01_i2v"
VOICE = "../manju/output/ep01_voice"
OUT = "../manju/output/ep01_assemble"
FONT = "C:/Windows/Fonts/msyhbd.ttc"
LEAD_IN_MS = 300   # 旁白 lead-in 静音(ms), 避免开口贴镜首

FPS = 16
W, H = 720, 1280
XFADE = 0.4
TIMESCALE = str(FPS * 1024)
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


def render_subtitle(text, out):
    """底部黄字黑描边 + 半透明黑底圆角条。720×1280 透明 PNG。"""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, 42)
    lines = wrap_text(text)
    line_h = 56
    total_h = len(lines) * line_h
    y0 = H - total_h - 120
    widths = [d.textlength(l, font=font) for l in lines]
    maxw = max(widths)
    pad_x, pad_y = 30, 18
    bx0 = (W - maxw) / 2 - pad_x
    by0 = y0 - pad_y
    bx1 = (W + maxw) / 2 + pad_x
    by1 = y0 + total_h + pad_y - 4
    d.rounded_rectangle([bx0, by0, bx1, by1], radius=14, fill=(0, 0, 0, 165))
    for i, (l, w) in enumerate(zip(lines, widths)):
        x = (W - w) / 2
        y = y0 + i * line_h
        d.text((x, y), l, font=font, fill=(255, 218, 70, 255),
               stroke_width=3, stroke_fill=(0, 0, 0, 255))
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


def enc_args(nframes):
    return ["-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-r", str(FPS), "-video_track_timescale", TIMESCALE]


def gen_seg_kb(src, dst, dur, kb_type, sub_png):
    vf = kenburns_vf(dur, kb_type)
    if sub_png:
        fc = f"[0:v]{vf}[bg];[bg][1:v]overlay=0:0:shortest=1[v]"
        run([FFMPEG, "-y", "-i", src, "-loop", "1", "-i", sub_png,
             "-filter_complex", fc, "-map", "[v]"] + enc_args(None) + [dst])
    else:
        run([FFMPEG, "-y", "-i", src, "-vf", vf] + enc_args(None) + [dst])


def gen_seg_dyn(src, dst, dur, sub_png, factor=3.0):
    base = (f"scale={W}:{H}:force_original_aspect_ratio=increase,"
            f"crop={W}:{H},setpts=PTS*{factor},fps={FPS}")
    if sub_png:
        fc = f"[0:v]{base}[bg];[bg][1:v]overlay=0:0:shortest=1[v]"
        run([FFMPEG, "-y", "-i", src, "-loop", "1", "-i", sub_png,
             "-filter_complex", fc, "-map", "[v]"] + enc_args(None) + [dst])
    else:
        run([FFMPEG, "-y", "-i", src, "-vf", base] + enc_args(None) + [dst])


# ---------------- 拼接 ----------------
def concat_hard(segs, dst):
    tmp = os.path.dirname(segs[0])
    lst = os.path.join(tmp, "list.txt")
    with open(lst, "w", encoding="utf-8") as f:
        for s in segs:
            f.write(f"file '{s}'\n")
    run([FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", lst,
         "-c", "copy", dst])


def concat_xfade(segs, durs, dst, xfade=XFADE):
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
         "-map", "[vout]"] + enc_args(None) + [dst])


def add_silent_audio(video, dst):
    dur = probe_dur(video)
    run([FFMPEG, "-y", "-i", video,
         "-f", "lavfi", "-t", f"{dur:.3f}",
         "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
         "-shortest", dst])


def mux_seg_audio(seg_video, voice_mp3, dst):
    """视频段 + 本镜旁白 → 音画段。旁白前 LEAD_IN_MS 静音, 后 apad 到视频时长(-shortest 截)。

    无旁白文件 → 退静音轨(anullsrc), v3 仍可建(某镜没配音不卡整片)。
    """
    if voice_mp3 and os.path.exists(voice_mp3):
        fc = (f"[1:a]aresample=44100,adelay={LEAD_IN_MS}|{LEAD_IN_MS},"
              f"apad,atrim=0:99[a]")
        run([FFMPEG, "-y", "-i", seg_video, "-i", voice_mp3,
             "-filter_complex", fc, "-map", "0:v", "-map", "[a]",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
             "-shortest", dst])
    else:
        dur = probe_dur(seg_video)
        run([FFMPEG, "-y", "-i", seg_video,
             "-f", "lavfi", "-t", f"{dur:.3f}",
             "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
             "-shortest", dst])


def concat_xfade_av(segs, durs, dst, xfade=XFADE):
    """链式 xfade(视频) + acrossfade(音频) 对齐。offset_i = sum(durs[:i]) - i*xfade。

    每段音画同长 → 视频重叠 0.4s 处音频也 acrossfade 0.4s, 两链同步。
    """
    n = len(segs)
    inputs = []
    for s in segs:
        inputs += ["-i", s]
    pv, pa, prev_v, prev_a = [], [], "0:v", "0:a"
    for i in range(1, n):
        offset = round(sum(durs[:i]) - i * xfade, 3)
        lv = "vout" if i == n - 1 else f"x{i}"
        la = "aout" if i == n - 1 else f"af{i}"
        pv.append(f"[{prev_v}][{i}:v]xfade=transition=fade:"
                  f"duration={xfade}:offset={offset}[{lv}]")
        pa.append(f"[{prev_a}][{i}:a]acrossfade=duration={xfade}"
                  f":curve1=tri:curve2=tri[{la}]")
        prev_v, prev_a = lv, la
    fc = ";".join(pv + pa)
    run([FFMPEG, "-y"] + inputs + ["-filter_complex", fc,
         "-map", "[vout]", "-map", "[aout]"] + enc_args(None)
         + ["-c:a", "aac", "-b:a", "128k", dst])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="v2", choices=["v1", "v2", "v3"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=f"ep01_{args.mode}_")
    print(f"[tmp] {tmp}")

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

    out = args.out or os.path.join(OUT, f"ep01_{args.mode}.mp4")
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
