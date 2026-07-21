"""ep02 i2v 脸放大多帧对比 contact(临时诊断工具)。

用户报"有些中帧脸部看不清楚" → 单中帧不代表性 + 缩略图脸太小。
本脚本: 每个有脸镜头取 [关键帧 | i2v@20% | i2v@50% | i2v@80%], 裁上半身(top_frac)放大脸部,
让脸占满格子。区分"某一瞬脸转开"(正常) vs "全程糊/扭"(真问题)。
"""
import glob
import os
import subprocess
from PIL import Image, ImageDraw, ImageFont

BASE = "../manju"
KF = f"{BASE}/output/ep02"
I2V = f"{BASE}/output/ep02_i2v"
TMP = f"{BASE}/output/ep02_diagnose/_facezoom_frames"
OUT = f"{BASE}/output/ep02_diagnose/i2v_facezoom_contact.png"
FFPROBE = "~/ffmpeg/ffprobe.exe"
FFMPEG = "~/ffmpeg/ffmpeg.exe"

FACE_SHOTS = [1, 2, 3, 4, 5, 7, 8, 9, 10]  # 含 8(人群, 用户报有问题); 跳 6 手部
FRACS = [0.2, 0.5, 0.8]
TW = 360
TOP_FRAC = 0.52  # 裁上半身(头+肩+胸), 脸占满格
PAD = 8


def latest(pat):
    fs = glob.glob(pat)
    return max(fs, key=os.path.getmtime) if fs else None


def probe_dur(mp4):
    try:
        return float(subprocess.check_output(
            [FFPROBE, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", mp4],
            text=True).strip())
    except Exception:
        return 2.56


def crop_zoom(path, tw=TW, top_frac=TOP_FRAC):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    ch = int(h * top_frac)
    im = im.crop((0, 0, w, ch))
    cw, ch2 = im.size
    return im.resize((tw, int(tw * ch2 / cw)), Image.LANCZOS)


def label(im, text, color=(255, 255, 0)):
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", 20)
    except Exception:
        f = ImageFont.load_default()
    d.rectangle([0, 0, im.size[0], 28], fill=(0, 0, 0))
    d.text((6, 4), text, fill=color, font=f)
    return im


def extract(mp4, t, dst):
    subprocess.run([FFMPEG, "-y", "-i", mp4, "-ss", f"{t:.2f}", "-frames:v", "1", dst],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    os.makedirs(TMP, exist_ok=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    for s in FACE_SHOTS:
        kf = latest(f"{KF}/ep02_shot{s:02d}_*.png")
        mp4 = latest(f"{I2V}/manju_ep02_shot{s:02d}_best.mp4")
        if not kf or not mp4:
            print(f"[skip] shot{s}: kf={'有' if kf else '缺'} mp4={'有' if mp4 else '缺'}")
            continue
        dur = probe_dur(mp4)
        cells = [label(crop_zoom(kf), f"shot{s} 关键帧")]
        for fr in FRACS:
            dst = f"{TMP}/s{s:02d}_{int(fr*100)}.png"
            try:
                extract(mp4, dur * fr, dst)
                cells.append(label(crop_zoom(dst), f"shot{s} i2v@{int(fr*100)}%"))
            except Exception as e:
                print(f"[skip] shot{s}@{fr}: {e}")
        rows.append(cells)
        print(f"[shot {s:2d}] 4 cells")
    if not rows:
        print("[ERR] 无行")
        return 1
    allw = max(sum(c.size[0] for c in row) + PAD * (len(row) + 1) for row in rows)
    h = sum(max(c.size[1] for c in row) + PAD for row in rows) + PAD
    canvas = Image.new("RGB", (allw, h), (40, 40, 40))
    y = PAD
    for row in rows:
        x = PAD
        for c in row:
            canvas.paste(c, (x, y))
            x += c.size[0] + PAD
        y += max(c.size[1] for c in row) + PAD
    canvas.save(OUT, quality=90)
    print(f"\n[saved] {OUT}  {canvas.size}")
    print("[判] 每行最左=关键帧脸; 右三=i2v 动起来 20/50/80%。全程糊/扭=真问题; 只一瞬不清=正常转脸")


if __name__ == "__main__":
    main()
