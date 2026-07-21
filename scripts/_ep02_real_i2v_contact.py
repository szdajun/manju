"""ep02 写实 Wan i2v 脸漂移 contact(2026-07-22)。

每镜一行: [关键帧(静) | i2v 0.5s | i2v 中帧 | i2v 4.5s], 验"动起来小雅脸扭没扭/漂没漂"。
shot10 原漂移(mid=0.108) → 收紧 motion 重抽过线(0.544), contact 末行重点验。
关键帧 ep02_real, i2v 产物 ep02_i2v_real/manju_wan_shot*.mp4。
"""
import glob
import os
import subprocess
from PIL import Image, ImageDraw, ImageFont

BASE = "../manju"
KF = f"{BASE}/output/ep02_real"
I2V = f"{BASE}/output/ep02_i2v_real"
TMP = f"{BASE}/output/ep02_i2v_real/_contact_frames"
OUT = f"{BASE}/output/ep02_i2v_real/_contact.png"
FFPROBE = "~/ffmpeg/ffprobe.exe"
FFMPEG = "~/ffmpeg/ffmpeg.exe"

SHOTS = list(range(1, 11))
TW = 240
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
        return 5.0


def load_fit(path, tw=TW):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    return im.resize((tw, int(tw * h / w)), Image.LANCZOS)


def label(im, text, color=(255, 255, 0)):
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", 16)
    except Exception:
        f = ImageFont.load_default()
    d.rectangle([0, 0, im.size[0], 24], fill=(0, 0, 0))
    d.text((6, 3), text, fill=color, font=f)
    return im


def extract(mp4, t, dst):
    subprocess.run([FFMPEG, "-y", "-i", mp4, "-ss", f"{t:.2f}", "-frames:v", "1", dst],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    os.makedirs(TMP, exist_ok=True)
    rows = []
    for s in SHOTS:
        kf = latest(f"{KF}/ep02_shot{s:02d}*.png")
        mp4 = latest(f"{I2V}/manju_wan_shot{s:02d}_*.mp4")
        if not kf or not mp4:
            print(f"[skip] shot{s}: kf={'有' if kf else '缺'} mp4={'有' if mp4 else '缺'}")
            continue
        dur = probe_dur(mp4)
        cells = [label(load_fit(kf), f"shot{s} 关键帧")]
        for i, frac in enumerate([0.1, 0.5, 0.9]):
            t = dur * frac
            dst = f"{TMP}/s{s:02d}_{frac}.png"
            try:
                extract(mp4, t, dst)
                cells.append(label(load_fit(dst), f"i2v {t:.1f}s"))
            except Exception as e:
                print(f"[skip] shot{s} 抽帧 {t:.1f}s 失败: {e}")
        rows.append(cells)
        print(f"[shot {s:2d}] {os.path.basename(mp4)} dur={dur:.2f}s")
    if not rows:
        print("[ERR] 无行; 产物可能还没就绪")
        return 1
    ncols = max(len(r) for r in rows)
    allw = ncols * TW + (ncols + 1) * PAD
    h = sum(max(c.size[1] for c in row) + PAD for row in rows) + PAD
    canvas = Image.new("RGB", (allw, h), (40, 40, 40))
    y = PAD
    for row in rows:
        x = PAD
        for c in row:
            canvas.paste(c, (x, y))
            x += c.size[0] + PAD
        y += max(c.size[1] for c in row) + PAD
    canvas.save(OUT, quality=88)
    print(f"\n[saved] {OUT}  {canvas.size}")


if __name__ == "__main__":
    main()
