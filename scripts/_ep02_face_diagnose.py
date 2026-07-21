"""ep02 脸变形诊断 contact sheet(临时诊断工具)。

用户报"脸变形可怕" → face_sim 数值过阈但视觉可怕(buffalo_l 真人脸检测器对漫画脸松,数值盲区)。
做三层对比定位变形在哪层, 对症下药:
  - 立绘诡异    → 换立绘(重生更自然脸 / 写实立绘)
  - 关键帧漂    → IPAdapter 0.85→0.95 锁紧
  - i2v 中帧扭曲 → LTX 运镜形变 → 降运镜 / face_swap 救脸

布局: 顶行 立绘(脸源); 每镜一行 [关键帧 | i2v@0.35d | i2v@0.65d](i2v 抓非中帧峰值扭曲)。
"""
import os
import glob
import subprocess
from PIL import Image, ImageDraw, ImageFont

BASE = "../manju"
REF = f"{BASE}/assets/characters/小雅立绘.png"
KF = f"{BASE}/output/ep02"
I2V = f"{BASE}/output/ep02_i2v"
TMP = f"{BASE}/output/ep02_diagnose/_frames"
OUT = f"{BASE}/output/ep02_diagnose/contact.png"
FFPROBE = "~/ffmpeg/ffprobe.exe"
FFMPEG = "~/ffmpeg/ffmpeg.exe"

# 脸主导镜(shot10 是两人告别, face_sim 0.233 最差, 必查; 6/8 无主角脸跳过)
SHOTS = [1, 3, 5, 7, 9, 10]
TW = 300
PAD = 6


def probe_dur(mp4):
    try:
        return float(subprocess.check_output(
            [FFPROBE, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", mp4],
            text=True).strip())
    except Exception:
        return 2.56


def latest(pat):
    fs = glob.glob(pat)
    return max(fs, key=os.path.getmtime) if fs else None


def load_fit(path, tw=TW):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    return im.resize((tw, int(tw * h / w)), Image.LANCZOS)


def label(im, text):
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", 20)
    except Exception:
        f = ImageFont.load_default()
    d.rectangle([0, 0, im.size[0], 28], fill=(0, 0, 0))
    d.text((6, 4), text, fill=(255, 255, 0), font=f)
    return im


def extract(mp4, t, dst):
    subprocess.run([FFMPEG, "-y", "-i", mp4, "-ss", f"{t:.2f}", "-frames:v", "1", dst],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    os.makedirs(TMP, exist_ok=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []

    # 顶行: 立绘
    rows.append([label(load_fit(REF), "立绘(脸源) · 小雅立绘.png")])

    for s in SHOTS:
        kf = latest(f"{KF}/ep02_shot{s:02d}_*.png")
        mp4 = latest(f"{I2V}/manju_ep02_shot{s:02d}_*.mp4")
        cells = []
        if kf:
            cells.append(label(load_fit(kf), f"shot{s} 关键帧(IPAdapter)"))
        if mp4:
            dur = probe_dur(mp4)
            for frac, tag in [(0.35, "i2v 35%"), (0.65, "i2v 65%")]:
                dst = f"{TMP}/s{s:02d}_{tag.replace(' ', '').replace('%','p')}.png"
                try:
                    extract(mp4, dur * frac, dst)
                    cells.append(label(load_fit(dst), f"shot{s} {tag}"))
                except Exception as e:
                    print(f"[skip] shot{s} {tag}: {e}")
        if cells:
            rows.append(cells)
            print(f"[shot {s:2d}] {len(cells)} 格")

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
    print("[next] 看图: 立绘脸本身诡异? → 换立绘 | 关键帧脸漂? → IPAdapter 锁紧 | i2v 帧扭曲? → 降运镜/face_swap")


if __name__ == "__main__":
    main()
