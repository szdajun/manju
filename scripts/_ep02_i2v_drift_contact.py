"""ep02 i2v 脸漂移对比 contact(临时诊断工具)。

全量重跑 i2v(Counterfeit 关键帧)后, 每镜抽中帧 vs 关键帧并排, 验"动起来脸扭没扭"。
漫画脸 insightface 检不出 → face_sim 无效, 只能靠肉眼: 关键帧脸干净的话, i2v 中帧脸是否还干净/同一个人。
每行 [shot{N} 关键帧 | shot{N} i2v 中帧]。
"""
import glob
import os
import subprocess
from PIL import Image, ImageDraw, ImageFont

BASE = "../manju"
KF = f"{BASE}/output/ep02"
I2V = f"{BASE}/output/ep02_i2v"
TMP = f"{BASE}/output/ep02_diagnose/_i2v_frames"
OUT = f"{BASE}/output/ep02_diagnose/i2v_drift_contact.png"
FFPROBE = "~/ffmpeg/ffprobe.exe"
FFMPEG = "~/ffmpeg/ffmpeg.exe"

SHOTS = list(range(1, 11))
TW = 280
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


def load_fit(path, tw=TW):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    return im.resize((tw, int(tw * h / w)), Image.LANCZOS)


def label(im, text, color=(255, 255, 0)):
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", 18)
    except Exception:
        f = ImageFont.load_default()
    d.rectangle([0, 0, im.size[0], 26], fill=(0, 0, 0))
    d.text((6, 3), text, fill=color, font=f)
    return im


def extract(mp4, t, dst):
    subprocess.run([FFMPEG, "-y", "-i", mp4, "-ss", f"{t:.2f}", "-frames:v", "1", dst],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    os.makedirs(TMP, exist_ok=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    for s in SHOTS:
        kf = latest(f"{KF}/ep02_shot{s:02d}_*.png")
        mp4 = latest(f"{I2V}/manju_ep02_shot{s:02d}_best.mp4")
        if not kf or not mp4:
            print(f"[skip] shot{s}: kf={'有' if kf else '缺'} mp4={'有' if mp4 else '缺'}")
            continue
        dur = probe_dur(mp4)
        mid = f"{TMP}/s{s:02d}_i2v_mid.png"
        try:
            extract(mp4, dur * 0.5, mid)
        except Exception as e:
            print(f"[skip] shot{s} 抽帧失败: {e}")
            continue
        rows.append([
            label(load_fit(kf), f"shot{s} 关键帧(静)"),
            label(load_fit(mid), f"shot{s} i2v 中帧(动)"),
        ])
        print(f"[shot {s:2d}] {os.path.basename(mp4)} dur={dur:.2f}s")
    if not rows:
        print("[ERR] 无行; i2v 产物可能还没就绪")
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
    print("[next] 肉眼: 右列(i2v 中帧)脸和左列(关键帧)是否同人同脸、没扭? 哪镜扭了单独 --seeds 3 重抽")


if __name__ == "__main__":
    main()
