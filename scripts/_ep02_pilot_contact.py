"""ep02 pilot before/after 对比 contact(临时诊断工具)。

pilot 重跑 shot4/10(cfg2.5/steps40/多seed/motion去转头)后, 出 before/after:
每镜一行 [关键帧 | 旧 i2v 65% | 新 best 35% | 新 best 65%], 肉眼判脸形变治没治。
旧版在 _old/shot{NN}_old.mp4, 新 best 在 manju_ep02_shot{NN}_best.mp4。
"""
import glob
import os
import subprocess
from PIL import Image, ImageDraw, ImageFont

BASE = "../manju"
KF = f"{BASE}/output/ep02"
I2V = f"{BASE}/output/ep02_i2v"
TMP = f"{BASE}/output/ep02_diagnose/_pilot_frames"
OUT = f"{BASE}/output/ep02_diagnose/pilot_contact.png"
FFPROBE = "~/ffmpeg/ffprobe.exe"
FFMPEG = "~/ffmpeg/ffmpeg.exe"

SHOTS = [4, 10]
TW = 320
PAD = 8


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
    for s in SHOTS:
        cells = []
        kf = latest(f"{KF}/ep02_shot{s:02d}_*.png")
        if kf:
            cells.append(label(load_fit(kf), f"shot{s} keyframe"))
        old = os.path.join(I2V, "_old", f"shot{s:02d}_old.mp4")
        if os.path.exists(old):
            dur = probe_dur(old)
            dst = f"{TMP}/s{s:02d}_old65.png"
            try:
                extract(old, dur * 0.65, dst)
                cells.append(label(load_fit(dst), f"shot{s} OLD i2v 65% (before)"))
            except Exception as e:
                print(f"[skip] shot{s} old65: {e}")
        new = latest(f"{I2V}/manju_ep02_shot{s:02d}_best.mp4")
        if new:
            dur = probe_dur(new)
            for frac, tag in [(0.35, "35%"), (0.65, "65%")]:
                dst = f"{TMP}/s{s:02d}_new{tag.replace('%','p')}.png"
                try:
                    extract(new, dur * frac, dst)
                    cells.append(label(load_fit(dst), f"shot{s} NEW best {tag} (after)"))
                except Exception as e:
                    print(f"[skip] shot{s} new{tag}: {e}")
        if cells:
            rows.append(cells)
            print(f"[shot {s:2d}] {len(cells)} cells")

    if not rows:
        print("[ERR] no cells; pilot 产物可能还没就绪")
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


if __name__ == "__main__":
    main()
