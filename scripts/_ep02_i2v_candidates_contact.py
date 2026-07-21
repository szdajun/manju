"""ep02 i2v 候选挑选 contact(临时诊断工具)。

5 镜(1,2,4,8,10)各 --seeds 3 重抽后, 每镜出 [关键帧 | 候选A | B | C](脸放大),
用户挑每镜最好的候选(seed)。漫画 face_sim 无效 → 必须肉眼挑。
candidates/shot{NN}_*.mp4 取最新 3 个(mtime)。
"""
import glob
import os
import subprocess
from PIL import Image, ImageDraw, ImageFont

BASE = "../manju"
KF = f"{BASE}/output/ep02"
I2V = f"{BASE}/output/ep02_i2v"
CAND = f"{I2V}/candidates"
TMP = f"{BASE}/output/ep02_diagnose/_cand_frames"
OUT = f"{BASE}/output/ep02_diagnose/i2v_candidates_contact_28.png"
FFPROBE = "~/ffmpeg/ffprobe.exe"
FFMPEG = "~/ffmpeg/ffmpeg.exe"

SHOTS = [2, 8]
N = 3
TW = 300
TOP_FRAC = 0.52
PAD = 8


def latest(pat):
    fs = glob.glob(pat)
    return max(fs, key=os.path.getmtime) if fs else None


def newest_n(pat, n=N):
    fs = sorted(glob.glob(pat), key=os.path.getmtime, reverse=True)
    return list(reversed(fs[:n]))  # 回到时间正序 A/B/C/...


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
    letters = ["A", "B", "C", "D", "E", "F"]
    for s in SHOTS:
        kf = latest(f"{KF}/ep02_shot{s:02d}_*.png")
        cands = newest_n(f"{CAND}/shot{s:02d}_*.mp4", N)
        if not kf or len(cands) < 1:
            print(f"[skip] shot{s}: kf={'有' if kf else '缺'} cands={len(cands)}")
            continue
        cells = [label(crop_zoom(kf), f"shot{s} 关键帧")]
        for i, mp4 in enumerate(cands):
            lt = letters[i] if i < len(letters) else str(i)
            dst = f"{TMP}/s{s:02d}_{lt}.png"
            try:
                dur = probe_dur(mp4)
                extract(mp4, dur * 0.5, dst)
                cells.append(label(crop_zoom(dst), f"shot{s} 候选{lt}"))
            except Exception as e:
                print(f"[skip] shot{s} cand{i}: {e}")
        rows.append(cells)
        print(f"[shot {s:2d}] {len(cands)} cands: {[os.path.basename(c) for c in cands]}")
    if not rows:
        print("[ERR] 无行; 候选可能还没跑完")
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
    print(f"[挑] shot2 选 A-F(脸稳的); shot8 选 A-F 或'都崩'→我简化构图重抽。报格式: 2X 8Y")


if __name__ == "__main__":
    main()
