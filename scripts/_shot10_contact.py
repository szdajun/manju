"""shot10 多 seed 候选 contact sheet (2026-07-22)。

眼睛/眼睑局部瑕疵只有人眼能判(arcface 测不到)。批量抽 3 seed 后, 把每候选的 f2.5(中)
+ f4.9(末) 拼成一张大图, 用户一行扫完挑"两人眼睛/眼睑都干净"的候选。PIL 拼图避开
ffmpeg drawtext 坑。跑在 manju venv (有 PIL)。

用法: .venv/Scripts/python.exe scripts/_shot10_contact.py
产物: output/ep02_i2v_real/_candidates/contact_sheet.png
"""
import glob
import os
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

FFMPEG = "~/ffmpeg/ffmpeg.exe"
CAND = "../manju/output/ep02_i2v_real/_candidates"
TS = [2.5, 4.9]   # 中 + 末 (眼睛/眼睑瑕疵集中段)
FONT = "C:/Windows/Fonts/msyhbd.ttc"


def extract(mp4, t, dst):
    subprocess.run([FFMPEG, "-y", "-i", mp4, "-ss", f"{t:.2f}", "-frames:v", "1", dst],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    rolls = sorted(glob.glob(os.path.join(CAND, "roll*.mp4")),
                   key=lambda p: int(os.path.basename(p).replace("roll", "").replace(".mp4", "")))
    if not rolls:
        print("[ERR] 无 roll*.mp4 候选")
        return 1
    tmp = os.path.join(CAND, "_frames")
    os.makedirs(tmp, exist_ok=True)
    font = ImageFont.truetype(FONT, 28)
    labh = 40
    w, h = 480, 832
    cols = len(TS)
    pad = 8
    grid_w = cols * (w + pad) + pad
    grid_h = len(rolls) * (h + labh + pad) + pad
    grid = Image.new("RGB", (grid_w, grid_h), (30, 30, 30))
    dr = ImageDraw.Draw(grid)
    tags = {"roll0": "roll0 参考(小雅fixed/婷婷eyelid坏)", }
    for ri, mp4 in enumerate(rolls):
        base = os.path.basename(mp4).replace(".mp4", "")
        y0 = ri * (h + labh + pad) + pad
        dr.text((pad, y0 + 6), f"{base}  左=小雅(锁脸) 右=婷婷(未锁)", fill=(255, 255, 0), font=font)
        for ci, t in enumerate(TS):
            dst = os.path.join(tmp, f"{base}_f{t}.png")
            extract(mp4, t, dst)
            im = Image.open(dst).resize((w, h))
            x0 = ci * (w + pad) + pad
            grid.paste(im, (x0, y0 + labh))
            dr.text((x0 + 6, y0 + labh + 6), f"t={t}s", fill=(0, 255, 0), font=font)
    out = os.path.join(CAND, "contact_sheet.png")
    grid.save(out)
    print(f"[saved] {out}  ({len(rolls)} 候选 × {cols} 帧)")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
