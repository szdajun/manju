"""临时: 把 ep01 10 写实关键帧(每镜取最新一次)拼成 contact sheet 供用户一眼挑 8 镜。
shot 号标签; face_sim 数值我另在对话里报。用完可删。"""
import glob
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

KF = "../manju/output/ep01_real"
OUT = "../manju/output/ep01_real/_keyframes_contact.png"

pairs = []
for s in range(1, 11):
    cands = sorted(glob.glob(os.path.join(KF, f"ep01_shot{s:02d}_*.png")))
    if cands:
        pairs.append((s, cands[-1]))  # 最新一次重抽
assert pairs, f"no keyframes in {KF}"

imgs = [Image.open(p).convert("RGB") for _, p in pairs]
tw = 460
thumbs = [im.resize((tw, int(im.size[1] * tw / im.size[0])), Image.LANCZOS) for im in imgs]
cols, rows = 2, (len(thumbs) + 1) // 2
th_max = max(t.size[1] for t in thumbs)
pad, labelh = 12, 34
W = cols * tw + (cols + 1) * pad
H = rows * (th_max + labelh) + (rows + 1) * pad
sheet = Image.new("RGB", (W, H), (16, 16, 20))
d = ImageDraw.Draw(sheet)
try:
    font = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", 22)
except Exception:
    font = ImageFont.load_default()
for i, (s, t) in enumerate(zip([p[0] for p in pairs], thumbs)):
    cx = pad + (i % cols) * (tw + pad)
    cy = pad + (i // cols) * (th_max + labelh + pad)
    sheet.paste(t, (cx, cy + labelh))
    d.rectangle([cx, cy, cx + tw, cy + labelh], fill=(0, 0, 0))
    d.text((cx + 8, cy + 6), f"shot {s}", fill=(255, 230, 60), font=font)
sheet.save(OUT)
print(f"[contact] {OUT} {sheet.size} ({len(pairs)} keyframes)")
