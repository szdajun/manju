"""临时: 把 6 张立绘候选拼成 2x3 contact sheet 便于一眼比对挑脸。用完即删。"""
import glob
import os
from PIL import Image, ImageDraw, ImageFont

SRC = "../manju/output/lihui_real"
OUT = "../manju/output/lihui_real/_contact.png"

files = sorted(glob.glob(os.path.join(SRC, "manju_lihui_real_*.png")))
imgs = [Image.open(f).convert("RGB") for f in files]
tw = 420
thumbs = [im.resize((tw, int(im.size[1] * tw / im.size[0])), Image.LANCZOS) for im in imgs]
cols, rows = 3, 2
th_max = max(t.size[1] for t in thumbs)
pad, labelh = 12, 30
W = cols * tw + (cols + 1) * pad
H = rows * (th_max + labelh) + (rows + 1) * pad
sheet = Image.new("RGB", (W, H), (16, 16, 20))
d = ImageDraw.Draw(sheet)
try:
    font = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", 20)
except Exception:
    font = ImageFont.load_default()
for i, t in enumerate(thumbs):
    cx = pad + (i % cols) * (tw + pad)
    cy = pad + (i // cols) * (th_max + labelh + pad)
    sheet.paste(t, (cx, cy + labelh))
    d.rectangle([cx, cy, cx + tw, cy + labelh], fill=(0, 0, 0))
    d.text((cx + 8, cy + 5), f"#{i+1}", fill=(255, 230, 60), font=font)
sheet.save(OUT)
print(f"[contact] {OUT} {sheet.size} ({len(files)} imgs)")
