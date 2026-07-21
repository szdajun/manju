"""小雅 6 立绘候选 contact sheet (3x2), 标 det_score/性别年龄, 便用户一眼挑脸。
排脸数据(_face_rank.py 已跑)硬编入避重跑。vision 不可用作图代客观。"""
import os
from PIL import Image, ImageDraw, ImageFont

SRC = "../manju/output/xiaoya_real"
OUT = os.path.join(SRC, "_contact.png")
# (标签, 文件名, det_score, 性别, 年龄) — 来自 _face_rank.py 输出, 按 det 降序
RANK = [
    ("#04", "manju_xiaoya_real_00004_.png", 0.8992, "F", 24),
    ("#01", "manju_xiaoya_real_00001_.png", 0.8926, "F", 22),
    ("#03", "manju_xiaoya_real_00003_.png", 0.8530, "F", 24),
    ("#06", "manju_xiaoya_real_00006_.png", 0.8365, "F", 20),
    ("#02", "manju_xiaoya_real_00002_.png", 0.8092, "F", 23),
    ("#05", "manju_xiaoya_real_00005_.png", 0.7647, "F", 23),
]
COLS, ROWS = 3, 2
CELL_W = 340
LABEL_H = 44
GAP = 12
try:
    font = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", 22)
except Exception:
    font = ImageFont.load_default()

cells = []
for tag, fn, det, g, age in RANK:
    im = Image.open(os.path.join(SRC, fn)).convert("RGB")
    ratio = CELL_W / im.width
    im = im.resize((CELL_W, int(im.height * ratio)))
    cell = Image.new("RGB", (CELL_W, im.height + LABEL_H), "white")
    cell.paste(im, (0, 0))
    d = ImageDraw.Draw(cell)
    d.text((10, im.height + 10), f"{tag}  det {det:.3f}  {g}{age}", fill="black", font=font)
    cells.append(cell)

cell_h = max(c.height for c in cells)
W = COLS * CELL_W + (COLS + 1) * GAP
H = ROWS * cell_h + (ROWS + 1) * GAP
sheet = Image.new("RGB", (W, H), (30, 30, 30))
for i, cell in enumerate(cells):
    r, c = divmod(i, COLS)
    sheet.paste(cell, (GAP + c * (CELL_W + GAP), GAP + r * (cell_h + GAP)))
sheet.save(OUT)
print(f"[OK] {OUT}  ({W}x{H})")
