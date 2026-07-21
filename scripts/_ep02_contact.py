"""ep02 写实关键帧 10 合 1 contact sheet (5x2), 标 face_sim, 便用户审写实质感+剧情连贯。
face_sim 数据来自 gen_ep02_real_batch 最近跑(crop 脸修复后), 硬编避重跑。"""
import glob
import os
from PIL import Image, ImageDraw, ImageFont

SRC = "../manju/output/ep02_real"
OUT = os.path.join(SRC, "_contact.png")
CELL_W = 300
LABEL_H = 38
GAP = 10
COLS = 5
# face_sim 来自下眼睑淡化重跑(2026-07-21); shot6 纯手部无脸
SIM = {1: 0.611, 2: 0.652, 3: 0.717, 4: 0.595, 5: 0.623,
       6: None, 7: 0.726, 8: 0.613, 9: 0.664, 10: 0.615}
try:
    font = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", 20)
except Exception:
    font = ImageFont.load_default()


def latest(pat):
    fs = glob.glob(pat)
    return max(fs, key=os.path.getmtime) if fs else None


cells = []
for s in range(1, 11):
    f = latest(os.path.join(SRC, f"ep02_shot{s:02d}_*.png"))
    if not f:
        continue
    im = Image.open(f).convert("RGB")
    ratio = CELL_W / im.width
    im = im.resize((CELL_W, int(im.height * ratio)))
    cell = Image.new("RGB", (CELL_W, im.height + LABEL_H), "white")
    cell.paste(im, (0, 0))
    d = ImageDraw.Draw(cell)
    sim = SIM[s]
    simstr = f"{sim:.2f}" if sim else "noface"
    d.text((8, im.height + 9), f"#{s:02d} sim{simstr}", fill="black", font=font)
    cells.append(cell)

ROWS = (len(cells) + COLS - 1) // COLS
cell_h = max(c.height for c in cells)
W = COLS * CELL_W + (COLS + 1) * GAP
H = ROWS * cell_h + (ROWS + 1) * GAP
sheet = Image.new("RGB", (W, H), (30, 30, 30))
for i, cell in enumerate(cells):
    r, c = divmod(i, COLS)
    sheet.paste(cell, (GAP + c * (CELL_W + GAP), GAP + r * (cell_h + GAP)))
sheet.save(OUT)
print(f"[OK] {OUT} ({W}x{H}, {len(cells)} frames)")
