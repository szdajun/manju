"""小雅立绘画风对比 contact(临时诊断工具)。

用户报"两美女脸都怪异不好看" → 根因 DreamShaper 半写实强画漫画=恐怖谷。
换 Counterfeit-V3.0(动漫专用)+骄傲半闭眼(prompt)重生立绘候选。
本脚本出 [DreamShaper立绘(现·诡异) | Counterfeit候选×3] 并排, 给用户挑画风。

before = assets/characters/小雅立绘.png(DreamShaper, 当前 REF)
after  = output/xiaoya_manga/ 里最新 3 张(Counterfeit, gen_xiaoya_manga.py --ckpt Counterfeit 产出)
"""
import glob
import os
from PIL import Image, ImageDraw, ImageFont

BASE = "../manju"
BEFORE = f"{BASE}/assets/characters/小雅立绘.png"
AFTER_DIR = f"{BASE}/output/xiaoya_manga"
OUT = f"{AFTER_DIR}/style_compare.png"
TW = 420
PAD = 10


def load_fit(path, tw=TW):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    return im.resize((tw, int(tw * h / w)), Image.LANCZOS)


def label(im, text, color=(255, 255, 0)):
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", 22)
    except Exception:
        f = ImageFont.load_default()
    d.rectangle([0, 0, im.size[0], 30], fill=(0, 0, 0))
    d.text((8, 4), text, fill=color, font=f)
    return im


def main():
    cells = []
    if os.path.exists(BEFORE):
        cells.append(label(load_fit(BEFORE), "DreamShaper立绘(现·诡异)"))
    else:
        print(f"[WARN] before 缺: {BEFORE}")
    cands = sorted(glob.glob(f"{AFTER_DIR}/manju_xiaoya_manga_*.png"), key=os.path.getmtime, reverse=True)
    print(f"[after] 候选(按新→旧): {[os.path.basename(c) for c in cands[:6]]}")
    for i, c in enumerate(cands[:3]):
        cells.append(label(load_fit(c), f"Counterfeit 候选{i+1}"))
    if not cells:
        print("[ERR] 无图可拼")
        return 1
    w = sum(c.size[0] for c in cells) + PAD * (len(cells) + 1)
    h = max(c.size[1] for c in cells) + PAD * 2
    canvas = Image.new("RGB", (w, h), (40, 40, 40))
    x = PAD
    for c in cells:
        canvas.paste(c, (x, PAD))
        x += c.size[0] + PAD
    canvas.save(OUT, quality=92)
    print(f"\n[saved] {OUT}  {canvas.size}")
    print("[next] 用户挑: Counterfeit 哪张好看? → 复制为 assets/characters/小雅立绘.png → 重生 10 镜关键帧")


if __name__ == "__main__":
    main()
