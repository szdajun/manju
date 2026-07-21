"""ep02 关键帧 画风 before/after contact(临时诊断工具)。

旧(DreamShaper 底模+REF) vs 新(Counterfeit 底模+REF) 关键帧, 验"脸怪异"治没治。
每行 [shot{N} DreamShaper(旧·怪) | shot{N} Counterfeit(新)]。
旧 = output/ep02/ 里该 shot 最老那张(_00001_, DreamShaper 期); 新 = 最新那张(_00002_/03_, Counterfeit 期)。
"""
import glob
import os
from PIL import Image, ImageDraw, ImageFont

KF = "../manju/output/ep02"
OUT = f"{KF}/kf_style_contact.png"
SHOTS = [1, 2, 3, 4, 5, 7, 9, 10]  # 有主角脸的镜(跳 6 手部特写 / 8 时代广场人群)
TW = 280
PAD = 8


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


def main():
    rows = []
    for s in SHOTS:
        fs = sorted(glob.glob(f"{KF}/ep02_shot{s:02d}_*.png"), key=os.path.getmtime)
        if len(fs) < 2:
            print(f"[skip] shot{s}: 只有 {len(fs)} 张, 无法 before/after 对比")
            continue
        old, new = fs[0], fs[-1]
        rows.append([
            label(load_fit(old), f"shot{s} DreamShaper(旧)"),
            label(load_fit(new), f"shot{s} Counterfeit(新)"),
        ])
        print(f"[shot {s:2d}] old={os.path.basename(old)}  new={os.path.basename(new)}")
    if not rows:
        print("[ERR] 无对比行")
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
