"""Generate LoRA training dataset from ref.png via mirror + rotate + crops."""
from PIL import Image, ImageOps
from pathlib import Path

REF = Path("assets/characters/kid_dance/ref.png")
OUT_DIR = Path("assets/characters/kid_dance/train_data/10_kid_dance")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    img = Image.open(REF).convert("RGB")
    base = OUT_DIR / "00_ref.png"
    img.save(base)
    variants = [
        ("01_mirror_h", ImageOps.mirror(img)),
        ("02_rot_-15", img.rotate(-15, fillcolor=(255, 255, 255))),
        ("03_rot_+15", img.rotate(+15, fillcolor=(255, 255, 255))),
        ("04_upper_half", img.crop((0, 0, img.width, img.height // 2 + 50))),
        ("05_three_quarter", img.crop((img.width // 8, 0, img.width, img.height))),
    ]
    for name, im in variants:
        im.save(OUT_DIR / f"{name}.png")
    # 再镜像 5 个变体各 1 份 -> 10 张; 3q 改色温 x 暖/冷 -> +4 张; = 14 张
    for name, im in variants:
        ImageOps.mirror(im).save(OUT_DIR / f"{name}_m.png")
    # 暖色温: ImageOps.colorize 需要 L-mode 输入, 先转灰再染色合成
    for name, im in variants[:3]:
        gray = ImageOps.grayscale(im)
        warm = ImageOps.colorize(gray, black=(20, 0, 0), white=(255, 200, 180))
        warm.save(OUT_DIR / f"{name}_warm.png")
    print(f"Wrote {len(list(OUT_DIR.glob('*.png')))} images to {OUT_DIR}")

if __name__ == "__main__":
    main()
