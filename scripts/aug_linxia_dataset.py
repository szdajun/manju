"""ep04 林夏 LoRA 数据集构建 v2 (2026-07-24)。
   改进: 加 logo_zoom + pants_zoom 特写变体 + caption 强化 logo 可见性。

数据增强 pipeline (从 1 张原图):
  5 基础变体 (rot+/-15, mirror, upper, 3q) + 2 新特写变体:
    - logo_zoom: 裁左胸 logo 区域 (180x180) 放大到 384x384, LoRA 反复看 logo
    - pants_zoom: 裁裤区中段 (~256x256) 放大到 384x384, LoRA 反复看裤
  caption:
    - 特写变体: 强化 'white arc logo on left chest clearly visible' / 'black high-waist pants with thin gray side stripe clearly visible'
    - 反向: NOT white pants, NOT light pants, NOT pale pants (防 LoRA 反学白裤)
"""
import os
import random
import sys
from pathlib import Path

from PIL import Image
import cv2
import numpy as np

MANJU = "../manju"
SRC_DIR = f"{MANJU}/output/ep03_lihui"
OUT_DIR = f"{MANJU}/output/linxia_lora"
TRAIN_SUBDIR = "train_data"
VAL_SUBDIR = "val_data"
TRIGGER_WORD = "linxia"

BASE_CAPTION = (
    f"a photo of {TRIGGER_WORD} woman, slim fit young Chinese woman age 26-30, "
    f"black hair in high ponytail, bright energetic eyes, "
    f"(misty-blue long-sleeve athletic top with white arc logo on left chest:1.3), "
    f"(white arc logo on left chest clearly visible:1.4), "
    f"(black high-waist athletic pants:1.3), "
    f"(thin gray side stripe on pants:1.3), "
    f"(black high-waist athletic pants with thin gray side stripe clearly visible:1.4), "
    f"white running shoes"
)

# 反向 caption (NEG 嵌入, LoRA 学习时强推不学这些)
NEG_TERMS = "NOT white pants, NOT light pants, NOT pale pants, NOT gray pants, NOT pastel pants"

# 特写变体独立 caption (忽略通用描述, 集中注意力在学细节)
LOGO_CAPTION = (
    f"closeup photo of {TRIGGER_WORD} woman upper chest, "
    f"misty-blue long-sleeve athletic top, "
    f"white arc logo on left chest CLEARLY VISIBLE, "
    f"(white arc logo:1.5), prominent small triangular arc mark on left chest, "
    f"real photograph, sharp focus"
)
PANTS_CAPTION = (
    f"photo of {TRIGGER_WORD} woman lower body, "
    f"BLACK HIGH-WAIST athletic pants, "
    f"thin GRAY SIDE STRIPE clearly visible running down outside leg, "
    f"(black high-waist athletic pants:1.5), "
    f"(thin gray side stripe on pants:1.5), "
    f"NOT white pants, NOT light pants, NOT pale pants, "
    f"real photograph, sharp focus"
)


def collect_inputs() -> list[Path]:
    paths = []
    for n in range(0, 16):
        p = Path(SRC_DIR) / f"manju_linxia_real_{n:05d}_.png"
        if p.exists():
            paths.append(p)
    canon = Path(SRC_DIR) / "linxia_real.png"
    if canon.exists():
        paths.append(canon)
    return paths


def aug_rotate(img: Image.Image, angle: float) -> Image.Image:
    return img.rotate(angle, resample=Image.BICUBIC, expand=True)


def aug_mirror(img: Image.Image) -> Image.Image:
    return Image.Image.transpose(img, method=Image.FLIP_LEFT_RIGHT)


def aug_crop_upper(img: Image.Image) -> Image.Image:
    w, h = img.size
    return img.crop((0, 0, w, int(h * 0.55)))


def aug_crop_3q(img: Image.Image) -> Image.Image:
    w, h = img.size
    margin = int(w * 0.125)
    return img.crop((margin, 0, w - margin, h))


def aug_logo_zoom(img: Image.Image) -> Image.Image:
    """裁左胸 logo 区域并放大到 512x512.
    立绘 512x912: 左胸 logo 约在 cx=200, cy=320 区域 (胸下沿).
    取 +/-150 around center, 放出 512x512 让 LoRA 看清 logo.
    注意: 这是右手位置 (即 left chest 当 person 自己视角的左胸 = 图的右胸)
    -> 用 right chest on image coords (x > w/2).
    实际上立绘构图里 logo 在 cx~190, 也就是 image 的 left chest (相对 viewer 是 image's right).
    """
    w, h = img.size
    # 1) 立绘构图: logo 是 viewer 视角的右侧 (左胸 when face front)
    # 但 base caption 写 'left chest', kohya 会从 pix 推到语义, 反正 caption 写 left 它学 left
    # 实证: 截 w*0.6 ~ w*0.85, h*0.30 ~ h*0.50 这段 (image right mid)
    cx = int(w * 0.70)  # image right (viewer 视角)
    cy = int(h * 0.40)  # 胸下沿
    half = 96
    crop = img.crop((max(0, cx - half), max(0, cy - half), min(w, cx + half), min(h, cy + half)))
    # resize to 512x512 - 即便 crop 小, 放大后 LoRA 看清 logo
    return crop.resize((512, 512), Image.LANCZOS)


def aug_pants_zoom(img: Image.Image) -> Image.Image:
    """裁裤区中段, 看清黑色高腰 + 灰侧线.
    立绘 512x912: 裤在中段 h*0.50 ~ h*0.75. 取 +/-180 around (256, 600), 放大 512x512.
    """
    w, h = img.size
    cx = w // 2
    cy = int(h * 0.62)  # 中段
    half = 180
    crop = img.crop((max(0, cx - half), max(0, cy - half), min(w, cx + half), min(h, cy + half)))
    return crop.resize((512, 512), Image.LANCZOS)


def gen_variants(img: Image.Image, src_stem: str) -> list[tuple]:
    """1 张原图 -> 5 基础 + 2 特写 = 7 变体. 返回 (PIL.Image, name_stem, caption_text)."""
    variants = []
    # 基础 5 变体 (同 v1)
    variants.append((aug_rotate(img, +15), f"{src_stem}_rot+15", f"{BASE_CAPTION}, rotated +15 degrees. {NEG_TERMS}"))
    variants.append((aug_rotate(img, -15), f"{src_stem}_rot-15", f"{BASE_CAPTION}, rotated -15 degrees. {NEG_TERMS}"))
    variants.append((aug_mirror(img), f"{src_stem}_mirror", f"{BASE_CAPTION}, horizontally mirrored. {NEG_TERMS}"))
    variants.append((aug_crop_upper(img), f"{src_stem}_upper", f"{BASE_CAPTION}, upper body shot. {NEG_TERMS}"))
    variants.append((aug_crop_3q(img), f"{src_stem}_3q", f"{BASE_CAPTION}, three quarter view. {NEG_TERMS}"))
    # 新特写 2 变体 - 用独立 caption 集中描述细节
    variants.append((aug_logo_zoom(img), f"{src_stem}_logo", LOGO_CAPTION))
    variants.append((aug_pants_zoom(img), f"{src_stem}_pants", PANTS_CAPTION))
    return variants


def save_variant(img: Image.Image, caption: str, out_path: Path):
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError(f"cv2.imencode failed for {out_path}")
    buf.tofile(str(out_path))
    txt_path = out_path.with_suffix(".txt")
    txt_path.write_text(caption, encoding="utf-8")


def main():
    # kohya 期望结构: train_data/10_linxia/*.png
    train_root = os.path.join(OUT_DIR, TRAIN_SUBDIR, "10_linxia")
    val_root = os.path.join(OUT_DIR, VAL_SUBDIR, "10_linxia")
    os.makedirs(train_root, exist_ok=True)
    os.makedirs(val_root, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, TRAIN_SUBDIR), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, VAL_SUBDIR), exist_ok=True)

    inputs = collect_inputs()
    print(f"[input] {len(inputs)} 张原图")
    if not inputs:
        print("[ERR] 没找到任何林夏立绘. 先跑 stage 1: gen_ep03_real_lihui.py --seeds 8")
        return 1

    random.seed(42)
    indices = list(range(len(inputs)))
    random.shuffle(indices)
    val_set = set(indices[: max(1, len(indices) // 5)])
    print(f"[split] train={len(inputs) - len(val_set)}  val={len(val_set)} (5 + 2 closeup variants per image)")

    n_train, n_val = 0, 0
    for i, src in enumerate(inputs):
        try:
            img = Image.open(src)
        except Exception as e:
            print(f"[skip] {src.name}: {e}")
            continue
        subdir = VAL_SUBDIR if i in val_set else TRAIN_SUBDIR
        src_stem = src.stem.replace("manju_linxia_real_", "src").replace("linxia_real", "canon")
        for var_img, var_stem, caption in gen_variants(img, src_stem):
            class_root = train_root if subdir == TRAIN_SUBDIR else val_root
            out_path = Path(class_root) / f"{var_stem}.png"
            try:
                save_variant(var_img, caption, out_path)
            except Exception as e:
                print(f"[FAIL] {out_path}: {e}")
                continue
            if subdir == TRAIN_SUBDIR:
                n_train += 1
            else:
                n_val += 1
        print(f"[{subdir[:5]}] {src.name} -> 7 variants")

    print(f"\n[done] train={n_train}  val={n_val}  ->  {OUT_DIR}/train_data/10_linxia/")
    print(f"[next] 跑 run_linxia_lora_train.bat (kohya_ss, 重新训练 v2 因 caption 变)")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
