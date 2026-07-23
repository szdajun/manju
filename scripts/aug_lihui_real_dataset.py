"""ep04 李辉(男角) LoRA 数据集构建 (2026-07-24).

镜像 aug_linxia_dataset.py, 改:
- SRC_DIR = output/lihui_real/ (李辉立绘候选)
- OUT_DIR = output/lihui_real_lora/ (避免覆盖 lihui_lora 命名混淆)
- TRIGGER_WORD = "lihui_real"  (与男角绑定)
- BASE_CAPTION 锁李辉服装指纹: 灰衬衫 + 黑领带 + 黑裤 + 解领带
- 增广 7 变体 (rot+/-15, mirror, upper, 3q, logo_zoom, pants_zoom)

李辉 (ep01 主角):
  - 男, 26-30, 短黑发
  - 服装: 灰衬衫/黑领带/黑裤 (ep01 v4 钉死)
  - 钩子: "我是牛马, 但我要做, 一个举过铁的牛马"

注: lihui_lora/ 输出路径给 ep01 李辉 LoRA 用, 跟 lihui_lora/ (历史给 xiaoya)
的命名路径碰巧相同但目的不同, 故 aug_lihui_real_dataset 输出到新路径 lihui_real_lora/.

用法 (ComfyUI venv, 含 PIL/cv2/numpy):
  ../ComfyUI/venv/Scripts/python.exe scripts/aug_lihui_real_dataset.py
"""
import os
import random
import sys
from pathlib import Path

from PIL import Image
import cv2
import numpy as np

MANJU = "../manju"
SRC_DIR = f"{MANJU}/output/lihui_real"
OUT_DIR = f"{MANJU}/output/lihui_real_lora"  # 新路径避免 lihui_lora 命名冲突
TRAIN_SUBDIR = "train_data"
VAL_SUBDIR = "val_data"
TRIGGER_WORD = "lihui_real"

BASE_CAPTION = (
    f"a photo of {TRIGGER_WORD} man, slim fit young Chinese man age 26-30, "
    f"messy unkempt short black hair, deep dark circles under eyes, "
    f"pale haggard face, exhausted weary look, "
    f"light blue plaid office shirt with collar, "
    f"blue lanyard with id badge around neck, "
    f"slim black dress pants, "
    f"(light blue plaid shirt clearly visible:1.3), "
    f"(blue lanyard id badge clearly visible:1.4), "
    f"(slim black dress pants clearly visible:1.4), "
    f"real photograph, sharp"
)

NEG_TERMS = "NOT gray shirt, NOT white shirt, NOT black pants to gray pants, NOT white pants, NOT light pants"

# 特写变体 caption
SHIRT_CAPTION = (
    f"closeup photo of {TRIGGER_WORD} man upper body wearing light blue plaid office shirt "
    f"with collar and blue lanyard with id badge, "
    f"(light blue plaid shirt:1.5), (blue lanyard id badge:1.5), "
    f"real photograph, sharp"
)
PANTS_CAPTION = (
    f"closeup photo of {TRIGGER_WORD} man lower body wearing slim black dress pants, "
    f"(slim black dress pants:1.5), NOT white pants, NOT light pants, NOT colored pants, "
    f"real photograph, sharp"
)


def collect_inputs() -> list[Path]:
    paths = []
    for n in range(1, 10):
        p = Path(SRC_DIR) / f"manju_lihui_real_{n:05d}_.png"
        if p.exists():
            paths.append(p)
    canon = Path(SRC_DIR) / "主角立绘写实.png"
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


def aug_shirt_zoom(img: Image.Image) -> Image.Image:
    """裁衬衫领口 + 领带区域 (上半中段)."""
    w, h = img.size
    cx, cy = w // 2, int(h * 0.30)  # 领口区域
    half = 150
    crop = img.crop((max(0, cx - half), max(0, cy - half), min(w, cx + half), min(h, cy + half)))
    return crop.resize((512, 512), Image.LANCZOS)


def aug_pants_zoom(img: Image.Image) -> Image.Image:
    """裁裤区中段. 李辉是 512x912, 裤在 h*0.50-0.75."""
    w, h = img.size
    cx, cy = w // 2, int(h * 0.62)
    half = 180
    crop = img.crop((max(0, cx - half), max(0, cy - half), min(w, cx + half), min(h, cy + half)))
    return crop.resize((512, 512), Image.LANCZOS)


def gen_variants(img: Image.Image, src_stem: str) -> list[tuple]:
    """1 张 -> 5 基础 + 2 特写 = 7 变体."""
    variants = []
    variants.append((aug_rotate(img, +15), f"{src_stem}_rot+15", f"{BASE_CAPTION}, rotated +15 degrees. {NEG_TERMS}"))
    variants.append((aug_rotate(img, -15), f"{src_stem}_rot-15", f"{BASE_CAPTION}, rotated -15 degrees. {NEG_TERMS}"))
    variants.append((aug_mirror(img), f"{src_stem}_mirror", f"{BASE_CAPTION}, horizontally mirrored. {NEG_TERMS}"))
    variants.append((aug_crop_upper(img), f"{src_stem}_upper", f"{BASE_CAPTION}, upper body shot. {NEG_TERMS}"))
    variants.append((aug_crop_3q(img), f"{src_stem}_3q", f"{BASE_CAPTION}, three quarter view. {NEG_TERMS}"))
    variants.append((aug_shirt_zoom(img), f"{src_stem}_shirt", SHIRT_CAPTION))
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
    train_root = os.path.join(OUT_DIR, TRAIN_SUBDIR, "10_lihui_real")
    val_root = os.path.join(OUT_DIR, VAL_SUBDIR, "10_lihui_real")
    os.makedirs(train_root, exist_ok=True)
    os.makedirs(val_root, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, TRAIN_SUBDIR), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, VAL_SUBDIR), exist_ok=True)

    inputs = collect_inputs()
    print(f"[input] {len(inputs)} 张原图")
    if not inputs:
        print("[ERR] no lihui 立绘. Run scripts/gen_lihui_real.py (or similar) first.")
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
        src_stem = src.stem.replace("manju_lihui_real_", "src").replace("主角立绘写实", "canon")
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

    print(f"\n[done] train={n_train}  val={n_val}  ->  {OUT_DIR}/train_data/10_lihui_real/")
    print(f"[next] Run scripts/run_lihui_real_lora_train.bat (kohya_ss).")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
