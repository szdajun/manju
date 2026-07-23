"""ep04 林夏 LoRA 数据集构建 (2026-07-23)。

复用 _aug_lihui_dataset.py 的 5 变体 + caption 套路, 改:
- SRC_DIR = output/ep03_lihui/ (林夏立绘候选)
- OUT_DIR = output/linxia_lora/ (林夏专用, 不覆盖小雅)
- TRIGGER_WORD = "linxia"
- BASE_CAPTION 锁林夏服装指纹 (雾蓝长袖+白弧 + 黑高腰+灰侧线 + 白鞋 + 高马尾)

stage 1: 用 gen_ep03_real_lihui.py 生 8-16 张候选立绘 (需 ComfyUI 8188 + GPU 闲, 12 min)
stage 2: 用户 eyeball 挑 1 张定档 (+ 可多选几张候选做多角度基底)
stage 3: 跑本脚本 → train_data/10_linxia/ + val_data/10_linxia/
stage 4: 跑 run_linxia_lora_train.bat (kohya_ss, 需 GPU, ~11 min)
stage 5: 跑 gen_ep04_lora_ablation.py --modes A,B,C 评估 LoRA 效果

用法 (ComfyUI venv, 含 PIL/cv2/numpy):
  ../ComfyUI/venv/Scripts/python.exe scripts/_aug_linxia_dataset.py
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
    f"misty-blue long-sleeve athletic top, white arc logo on left chest, "
    f"black high-waist athletic pants, thin gray side stripe on pants, "
    f"white running shoes"
)


def collect_inputs() -> list[Path]:
    paths = []
    # 林夏立绘候选 ep03 阶段已生成的 (manju_linxia_real_00000_ ~)
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


def aug_crop(img: Image.Image, mode: str) -> Image.Image:
    w, h = img.size
    if mode == "upper_body":
        return img.crop((0, 0, w, int(h * 0.55)))  # 林夏是 512x912, 上半身 55% 含马尾+胸+臂
    elif mode == "full_body":
        return img.copy()
    elif mode == "three_quarter":
        margin = int(w * 0.125)
        return img.crop((margin, 0, w - margin, h))
    else:
        raise ValueError(f"unknown crop mode: {mode}")


def gen_variants(img: Image.Image, src_stem: str) -> list[tuple[Image.Image, str, str]]:
    variants = []
    variants.append((aug_rotate(img, +15), f"{src_stem}_rot+15", "rotated +15 degrees"))
    variants.append((aug_rotate(img, -15), f"{src_stem}_rot-15", "rotated -15 degrees"))
    variants.append((aug_mirror(img), f"{src_stem}_mirror", "horizontally mirrored"))
    variants.append((aug_crop(img, "upper_body"), f"{src_stem}_upper", "upper body shot"))
    variants.append((aug_crop(img, "three_quarter"), f"{src_stem}_3q", "three quarter view"))
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
    # kohya 要求 class-folder 结构: train_data/{class_name}/*.png
    # 类似 10_xiaoya 那条: 10 = repeat steps (一个 epoch 重复 10 次, sd15 LoRA 标准), xiaoya = class name
    train_root = os.path.join(OUT_DIR, TRAIN_SUBDIR, "10_linxia")
    val_root = os.path.join(OUT_DIR, VAL_SUBDIR, "10_linxia")
    os.makedirs(train_root, exist_ok=True)
    os.makedirs(val_root, exist_ok=True)
    # 把 train/val_data 原本直接放图的结构改成 10_linxia/ 子目录
    # 实际写盘时直接放 train_data/10_linxia/, 不再需要再包一层
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
    print(f"[split] train={len(inputs) - len(val_set)}  val={len(val_set)}")

    n_train, n_val = 0, 0
    for i, src in enumerate(inputs):
        try:
            img = Image.open(src)
        except Exception as e:
            print(f"[skip] {src.name}: {e}")
            continue
        subdir = VAL_SUBDIR if i in val_set else TRAIN_SUBDIR
        src_stem = src.stem.replace("manju_linxia_real_", "src").replace("linxia_real", "canon")
        for var_img, var_stem, var_suffix in gen_variants(img, src_stem):
            # kohya 期望结构: {train,val}_data/10_linxia/{filename}.png
            class_root = train_root if subdir == TRAIN_SUBDIR else val_root
            out_path = Path(class_root) / f"{var_stem}.png"
            caption = f"{BASE_CAPTION}, {var_suffix}"
            try:
                save_variant(var_img, caption, out_path)
            except Exception as e:
                print(f"[FAIL] {out_path}: {e}")
                continue
            if subdir == TRAIN_SUBDIR:
                n_train += 1
            else:
                n_val += 1
        print(f"[{subdir[:5]}] {src.name} -> 5 变体")

    print(f"\n[done] train={n_train}  val={n_val}  ->  {OUT_DIR}/train_data/10_linxia/ + {OUT_DIR}/val_data/10_linxia/")
    print(f"[next] 跑 run_linxia_lora_train.bat (kohya_ss --train_data_dir {OUT_DIR}/train_data)")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
