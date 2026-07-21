"""写实立绘生成(SD1.5 写真向底模 txt2img, 多 seed 抽, 选脸最干净那张当脸参)。

本地 Wan2.2-5B 死路后写实链路: 写实立绘(本脚本) → [关键帧] → Seedance i2v。
立绘是脸源(全合成), 直接喂 Seedance 验真人脸政策+运镜+face_sim。
底模 DreamShaper_8(写实向, 现有) + 强 photoreal prompt; 出图偏画风则换 Realistic Vision V6。

用法: uv run python scripts/gen_real_lihui.py --seeds 6
产物: output/lihui_real/lihui_seedNN.png(候选) → 挑最佳复制 主角立绘写实.png
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.comfyui_client import ComfyUIClient, build_txt2img_workflow  # noqa: E402

CKPT = "DreamShaper_8_pruned.safetensors"
OUT = "../manju/output/lihui_real"
PREFIX = "manju_lihui_real"

POS = (
    "photorealistic portrait photograph of a 30-year-old Chinese male office worker, "
    "front-facing half body, messy unkempt short hair, heavy dark eye circles, pale tired face, "
    "light blue plaid shirt, blue lanyard with small ID badge around neck, slightly thin build, "
    "exhausted lifeless resigned expression, looking at camera, "
    "real photograph, 50mm lens, cinematic soft lighting, shallow depth of field, "
    "detailed skin texture, plain neutral background, vertical portrait, 8k uhd, sharp focus"
)
NEG = (
    "illustration, anime, cartoon, painting, 3d render, cg, digital art, "
    "text, watermark, signature, extra fingers, deformed hands, mutated, bad anatomy, "
    "blurry, oversaturated, plastic skin, airbrushed"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    c = ComfyUIClient(args.server)

    cands = []
    for i in range(args.seeds):
        seed = 1000 + i * 7   # 固定可复现种子族
        wf = build_txt2img_workflow(
            POS, NEG, ckpt=args.ckpt, width=512, height=912, seed=seed,
            steps=32, cfg=6.5, filename_prefix=PREFIX)
        try:
            saved = c.generate(wf, OUT)
            cands.extend(saved)
            print(f"[seed {seed}] -> {[os.path.basename(x) for x in saved]}", flush=True)
        except Exception as e:
            print(f"[seed {seed}] FAIL: {e}", flush=True)

    print(f"\n[cands] {len(cands)} 张:")
    for x in cands:
        print("  ", os.path.basename(x))
    print(f"\n[next] vision 挑脸最干净+最写实的一张 -> 复制为 "
          f"{os.path.join(OUT, '主角立绘写实.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
