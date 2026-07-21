"""ep02 小雅漫画立绘生成(SD1.5 DreamShaper txt2img manga, 多 seed 抽, det_score 挑脸)。

漫画版先行: DreamShaper + 强 manga prompt 出小雅女角立绘 → IPAdapter REF(ep02 关键帧锁脸源)。
底模 DreamShaper_8(同 gen_shot 关键帧, STYLE 强推 manga 已验证能出漫风) + manga prompt; 复用 gen_real_lihui 流程。

用法: .venv/Scripts/python.exe scripts/gen_xiaoya_manga.py --seeds 6
产物: output/xiaoya_manga/xiaoya_seedNN.png(候选) → det_score 挑最佳 → 复制 assets/characters/小雅立绘.png
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.comfyui_client import ComfyUIClient, build_txt2img_workflow  # noqa: E402

CKPT = "DreamShaper_8_pruned.safetensors"
OUT = "../manju/output/xiaoya_manga"
PREFIX = "manju_xiaoya_manga"

# 小雅漫画立绘(正脸半身, fit toned 健身后好身材, 锁发型/服装当 IPAdapter 指纹)
POS = (
    "masterpiece, best quality, highly detailed, japanese manga style character portrait of a young Chinese woman, "
    "front-facing half body, long straight black hair, eyes half-closed, smug proud smile, confident haughty expression, "
    "proud beautiful woman, fit toned slim body, "
    "wearing a white office blouse and pencil skirt, thick black lineart, flat shading, high contrast, "
    "clean simple light background, looking at camera, vertical manga portrait, sharp clean lines"
)
NEG = (
    "realistic, photorealistic, 3d, photograph, painting, western cartoon, chibi, "
    "text, watermark, signature, extra fingers, deformed hands, mutated, bad anatomy, "
    "blurry, oversaturated, multiple views, reference sheet, lowres"
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
        seed = 2000 + i * 7   # 固定可复现种子族(避开 lihui 写实立绘用的 1000 族)
        wf = build_txt2img_workflow(
            POS, NEG, ckpt=args.ckpt, width=512, height=912, seed=seed,
            steps=32, cfg=6.5, filename_prefix=PREFIX)
        try:
            saved = c.generate(wf, OUT)
            cands.extend(saved)
            print(f"[seed {seed}] -> {[os.path.basename(x) for x in saved]}", flush=True)
        except Exception as e:
            print(f"[seed {seed}] FAIL: {e}", flush=True)

    print(f"\n[cands] {len(cands)} 张 in {OUT}")
    print("[next] det_score 挑脸最干净+最贴小雅的一张 -> 复制为 assets/characters/小雅立绘.png")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
