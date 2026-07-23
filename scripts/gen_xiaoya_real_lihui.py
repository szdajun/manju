"""写实小雅立绘(漫剧→写实 pivot 第一步, 免费本地)。

ep02 写实化: DreamShaper_8 txt2img 抽 6 seed → 用户挑脸最干净+最写实的一张 → 脸参喂后续 IPAdapter 关键帧 + Seedance i2v。
女角 prompt 取自 lib/ep02_shots.CHAR(年轻中国女性/长发/白领) 转写实 photoreal。
DreamShaper_8 偏画风则换 Realistic Vision V6(--ckpt)。

用法: .venv/Scripts/python.exe scripts/gen_xiaoya_real_lihui.py --seeds 6
产物: output/xiaoya_real/manju_xiaoya_real_seedNN.png(候选) → 挑最佳复制 小雅立绘写实.png
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.comfyui_client import ComfyUIClient, build_txt2img_workflow  # noqa: E402

CKPT = "DreamShaper_8_pruned.safetensors"
OUT = "../manju/output/xiaoya_real"
PREFIX = "manju_xiaoya_real"

POS = (
    "photorealistic portrait photograph of a slim fit young Chinese woman in her mid-20s, "
    "front-facing half body, long straight black hair, bright cheerful face, confident smile, "
    "fit toned body, wearing a white office blouse and pencil skirt, looking at camera, "
    "real photograph, 50mm lens, cinematic soft lighting, shallow depth of field, "
    "detailed skin texture, plain neutral background, vertical portrait, 8k uhd, sharp focus, "
    "bright energetic fresh eyes, thin smooth natural lower eyelids, clean under-eye area, "
    "well-rested fresh youthful look, bright alert eyes, smooth lower eyelid contour, "
    # 2026-07-22 治浓妆(用户报换脸后"化妆很浓"→ inswapper 把立绘浓妆复制到每帧):
    # 素颜/淡妆正向, 把"浓妆感"从源头压掉, 否则换脸输出永远带妆。
    "bare-faced natural look, minimal no-makeup look, clean fresh bare skin, "
    "nude natural pink lips, no eyeliner, no eyeshadow, natural soft eyebrows, "
    "plain unmade-up face, healthy fresh complexion"
)
# 反厚下眼睑(2026-07-22 治本): 旧版 NEG 无眼睑反词 → DreamShaper 默认生成厚下眼睑/眼袋
# → 立绘本身厚 → IPAdapter 0.95 烙进每帧关键帧, EYE_FIX 正向词被压过。这里回到源头堵。
# 2026-07-22 同源治浓妆: 加反眼线/眼影/口红/腮红/粉底词。
NEG = (
    "illustration, anime, cartoon, painting, 3d render, cg, digital art, "
    "text, watermark, signature, extra fingers, deformed hands, mutated, bad anatomy, "
    "blurry, oversaturated, plastic skin, airbrushed, "
    "puffy under-eye, thick lower eyelid, heavy lower eyelid, prominent lower eyelid, "
    "heavy eyelids, hooded eyelids, drooping eyelids, sagging eyelids, puffy eyelids, "
    "eye bags, bags under eyes, prominent eye bags, dark circles, tired eyes, "
    "lower eyelid too high, sanpakugan, sclera showing, uneven eyelids, squinting, "
    "heavy makeup, eyeliner, eyeshadow, lipstick, red lips, bright lipstick, blush, "
    "contouring, heavy foundation, drawn-on eyebrows, dramatic eye makeup, smokey eyes, "
    "false eyelashes, lip gloss, mascara, made-up face, visible cosmetics"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--seed-base", type=int, default=7000,
                    help="seed 起始(7000=2026-07-22 淡妆批, 避开旧 2000 族)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    c = ComfyUIClient(args.server)

    cands = []
    for i in range(args.seeds):
        seed = args.seed_base + i * 7
        wf = build_txt2img_workflow(
            POS, NEG, ckpt=args.ckpt, width=512, height=912, seed=seed,
            steps=32, cfg=6.5, filename_prefix=PREFIX)
        try:
            saved = c.generate(wf, OUT)
            cands.extend(saved)
            print(f"[seed {seed}] -> {[os.path.basename(x) for x in saved]}", flush=True)
        except Exception as e:
            print(f"[seed {seed}] FAIL: {e}", flush=True)

    print(f"\n[cands] {len(cands)} 张 in {OUT}:")
    for x in cands:
        print("  ", os.path.basename(x))
    print(f"\n[next] vision 挑脸最干净+最写实的一张 → 复制为 小雅立绘写实.png(脸参)")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
