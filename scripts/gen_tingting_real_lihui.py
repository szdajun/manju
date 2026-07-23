"""写实婷婷立绘(配角, 2026-07-22 Phase 2)。

ep02 双锁脸治本: 婷婷原本无立绘(配角省资产), Wan i2v 未锁随 seed 漂 → 末镜眼瑕。
现给婷婷生干净立绘, 双人镜走双 IPAdapter 各锁一边脸。
**指纹必须与小雅可区分**(波波头+眼镜+更成熟), 否则双锁会把两张脸锁混。
同反厚下眼睑 NEG(2026-07-22 治本, 与 gen_xiaoya_real_lihui 一致)。

用法: .venv/Scripts/python.exe scripts/gen_tingting_real_lihui.py --seeds 6
产物: output/tingting_real/manju_tingting_real_seedNN.png(候选) → 挑最佳复制 婷婷立绘写实.png
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.comfyui_client import ComfyUIClient, build_txt2img_workflow  # noqa: E402
from lib.ep02_shots import TINGTING  # noqa: E402

CKPT = "DreamShaper_8_pruned.safetensors"
OUT = "../manju/output/tingting_real"
PREFIX = "manju_tingting_real"

POS = (
    f"photorealistic portrait photograph of {TINGTING}, "
    "front-facing half body, looking at camera, "
    "real photograph, 50mm lens, cinematic soft lighting, shallow depth of field, "
    "detailed skin texture, plain neutral background, vertical portrait, 8k uhd, sharp focus, "
    "bright energetic fresh eyes, thin smooth natural lower eyelids, clean under-eye area, "
    "well-rested fresh youthful look, bright alert eyes, smooth lower eyelid contour"
)
# 反厚下眼睑(2026-07-22 治本, 与 gen_xiaoya_real_lihui 一致): 立绘源头堵, 别让 IPAdapter 烙厚眼睑。
NEG = (
    "illustration, anime, cartoon, painting, 3d render, cg, digital art, "
    "text, watermark, signature, extra fingers, deformed hands, mutated, bad anatomy, "
    "blurry, oversaturated, plastic skin, airbrushed, "
    "puffy under-eye, thick lower eyelid, heavy lower eyelid, prominent lower eyelid, "
    "heavy eyelids, hooded eyelids, drooping eyelids, sagging eyelids, puffy eyelids, "
    "eye bags, bags under eyes, prominent eye bags, dark circles, tired eyes, "
    "lower eyelid too high, sanpakugan, sclera showing, uneven eyelids, squinting"
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
        seed = 3000 + i * 7   # 3000 族(避开小雅 2000 族), 可复现
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
    print(f"\n[next] 挑脸最干净+最写实+最区别于小雅(波波头/眼镜)的一张 → 复制为 婷婷立绘写实.png")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
