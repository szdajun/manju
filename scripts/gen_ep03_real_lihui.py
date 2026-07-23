"""写实林夏立绘(ep03 主角, 2026-07-23)。

ep03 单人 5 镜短剧, 验证"双 IPAdapter 堆叠 (脸 ref 0.95 + 全身 ref 0.70) 锁衣服"。
立绘是 IPAdapter 全身 ref 的源 (512×912 原图保留脸+衣+鞋完整)。
另用 crop_face.py 得 linxia_real_face.png 给脸 ref。

服装指纹 (钉死, 跨镜必须一致):
- 上衣: 雾蓝色长袖运动上衣, 左胸小型白色弧线标志
- 裤子: 黑色高腰运动裤, 裤侧一条细灰线
- 鞋子: 白色跑鞋
- 发型: 黑色高马尾
- 年龄: 26-30

种子族 5000+i*7 (避开 ep02 小雅 7000 族 + 婷婷 3000 族 + ep01 主角 1000 族)。

用法: .venv/Scripts/python.exe scripts/gen_ep03_real_lihui.py --seeds 8
产物: output/ep03_lihui/manju_linxia_real_seedNN.png (候选)
下一步: 用户 eyeball 挑 1 张最干净 + 最写实 + 服装最完整的 → 复制为 linxia_real.png
       再用 crop_face.py 得 linxia_real_face.png (脸 ref)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.comfyui_client import ComfyUIClient, build_txt2img_workflow  # noqa: E402
from lib.ep03_shots import CHAR, REAL_NEG  # noqa: E402

CKPT = "DreamShaper_8_pruned.safetensors"
OUT = "../manju/output/ep03_lihui"
PREFIX = "manju_linxia_real"

POS = (
    f"photorealistic portrait photograph of {CHAR}, "
    "standing upright full body pose, "
    "standing in a relaxed confident posture with both feet visible, weight evenly distributed, "
    "looking directly at the camera, "
    "NOT sitting, NOT cross-legged, NOT kneeling, NOT crouching, "
    "real photograph, 50mm lens, cinematic soft lighting, shallow depth of field, "
    "detailed skin texture, plain neutral background, vertical full body portrait, 8k uhd, sharp focus, "
    "bright energetic fresh eyes, thin smooth natural lower eyelids, clean under-eye area, "
    "well-rested fresh youthful look, bright alert eyes, smooth lower eyelid contour, "
    "(standing full body pose:1.3), (both feet visible:1.2), (looking at camera:1.2), "
    "(misty-blue long-sleeve athletic top:1.4), (white arc logo on left chest:1.3), "
    "(black high-waist athletic pants:1.3), (thin gray side stripe on pants:1.2), "
    "(white running shoes:1.2), (high ponytail:1.2)"
)
# 反厚下眼睑 + 反换衣 (2026-07-23, 与 ep02 婷婷同款 NEG + 加 color shift / outfit change 反锁衣失败)
NEG = REAL_NEG + (
    ", heavy eyelids, hooded eyelids, drooping eyelids, sagging eyelids, puffy eyelids, "
    "lower eyelid too high, sanpakugan, sclera showing, uneven eyelids, squinting"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8, help="候选数量 (推荐 8)")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    c = ComfyUIClient(args.server)

    cands = []
    for i in range(args.seeds):
        seed = 5000 + i * 7   # 5000 族 (避开 ep01 1000 / 婷婷 3000 / 小雅 7000), 可复现
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
    print(f"\n[next] 挑脸最干净+最写实+服装最完整 (雾蓝上衣+白弧标志 / 黑色高腰+灰侧线 / 白鞋) 的一张")
    print(f"       复制为 {OUT}/linxia_real.png → 用 crop_face.py 得 linxia_real_face.png (脸 ref)")
    print(f"       全身 ref 直接用原图 linxia_real.png (512×912, 不裁, 保留脸+衣+鞋完整)")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)