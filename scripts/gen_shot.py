"""manju 单镜生图入口(本地 ComfyUI + IPAdapter-Plus 锁脸)。

用法:
    uv run python scripts/gen_shot.py                       # 默认跑 ep01 镜4
    uv run python scripts/gen_shot.py --shot 7              # 换镜(prompt 字典里加)
    uv run python scripts/gen_shot.py --positive "..." --weight 0.9 --seed 42

依赖: lib/comfyui_client.py(零第三方依赖)。需本地 ComfyUI 跑在 127.0.0.1:8188。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.comfyui_client import ComfyUIClient, build_face_shot_workflow  # noqa: E402

REF_DEFAULT = "../manju/assets/characters/主角立绘.png"
OUT_DEFAULT = "../manju/output/ep01_test"

# 角色指纹(每镜都带, 锁服装/发型/道具, 配合 IPAdapter 锁脸)
CHAR = ("1boy, solo, weary exhausted salaryman, deep dark circles under eyes, "
        "messy unkempt short black hair, pale haggard face, light blue plaid shirt, "
        "blue lanyard with id badge around neck, thin frail body")
STYLE = ("japanese manga style, thick black lineart, flat shading, high contrast, "
         "simple background, humorous, exhausted, sweat drops, comic, vertical composition")

SHOTS = {
    1: ("close-up, dimly lit office late at night, cold blue glow from computer screen lighting up "
        "his exhausted face, half-empty cup of cold coffee on the desk, screen showing a late-night clock, dark office"),
    2: ("medium shot, slumped back dead in an office chair, id lanyard hanging down, "
        "motivational poster on the wall behind him, completely drained and exhausted, office"),
    3: ("wide shot, standing at the entrance of a bright gym, muscular bodybuilders lifting weights in the background, "
        "protagonist looks small and out of place, still wearing his office id badge around his neck, gym entrance"),
    4: ("face turning red straining to lift a small dumbbell with both hands, "
        "background muscular bodybuilder easily curling a large dumbbell one-handed, gym"),
    5: ("medium shot, falling asleep on a running treadmill, eyes closed, body tilting backwards about to fall off, "
        "comedic dynamic falling pose, treadmill gym"),
    6: ("close-up, pouring white protein powder into a coffee mug and stirring it, drinking a sip with whole face "
        "scrunched up in disgust, exaggerated painful grimace"),
    7: ("doing plank pose, whole body shaking violently and trembling, timer on the floor showing 3 seconds, "
        "a fly has landed on the tip of his nose, gym floor"),
    8: ("low angle, collapsed flat on his back on the gym floor with arms and legs spread, "
        "a muscular male trainer in sportswear looking down at him, protagonist weakly giving a thumbs up from the floor"),
    9: ("medium shot, looking at a message notification on his smartphone, standing in the gym, "
        "expression both despairing and relieved, holding phone"),
    10: ("close-up, sitting at an office desk but wearing a sports sweatband on his forehead, "
         "a small dumbbell placed beside the desk, determined yet helpless eyes, office desk"),
}

NEG_DEFAULT = ("text, watermark, logo, signature, extra fingers, deformed hands, mutated hands, "
               "bad anatomy, extra limbs, blurry, realistic, photorealistic, 3d, photograph, "
               "lowres, jpeg artifacts, multiple views, reference sheet")


def build_positive(shot, extra=""):
    return f"{CHAR}, {SHOTS[shot]}, {STYLE}" + (f", {extra}" if extra else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=REF_DEFAULT, help="参考脸图(立绘)")
    ap.add_argument("--shot", type=int, default=4, choices=list(SHOTS.keys()))
    ap.add_argument("--positive", default=None, help="覆盖 positive(忽略 --shot)")
    ap.add_argument("--negative", default=NEG_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--weight", type=float, default=0.8, help="IPAdapter 锁脸强度 0-1")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=912)
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    args = ap.parse_args()

    positive = args.positive or build_positive(args.shot)
    print(f"[shot {args.shot}] positive=\n  {positive}\n")

    c = ComfyUIClient(args.server)
    ref_name = c.upload_image(args.ref)
    print(f"[upload] ref -> {ref_name}")

    wf = build_face_shot_workflow(
        ref_name, positive, args.negative,
        seed=args.seed, ipadapter_weight=args.weight,
        width=args.width, height=args.height,
        filename_prefix=f"ep01_shot{args.shot:02d}",
    )
    print("[queue] generating (首次加载 ckpt+ViT-H 较慢, 请稍候)...")
    outs = c.generate(wf, args.out)
    print(f"[done] -> {outs}")
    return outs


if __name__ == "__main__":
    main()
