"""ep04 LoRA A/B/C 验证脚本 (2026-07-23, gitignore 探针)。
   支持 xiaoya + linxia 双角色 (--character 切换)。

A/B/C 三组, 5 镜各 1 seed = 15 张关键帧:
- A: 无 LoRA + 单 IPAdapter face 0.95 (baseline)
- B: LoRA + 单 IPAdapter face 0.95
- C: LoRA + 双 IPAdapter face 0.95 + body 0.70

用法:
  # 小雅 (复现 ep02 路径)
  python scripts/gen_ep04_lora_ablation.py --character xiaoya --modes A,B,C
  # 林夏 (复现 ep03 路径)
  python scripts/gen_ep04_lora_ablation.py --character linxia --modes A,B,C --lora linxia_lora.safetensors

上传 refs 时记得:
  python scripts/gen_ep04_lora_ablation.py --character linxia --upload-refs
"""
import argparse
import os
import sys

MANJU = "../manju"
sys.path.insert(0, MANJU)
sys.path.insert(0, os.path.join(MANJU, "scripts"))
from lib.comfyui_client import (  # noqa: E402
    ComfyUIClient, build_face_shot_workflow, build_face_body_shot_workflow)
from lib.ep04_shots import (  # noqa: E402
    CHAR_BY_NAME, SHOTS_BY_NAME, LINES_BY_NAME, I2V_MOTION_BY_NAME,
    SPEAKER_BY_NAME, REAL_STYLE, REAL_NEG,
)

SERVER = "http://127.0.0.1:8188"
OUT = "../manju/output/ep04_real"
KW, KH = 480, 832

# 角色 → LoRA 默认名 (训练后的 safetensors 在 ../ComfyUI/models/loras/)
LORA_NAME_DEFAULT = {
    "xiaoya": "xiaoya_lora.safetensors",
    "linxia": "linxia_lora.safetensors",
}

# 角色 → 立绘参考图 (对应 IPAdapter 立绘源)
# LoadImage 节点 input 期望 filename-only (相对 ComfyUI/input/), 跑前 --upload-refs 上传
REFS_BY_NAME = {
    "xiaoya": {
        "face": "小雅立绘写实_face.png",
        "body": "小雅立绘写实.png",
    },
    "linxia": {
        "face": "linxia_real_face.png",
        "body": "linxia_real.png",
    },
}

# 角色 → I2V motion 强制 clothes 不变式 (Seeds 1.0 pro text 引导, 防衣服漂移)
I2V_CLOTHES_LOCK = {
    "xiaoya": "the white office blouse stays white throughout, the pencil skirt stays the same throughout",
    "linxia": (
        "the misty-blue athletic top stays misty-blue throughout, "
        "the white arc logo on the left chest stays visible throughout, "
        "the black high-waist athletic pants stay black throughout, "
        "the thin gray side stripe stays visible throughout, "
        "the white running shoes stay white throughout, the high ponytail stays high throughout"
    ),
}

MODES = {"A": "baseline", "B": "lora_only", "C": "lora_dual"}


def build_positive(character: str, shot: int, with_lora: bool) -> str:
    """关键帧正向 prompt = 该镜场景 + 角色 CHAR + trigger word (LoRA 激活) + 风格."""
    scene = SHOTS_BY_NAME[character][shot]
    char = CHAR_BY_NAME[character]
    trigger = "" if not with_lora else f", {character}"  # trigger word 紧跟 CHAR, 防被 base 知识稀释
    return f"{scene}, {char}{trigger}, {REAL_STYLE}"


def gen_for_mode(c, character, mode, shots, lora_name, weights=(0.95, 0.70), out_dir=None):
    """生成 mode 单组关键帧 (5 镜 × 1 seed = 5 张)."""
    out_dir = out_dir or os.path.join(OUT, character, f"_ablation_{mode}")
    os.makedirs(out_dir, exist_ok=True)
    refs = REFS_BY_NAME[character]
    for s in shots:
        pos = build_positive(character, s, with_lora=(mode != "A"))
        if mode == "A":
            wf = build_face_shot_workflow(
                ref_image_name=refs["face"],
                positive=pos, negative=REAL_NEG,
                width=KW, height=KH, seed=42,
                ipadapter_weight=weights[0],
                filename_prefix=f"ep04_{character}_{mode}_shot{s:02d}",
            )
        elif mode == "B":
            wf = build_face_shot_workflow(
                ref_image_name=refs["face"],
                positive=pos, negative=REAL_NEG,
                width=KW, height=KH, seed=42,
                ipadapter_weight=weights[0],
                filename_prefix=f"ep04_{character}_{mode}_shot{s:02d}",
            )
            # 注入 Load LoRA 节点 (ckpt → LoRA → unified → ipa)
            lora_node = {
                "class_type": "LoraLoader",
                "inputs": {"model": ["ckpt", 0], "clip": ["ckpt", 1], "lora_name": lora_name,
                           "strength_model": 0.85, "strength_clip": 0.85},
            }
            wf["lora"] = lora_node
            wf["unified"]["inputs"]["model"] = ["lora", 0]
            wf["pos"]["inputs"]["clip"] = ["lora", 1]
            wf["neg"]["inputs"]["clip"] = ["lora", 1]
        elif mode == "C":
            wf = build_face_body_shot_workflow(
                ref_face_name=refs["face"],
                ref_body_name=refs["body"],
                positive=pos, negative=REAL_NEG,
                width=KW, height=KH, seed=42,
                weight_face=weights[0], weight_body=weights[1],
                filename_prefix=f"ep04_{character}_{mode}_shot{s:02d}",
            )
            lora_node = {
                "class_type": "LoraLoader",
                "inputs": {"model": ["ckpt", 0], "clip": ["ckpt", 1], "lora_name": lora_name,
                           "strength_model": 0.85, "strength_clip": 0.85},
            }
            wf["lora"] = lora_node
            wf["unified"]["inputs"]["model"] = ["lora", 0]
            wf["pos"]["inputs"]["clip"] = ["lora", 1]
            wf["neg"]["inputs"]["clip"] = ["lora", 1]
        try:
            saved = c.generate(wf, out_dir)
            for f in saved:
                new = os.path.join(out_dir, f"shot{s:02d}.png")
                if os.path.exists(f) and not os.path.exists(new):
                    os.rename(f, new)
            print(f"[{character}/{mode} shot{s:02d}] OK -> {[os.path.basename(x) for x in saved]}", flush=True)
        except Exception as e:
            print(f"[{character}/{mode} shot{s:02d}] FAIL: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", choices=["xiaoya", "linxia"], default="xiaoya")
    ap.add_argument("--modes", default="A,B,C")
    ap.add_argument("--shots", default="1,2,3,4,5")
    ap.add_argument("--server", default=SERVER)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--lora", default=None, help="LoRA safetensors 名 (默认从 character 推)")
    ap.add_argument("--upload-refs", action="store_true",
                    help="自动上传 立绘 + face crop 到 ComfyUI input/")
    args = ap.parse_args()

    modes = [m for m in args.modes.split(",") if m in MODES]
    shots = [int(s) for s in args.shots.split(",") if s.strip()]
    if not modes:
        print(f"[ERR] --modes 必须包含 A/B/C 之一, got {args.modes}")
        return 1
    if args.character not in REFS_BY_NAME:
        print(f"[ERR] unknown --character {args.character}")
        return 1
    lora_name = args.lora or LORA_NAME_DEFAULT[args.character]

    c = ComfyUIClient(args.server)
    if args.upload_refs:
        refs_full = {
            "xiaoya": {
                "face": "../manju/output/xiaoya_real/小雅立绘写实_face.png",
                "body": "../manju/output/xiaoya_real/小雅立绘写实.png",
            },
            "linxia": {
                "face": "../manju/output/ep03_lihui/linxia_real_face.png",
                "body": "../manju/output/ep03_lihui/linxia_real.png",
            },
        }
        for k, full in refs_full[args.character].items():
            if os.path.exists(full):
                c.upload_image(full)
                print(f"[upload] {k}: {full}")
        print(f"[upload] refs for {args.character} OK")

    out_root = os.path.join(args.out, args.character)
    os.makedirs(out_root, exist_ok=True)
    for mode in modes:
        print(f"\n=== character={args.character} mode={mode} ({MODES[mode]}) lora={lora_name if mode != 'A' else None} ===")
        gen_for_mode(c, args.character, mode, shots, lora_name, out_dir=os.path.join(out_root, f"_ablation_{mode}"))
    print(f"\n[done] outputs in {out_root}/_ablation_[ABC]/")
    print(f"[next] vision 抽 contact sheet 对比 3 组服装属性清单:")
    print(f"  xiaoya: 白衬衫 + 铅笔裙 + 长直黑 (3 项)")
    print(f"  linxia: 雾蓝长袖+白弧 + 黑高腰+灰侧线 + 白鞋 + 高马尾 (4 项)")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
