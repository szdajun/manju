#!/usr/bin/env python3
"""scripts/gen_character_assets.py — 角色定妆图生成 (P2, 资产管理环节, 2026-08-15).

从角色指纹生成 定妆图: 正面肖像 (→ crop_face 512×512 = face_ref) + 全身 (body_ref).
回填 script.json characters[KEY].face_ref/body_ref → kf 阶段 IPAdapter 零改动消费.

用法:
  python scripts/gen_character_assets.py --ep 20                        # 生成 ep20 所有缺 ref 角色
  python scripts/gen_character_assets.py --ep 20 --chars LINXIA,LICOACH
  python scripts/gen_character_assets.py --ep 20 --dry-run             # 只打印 prompt
  python scripts/gen_character_assets.py --ep 20 --seeds 3             # 每角色多 seed 生成, 手动选
"""
import argparse
import glob
import os
import subprocess
import sys

MANJU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MANJU)

from lib.comfyui_client import ComfyUIClient, build_txt2img_workflow  # noqa: E402
from lib.storyboard import dump_json, load_json_script, normalize_ep  # noqa: E402

SDXL = "sd_xl_base_1.0.safetensors"
DEFAULT_CKPT = SDXL
NEG = ("distorted, deformed, blurry, low quality, extra limbs, bad anatomy, "
       "watermark, text, cartoon, 3d render, anime")
FACE_SUFFIX = ("portrait, head and shoulders, looking at viewer, front view, "
               "plain neutral background, studio lighting, photorealistic")
BODY_SUFFIX = ("full body, standing facing camera, arms relaxed, "
               "showing complete outfit and body proportions, "
               "plain neutral background, studio lighting, photorealistic")

FACE_W, FACE_H = 768, 1024
BODY_W, BODY_H = 768, 1152

FITNESS_PY = "../fitness-video-pipeline/.venv/Scripts/python.exe"


def build_prompts(fingerprint: str):
    """指纹 → (face_prompt, body_prompt, NEG)."""
    fp = (fingerprint or "").strip()
    return (f"{fp}, {FACE_SUFFIX}" if fp else FACE_SUFFIX,
            f"{fp}, {BODY_SUFFIX}" if fp else BODY_SUFFIX,
            NEG)


def crop_face(src, dst):
    """shell 调 fitness venv crop_face.py → 512×512 脸裁."""
    subprocess.run([FITNESS_PY, os.path.join(MANJU, "scripts", "crop_face.py"),
                    src, dst], check=True, capture_output=True, text=True)


def generate(client, prompt, neg, ckpt, width, height, seed, prefix, out_dir):
    """生成单图, 返回落盘路径 (最新)."""
    wf = build_txt2img_workflow(prompt, neg, ckpt=ckpt, width=width, height=height,
                                seed=seed, steps=30, cfg=7.0,
                                filename_prefix=prefix)
    saved = client.generate(wf, out_dir)
    if not saved:
        return None
    return max(saved, key=os.path.getmtime)  # 最新产物


def _pending_chars(script, chars_arg):
    """要处理的角色: --chars 指定, 或默认缺 face_ref 的角色."""
    if chars_arg:
        return [c.strip() for c in chars_arg.split(",") if c.strip()]
    return [k for k, v in script["characters"].items()
            if not v.get("face_ref") or not v.get("body_ref")]


def main():
    ap = argparse.ArgumentParser(description="角色定妆图生成 (P2)")
    ap.add_argument("--ep", required=True, help="集数, 读 output/ep{ep}/script.json")
    ap.add_argument("--chars", default=None, help="指定角色 (逗号分隔; 默认缺 ref 的)")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help=f"checkpoint (默认 {DEFAULT_CKPT})")
    ap.add_argument("--seeds", type=int, default=1,
                    help="每角色生成 seed 数 (多张供手动选; 默认 1)")
    ap.add_argument("--dry-run", action="store_true", help="只打印 prompt 不生成")
    ap.add_argument("--out", default=None,
                    help="资产目录 (默认 assets/characters/{KEY})")
    args = ap.parse_args()

    ep = normalize_ep(args.ep)
    script_path = os.path.join(MANJU, "output", f"ep{ep}", "script.json")
    if not os.path.exists(script_path):
        print(f"[ERR] 缺剧本: {script_path} (先跑 novel_to_script / gen_script)")
        return 1
    script = load_json_script(script_path)
    chars = _pending_chars(script, args.chars)
    if not chars:
        print("[skip] 所有角色已有 face_ref+body_ref, 无需生成")
        return 0

    client = ComfyUIClient()
    changed = False
    for ck in chars:
        ch = script["characters"].get(ck)
        if not ch:
            print(f"[ERR] 角色 {ck} 不在剧本")
            continue
        fp = ch.get("fingerprint", "")
        face_p, body_p, neg = build_prompts(fp)
        print(f"\n=== 角色 {ck} ===", flush=True)
        print(f"  face: {face_p[:90]}...")
        print(f"  body: {body_p[:90]}...")
        if args.dry_run:
            continue

        char_dir = args.out or os.path.join(MANJU, "assets", "characters", ck)
        os.makedirs(char_dir, exist_ok=True)
        # 1) 正面肖像 → crop 512×512 face_ref
        best_face, best_body = None, None
        for i in range(args.seeds):
            seed = 42 + i * 101
            face_src = generate(client, face_p, neg, args.ckpt, FACE_W, FACE_H,
                                seed, f"char_{ck}_face", char_dir)
            if face_src:
                best_face = face_src
            body_src = generate(client, body_p, neg, args.ckpt, BODY_W, BODY_H,
                                seed, f"char_{ck}_body", char_dir)
            if body_src:
                best_body = body_src
        if not best_face or not best_body:
            print(f"[ERR] {ck} 生成失败 (ComfyUI 起着? {args.ckpt} 在?)")
            continue
        face_ref = os.path.join(char_dir, f"{ck}_face.png")
        if not os.path.exists(face_ref):
            crop_face(best_face, face_ref)   # 512×512
        body_ref = best_body
        print(f"  face_ref: {face_ref}  body_ref: {body_ref}", flush=True)
        ch["face_ref"] = face_ref
        ch["body_ref"] = body_ref
        changed = True

    if changed:
        dump_json(script, script_path)
        print(f"\n已回填 script.json: {script_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
