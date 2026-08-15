#!/usr/bin/env python3
"""scripts/gen_scene_assets.py — 场景参考图生成 (P1-2, 2026-08-16).

从剧本 scenes (或手动 --scene) → SDXL 生成**空景参考图** (无角色), 供 kf 背景锁.
复用 gen_character_assets.py 模式 (SDXL + ComfyUIClient).

用法:
  python scripts/gen_scene_assets.py --ep 21                        # 从剧本 scenes 生成
  python scripts/gen_scene_assets.py --ep 21 --scenes "健身房,卧室"  # 手动指定
  python scripts/gen_scene_assets.py --ep 21 --dry-run              # 只打印 prompt
产物: assets/scenes/{scene_key}.png
"""
import argparse
import glob
import os
import re
import sys

MANJU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MANJU)

from lib.comfyui_client import ComfyUIClient, build_txt2img_workflow  # noqa: E402
from lib.storyboard import load_json_script, normalize_ep  # noqa: E402

SDXL = "sd_xl_base_1.0.safetensors"
NEG = ("distorted, deformed, blurry, low quality, watermark, text, cartoon, people, person, "
       "human, character, face, 3d render")
# 空景后缀: 无人物, 广角, 一致室内
SCENE_SUFFIX = ("empty room, no people, no person, wide angle establishing shot, "
                "clean and consistent interior, photorealistic, bright even lighting")

SCENE_W, SCENE_H = 1024, 1024   # SDXL 方形空景


def build_scene_prompt(scene_desc: str):
    """场景描述 → prompt (空景, 无人物)."""
    desc = (scene_desc or "").strip()
    return f"{desc}, {SCENE_SUFFIX}" if desc else SCENE_SUFFIX


def _scene_key(desc: str) -> str:
    """场景描述 → 文件 key (取前 2 个 CJK/英文词)."""
    desc = re.sub(r"[^\w一-鿿]+", " ", desc or "").strip()
    words = [w for w in desc.split() if w][:3]
    return "_".join(words) or "scene"


def _scenes_from_script(script) -> list[str]:
    """剧本 scenes 字段; 缺则从各镜 shot 描述提取场景短语."""
    scenes = script.get("scenes") or []
    if scenes:
        return [str(s) for s in scenes if str(s).strip()]
    # fallback: 从 shot 描述提取干净场景短语 (interior/background/room 后即停, 防抓进角色)
    pat = re.compile(r"([a-z][\w\s,.-]{0,30}?(?:interior|background|room))", re.I)
    found = []
    for v in script.get("shots", {}).values():
        m = pat.search(v.get("shot", ""))
        if m and m.group(1).strip() not in found:
            found.append(m.group(1).strip())
    return found or ["modern gym interior"]


def main():
    ap = argparse.ArgumentParser(description="场景参考图生成 (P1-2)")
    ap.add_argument("--ep", default=None, help="集数 (读 output/ep{ep}/script.json 的 scenes)")
    ap.add_argument("--scenes", default=None, help="手动指定场景 (逗号分隔; 优先于剧本)")
    ap.add_argument("--dry-run", action="store_true", help="只打印 prompt")
    ap.add_argument("--out", default=None, help="输出目录 (默认 assets/scenes)")
    args = ap.parse_args()

    if args.scenes:
        scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    elif args.ep:
        ep = normalize_ep(args.ep)
        script_path = os.path.join(MANJU, "output", f"ep{ep}", "script.json")
        if not os.path.exists(script_path):
            print(f"[ERR] 缺剧本: {script_path}")
            return 1
        scenes = _scenes_from_script(load_json_script(script_path))
    else:
        print("[ERR] 需 --ep 或 --scenes")
        return 1

    if not scenes:
        print("[ERR] 无场景可生成")
        return 1

    out_dir = args.out or os.path.join(MANJU, "assets", "scenes")
    os.makedirs(out_dir, exist_ok=True)
    client = ComfyUIClient()
    print(f"[scene] {len(scenes)} 个场景 -> {out_dir}\n", flush=True)
    for i, desc in enumerate(scenes):
        key = _scene_key(desc)
        prompt = build_scene_prompt(desc)
        print(f"=== 场景{i + 1}: {desc} ===", flush=True)
        print(f"  prompt: {prompt[:90]}...", flush=True)
        if args.dry_run:
            continue
        wf = build_txt2img_workflow(prompt, NEG, ckpt=SDXL, width=SCENE_W, height=SCENE_H,
                                    seed=42 + i * 7, steps=25, cfg=7.0,
                                    filename_prefix=f"scene_{key}")
        saved = client.generate(wf, out_dir)
        if saved:
            src = max(saved, key=os.path.getmtime)
            dst = os.path.join(out_dir, f"{key}.png")
            if not os.path.exists(dst) and os.path.abspath(src) != os.path.abspath(dst):
                os.replace(src, dst)
            print(f"  -> {dst}", flush=True)
        else:
            print(f"  [ERR] 生成失败 (ComfyUI 起着?)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
