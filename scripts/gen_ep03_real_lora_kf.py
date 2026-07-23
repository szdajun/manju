"""ep03 写实版 LoRA 关键帧 (5 镜)。

复用 gen_ep03_real_batch 双 IPAdapter 链 (face 0.95 + body 0.70) + 注入 LoRA 节点.
默认用 linxia_lora.safetensors (1200 step, loss 0.049).

用法:
  python scripts/gen_ep03_real_lora_kf.py                # 全 5 镜, 默认 linxia_lora
  python scripts/gen_ep03_real_lora_kf.py --shots 1,3,5  # 指定
  python scripts/gen_ep03_real_lora_kf.py --no-lora     # baseline 对照 (无 LoRA, 双 IPA)

产出落到 output/ep03_real_lora1200/ 或 --lora 后缀自动.
"""
import argparse
import os
import re
import subprocess
import sys

MANJU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MANJU)

from lib.comfyui_client import (  # noqa: E402
    ComfyUIClient, build_face_body_shot_workflow)
from lib.ep03_shots import CHAR, SHOTS, I2V_MOTION, REAL_STYLE, REAL_NEG  # noqa: E402

REF_FACE = "../manju/output/ep03_lihui/linxia_real_face.png"
REF_BODY = "../manju/output/ep03_lihui/linxia_real.png"
FIT_PY = "../fitness-video-pipeline/.venv/Scripts/python.exe"
DEFAULT_CKPT = "DreamShaper_8_pruned.safetensors"
KW, KH = 480, 832


def build_positive(shot):
    return f"{CHAR}, {SHOTS[shot]}, {REAL_STYLE}, {I2V_MOTION[shot]}"


def inject_lora(wf, lora_filename, strength=0.85):
    """Load LoRA 节点串在 ckpt -> unified 之间, 替换 unified.model 与 CLIP text encode clip."""
    lora_node = {
        "class_type": "LoraLoader",
        "inputs": {"model": ["ckpt", 0], "clip": ["ckpt", 1], "lora_name": lora_filename,
                   "strength_model": strength, "strength_clip": strength},
    }
    wf["lora"] = lora_node
    wf["unified"]["inputs"]["model"] = ["lora", 0]
    wf["pos"]["inputs"]["clip"] = ["lora", 1]
    wf["neg"]["inputs"]["clip"] = ["lora", 1]
    return wf


def _face_sim_pair(ref, shot_path):
    try:
        out = subprocess.check_output(
            [FIT_PY, "../manju/scripts/face_sim.py", ref, shot_path],
            cwd="../manju", text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    m = re.search(r"cosine_similarity\s*=\s*([\d.]+)", out)
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="1,2,3,4,5")
    ap.add_argument("--out-suffix", default="lora1200", help="输出目录后缀 (空=默认), 默认 lora1200")
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--weight-face", type=float, default=0.95)
    ap.add_argument("--weight-body", type=float, default=0.70)
    ap.add_argument("--seed", type=int, default=42, help="seed 锁 (与原 ep03_real 同 seed 公平对照)")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--lora", default="linxia_lora.safetensors",
                    help="LoRA safetensors 文件名 (相对 ComfyUI/models/loras/)")
    ap.add_argument("--no-lora", action="store_true", help="跳过 LoRA 注入 (baseline 对照)")
    ap.add_argument("--lora-strength", type=float, default=0.85)
    args = ap.parse_args()

    out = f"../manju/output/ep03_real_{args.out_suffix}" if args.out_suffix else "../manju/output/ep03_real"
    os.makedirs(out, exist_ok=True)

    assert os.path.exists(REF_FACE), f"缺脸 ref: {REF_FACE}"
    assert os.path.exists(REF_BODY), f"缺全身 ref: {REF_BODY}"
    shots = [int(x) for x in args.shots.split(",") if x.strip()]

    c = ComfyUIClient(args.server)
    face_name = c.upload_image(REF_FACE)
    body_name = c.upload_image(REF_BODY)
    print(f"[upload] {face_name}, {body_name}")

    results = []
    for s in shots:
        pos = build_positive(s)
        wf = build_face_body_shot_workflow(
            ref_face_name=face_name,
            ref_body_name=body_name,
            positive=pos, negative=REAL_NEG,
            ckpt=args.ckpt, seed=args.seed,
            width=KW, height=KH,
            weight_face=args.weight_face, weight_body=args.weight_body,
            filename_prefix=f"ep03_real_{args.out_suffix}_shot{s:02d}",
        )
        if not args.no_lora:
            inject_lora(wf, args.lora, strength=args.lora_strength)
            tag = f"DUAL+LoRA({args.lora})"
        else:
            tag = "DUAL (no LoRA)"
        try:
            saved = c.generate(wf, out)
            for f in saved:
                ks = [v for v in wf.values() if v["class_type"] == "KSampler"][0]
                seed = ks["inputs"]["seed"]
                new = os.path.join(out, f"ep03_shot{s:02d}_seed{seed}.png")
                if os.path.exists(f) and not os.path.exists(new):
                    os.rename(f, new)
                    f = new
                sim = _face_sim_pair(REF_FACE, f)
                results.append((s, tag, seed, f, sim))
                print(f"[shot {s} | {tag} | seed {seed}] {os.path.basename(f)} face_sim={sim}", flush=True)
        except Exception as e:
            print(f"[shot {s}] FAIL: {e}", flush=True)
            results.append((s, tag, None, None, None))

    n_ok = sum(1 for r in results if r[3])
    sim_avg = sum(r[4] for r in results if r[4] is not None) / max(1, sum(1 for r in results if r[4] is not None))
    print(f"\n=== 汇总 ===")
    print(f"  出图: {n_ok}/{len(results)}")
    print(f"  face_sim 均值: {sim_avg:.3f}")
    for s, tag, seed, f, sim in results:
        if f:
            print(f"  shot{s:02d} seed={seed} sim={sim:.3f}" if sim is not None else f"  shot{s:02d} seed={seed} sim=-")
    print(f"\n[next] vision 对比 ep03_real/ 现有 5 镜 (seed 锁 {args.seed} 公平对照)")
    print(f"  产物: {out}/")


if __name__ == "__main__":
    sys.exit(main() or 0)
