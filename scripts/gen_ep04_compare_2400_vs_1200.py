"""ep04 2400 vs 1200 side-by-side contact sheet (2026-07-24).

跑 mode C 对比 (LoRA + dual IPAdapter,治本路径) 用两个 ckpt:
  - 1200 step: ../ComfyUI/models/loras/{char}_lora.safetensors
  - 2400 step (continue): ../ComfyUI/models/loras/{char}_lora_continue-step00002400.safetensors

产物:
  output/ep04_real/{character}_2400_vs_1200/{1200,2400}/_ablation_C/shot{1..5}.png
  output/ep04_real/{character}_2400_vs_1200/compare/contact_sheet.png     5 rows x 3 cols (A baseline + C1200 + C2400)
  output/ep04_real/{character}_2400_vs_1200/compare/manifest.md

用法 (需 ComfyUI 8188 已起, refs 已上传):
  python scripts/gen_ep04_compare_2400_vs_1200.py --character linxia
  python scripts/gen_ep04_compare_2400_vs_1200.py --character xiaoya
"""
import argparse
import os
import sys

MANJU = "../manju"
sys.path.insert(0, MANJU)
sys.path.insert(0, os.path.join(MANJU, "scripts"))
from lib.comfyui_client import ComfyUIClient, build_face_shot_workflow  # noqa: E402
from scripts.gen_ep04_lora_ablation import (  # noqa: E402
    build_positive, REFS_BY_NAME, REAL_NEG, KW, KH, MODES,
)
from lib.ep04_shots import SHOTS_BY_NAME  # noqa: E402


def run_mode_c(client, character, shots, lora_name, lora_path_full, out_dir):
    """跑 5 镜 mode C 用指定 LoRA 路径."""
    os.makedirs(out_dir, exist_ok=True)
    refs = REFS_BY_NAME[character]
    from lib.comfyui_client import build_face_body_shot_workflow
    for s in shots:
        pos = build_positive(character, s, with_lora=True)
        wf = build_face_body_shot_workflow(
            ref_face_name=refs["face"],
            ref_body_name=refs["body"],
            positive=pos, negative=REAL_NEG,
            width=KW, height=KH, seed=42,
            weight_face=0.95, weight_body=0.70,
            filename_prefix=f"ep04_{character}_C_{lora_name}_shot{s:02d}",
        )
        lora_node = {
            "class_type": "LoraLoader",
            "inputs": {"model": ["ckpt", 0], "clip": ["ckpt", 1], "lora_name": os.path.basename(lora_path_full),
                       "strength_model": 0.85, "strength_clip": 0.85},
        }
        wf["lora"] = lora_node
        wf["unified"]["inputs"]["model"] = ["lora", 0]
        wf["pos"]["inputs"]["clip"] = ["lora", 1]
        wf["neg"]["inputs"]["clip"] = ["lora", 1]
        try:
            saved = client.generate(wf, out_dir)
            for f in saved:
                new = os.path.join(out_dir, f"shot{s:02d}.png")
                if os.path.exists(f) and not os.path.exists(new):
                    os.rename(f, new)
            print(f"[{character} C {lora_name} shot{s:02d}] OK -> {[os.path.basename(x) for x in saved]}", flush=True)
        except Exception as e:
            print(f"[{character} C {lora_name} shot{s:02d}] FAIL: {e}", flush=True)


def make_compare_contact(out_root):
    """5 行 x 3 列 (A baseline + C1200 + C2400) 一张大对比图."""
    import cv2, numpy as np
    cell_w, cell_h = 480, 832  # 原图
    pad = 8
    head_h = 60
    label_w = 120
    col_labels = ["A baseline", "C 1200-step", "C 2400-step"]
    cols_paths = {
        "A baseline": os.path.join(out_root, "A", "_ablation_C"),
        "C 1200-step": os.path.join(out_root, "1200", "_ablation_C"),
        "C 2400-step": os.path.join(out_root, "2400", "_ablation_C"),
    }
    shots = list(range(1, 6))
    n_rows = len(shots)
    n_cols = len(col_labels)
    sheet_w = n_cols * cell_w + (n_cols + 1) * pad
    sheet_h = n_rows * cell_h + (n_rows + 1) * pad
    sheet = np.full((sheet_h + head_h, sheet_w + label_w, 3), 30, dtype=np.uint8)
    # 列标签
    for j, lbl in enumerate(col_labels):
        x = label_w + pad + j * (cell_w + pad) + cell_w // 2 - 120
        cv2.putText(sheet, lbl, (x, head_h // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2, cv2.LINE_AA)
    # 行
    for i, s in enumerate(shots):
        y0 = head_h + pad + i * (cell_h + pad)
        cv2.putText(sheet, f"shot{s}", (10, y0 + cell_h // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (200, 200, 200), 2, cv2.LINE_AA)
        for j, lbl in enumerate(col_labels):
            x0 = label_w + pad + j * (cell_w + pad)
            path = os.path.join(cols_paths[lbl], f"shot{s:02d}.png")
            if os.path.exists(path):
                img = cv2.imread(path)
                if img is not None:
                    img = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
                    sheet[y0:y0 + cell_h, x0:x0 + cell_w] = img
            else:
                cv2.rectangle(sheet, (x0, y0), (x0 + cell_w, y0 + cell_h), (0, 0, 80), 2)
                cv2.putText(sheet, "MISSING", (x0 + 180, y0 + cell_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 255), 3, cv2.LINE_AA)
    os.makedirs(os.path.join(out_root, "compare"), exist_ok=True)
    out = os.path.join(out_root, "compare", "contact_sheet.png")
    cv2.imwrite(out, sheet)
    return out, sheet.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", choices=["xiaoya", "linxia"], required=True)
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--lora-1200", default=None, help="1200 step LoRA safetensors 文件名 (相对 ComfyUI/models/loras/)")
    ap.add_argument("--lora-2400", default="linxia_lora_continue-step00002400.safetensors",
                    help="2400 step continue LoRA safetensors 文件名")
    ap.add_argument("--skip-run-A", action="store_true", help="跳过 mode A (baseline) 直接用 1200 已有 A")
    args = ap.parse_args()

    if args.lora_1200 is None:
        args.lora_1200 = f"{args.character}_lora.safetensors"

    out_root = args.out_root or f"../manju/output/ep04_real/{args.character}_2400_vs_1200"
    os.makedirs(out_root, exist_ok=True)

    client = ComfyUIClient(args.server)
    shots = list(range(1, 6))

    print(f"\n=== A baseline (no LoRA) ===")
    if not args.skip_run_A:
        # mode A from REFS_BY_NAME[char] (no lora injection)
        from lib.comfyui_client import build_face_shot_workflow
        refs = REFS_BY_NAME[args.character]
        out_a = os.path.join(out_root, "A", "_ablation_C")
        os.makedirs(out_a, exist_ok=True)
        for s in shots:
            pos = build_positive(args.character, s, with_lora=False)
            wf = build_face_shot_workflow(
                ref_image_name=refs["face"],
                positive=pos, negative=REAL_NEG,
                width=KW, height=KH, seed=42,
                ipadapter_weight=0.95,
                filename_prefix=f"ep04_{args.character}_A_compare_shot{s:02d}",
            )
            try:
                saved = client.generate(wf, out_a)
                for f in saved:
                    new = os.path.join(out_a, f"shot{s:02d}.png")
                    if os.path.exists(f) and not os.path.exists(new):
                        os.rename(f, new)
                print(f"[A shot{s:02d}] OK", flush=True)
            except Exception as e:
                print(f"[A shot{s:02d}] FAIL: {e}", flush=True)

    print(f"\n=== C 1200-step ({args.lora_1200}) ===")
    run_mode_c(
        client, args.character, shots,
        lora_name="c1200",
        lora_path_full=args.lora_1200,
        out_dir=os.path.join(out_root, "1200", "_ablation_C"),
    )

    print(f"\n=== C 2400-step ({args.lora_2400}) ===")
    run_mode_c(
        client, args.character, shots,
        lora_name="c2400",
        lora_path_full=args.lora_2400,
        out_dir=os.path.join(out_root, "2400", "_ablation_C"),
    )

    print(f"\n=== compare contact sheet ===")
    out_png, shape = make_compare_contact(out_root)
    print(f"[OK] {out_png}  shape={shape}")


if __name__ == "__main__":
    sys.exit(main() or 0)
