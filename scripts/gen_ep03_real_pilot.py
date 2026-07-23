"""ep03 双 IPAdapter pilot — A/B 锁衣服实验 (2026-07-23)。

跑 shot3 × 3 seed × 2 模式 = 6 张关键帧, vision 对比 face-only vs face-body。
唯一变量: 是否加 body ref weight 0.70。
输出:
  output/ep03_real/_pilot/face_only/seed_NNNN.png (3 张)
  output/ep03_real/_pilot/face_body/seed_NNNN.png (3 张)
  output/ep03_real/_pilot/_ablation_face_vs_body.png (拼图)

依赖:
  - ComfyUI Win8188 已起 (本脚本前置条件)
  - 立绘 face/body ref 已生成并放到 ComfyUI input 目录 (默认 upload 目录):
      linxia_real_face.png + linxia_real.png
  - 跑本脚本前需先 (或脚本自动) 把 ref 上传到 ComfyUI 的 ComfyUI/input/

用法:
  .venv/Scripts/python.exe scripts/gen_ep03_real_pilot.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.comfyui_client import (  # noqa: E402
    ComfyUIClient,
    build_face_shot_workflow,           # baseline (face only)
    build_face_body_shot_workflow,      # 新 builder (face + body)
)
from lib.ep03_shots import SHOTS, REAL_STYLE, REAL_NEG  # noqa: E402

SERVER = "http://127.0.0.1:8188"
OUT = "../manju/output/ep03_real/_pilot"
FACE_REF = "linxia_real_face.png"  # 假定已上传到 ComfyUI/input
BODY_REF = "linxia_real.png"       # 假定已上传到 ComfyUI/input
SHOT_ID = 3                         # pilot 用 shot3 (服装最完整可见 + Plan agent 拍板)
SEEDS = [1001, 1002, 1003]          # 3 个固定 seed, A/B 唯一变量
KW, KH = 480, 832                   # Wan i2v 原生桶


def _build_prompt():
    """构 shot3 的 positive (SHOTS[3] + REAL_STYLE)。"""
    return f"{SHOTS[SHOT_ID]}, {REAL_STYLE}"


def _upload_refs(client: ComfyUIClient, refs: list[str]):
    """把 ref 上传到 ComfyUI input 目录 (若还没)。"""
    # ComfyUIClient 假设 ref 已存在, 不内置 upload。本函数给上层 fallback 提示。
    for r in refs:
        print(f"[ref] 假定已上传 {r} 到 ComfyUI input 目录 (upload 步骤手工或独立脚本)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=SERVER)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--shot", type=int, default=SHOT_ID)
    ap.add_argument("--seeds", default="1001,1002,1003")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    out_face_only = os.path.join(args.out, "face_only")
    out_face_body = os.path.join(args.out, "face_body")
    os.makedirs(out_face_only, exist_ok=True)
    os.makedirs(out_face_body, exist_ok=True)

    c = ComfyUIClient(args.server)
    _upload_refs(c, [FACE_REF, BODY_REF])

    pos = _build_prompt()
    neg = REAL_NEG

    # ---------- A: face-only baseline (ep02 范式, weight=0.95) ----------
    print(f"\n=== A: face-only (baseline) — shot{args.shot} × {len(seeds)} seeds ===")
    a_files = []
    for seed in seeds:
        wf = build_face_shot_workflow(
            ref_image_name=FACE_REF,
            positive=pos, negative=neg,
            width=KW, height=KH, seed=seed,
            steps=28, cfg=6.5, ipadapter_weight=0.95,
            filename_prefix=f"ep03_faceonly_shot{args.shot}",
        )
        try:
            saved = c.generate(wf, out_face_only)
            for f in saved:
                # 改名前缀: seed_NNNN.png 方便 contact sheet
                new = os.path.join(out_face_only, f"seed_{seed}.png")
                if os.path.exists(f) and not os.path.exists(new):
                    os.rename(f, new)
                    f = new
                a_files.append(f)
            print(f"[seed {seed}] OK -> {[os.path.basename(x) for x in saved]}", flush=True)
        except Exception as e:
            print(f"[seed {seed}] FAIL: {e}", flush=True)

    # ---------- B: face + body (新 builder, face 0.95 + body 0.70) ----------
    print(f"\n=== B: face+body (新 builder) — shot{args.shot} × {len(seeds)} seeds ===")
    b_files = []
    for seed in seeds:
        wf = build_face_body_shot_workflow(
            ref_face_name=FACE_REF,
            ref_body_name=BODY_REF,
            positive=pos, negative=neg,
            width=KW, height=KH, seed=seed,
            steps=28, cfg=6.5,
            weight_face=0.95, weight_body=0.70,
            filename_prefix=f"ep03_facebody_shot{args.shot}",
        )
        try:
            saved = c.generate(wf, out_face_body)
            for f in saved:
                new = os.path.join(out_face_body, f"seed_{seed}.png")
                if os.path.exists(f) and not os.path.exists(new):
                    os.rename(f, new)
                    f = new
                b_files.append(f)
            print(f"[seed {seed}] OK -> {[os.path.basename(x) for x in saved]}", flush=True)
        except Exception as e:
            print(f"[seed {seed}] FAIL: {e}", flush=True)

    # ---------- 输出汇总 ----------
    print(f"\n=== pilot 汇总 ===")
    print(f"  face-only: {len(a_files)} 张 -> {out_face_only}")
    print(f"  face-body: {len(b_files)} 张 -> {out_face_body}")
    print(f"\n[next] vision 抽帧对比:")
    print(f"  - 看每组 (A vs B 同 seed) 的脸是否像林夏")
    print(f"  - 看上衣是否仍雾蓝色 (vs 漂成白/蓝/黑)")
    print(f"  - 看左胸白色弧线标志位置/是否消失")
    print(f"  - 看裤子黑色高腰 + 灰侧线是否保留")
    print(f"  判定: ≥2/3 seeds 中 face+body 比 face-only 衣服更接近 canonical = 通过 Gate B")


if __name__ == "__main__":
    sys.exit(main() or 0)