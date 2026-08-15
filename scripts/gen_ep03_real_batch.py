"""ep03 写实版批量关键帧 (5 镜)。双 IPAdapter (face 0.95 + body 0.70) 锁脸+锁衣服。

ep03 单人 5 镜短剧 (林夏健身房日常), 沿用 ep02 写实路径:
  - SD1.5 DreamShaper_8 + IPAdapter-Plus FACE (face ref) + IPAdapter-Plus FACE (body ref, 试锁衣)
  - 关键帧尺寸 480×832 (Wan i2v 原生桶, 与 ep02 一致)
  - 5 镜 SHOTS/I2V_MOTION/SPEAKER 全部来自 lib/ep03_shots
  - 钉死服装指纹每镜: 雾蓝长袖+白弧标志 / 黑色高腰+灰侧线 / 白鞋 / 高马尾
  - I2V_MOTION 末尾必带"the misty-blue top stays misty-blue throughout"等 3 句不变式

face_sim 子进程借 fitness venv insightface (manju venv 缺, 沿用 ep02 套路)。

用法:
    .venv/Scripts/python.exe scripts/gen_ep03_real_batch.py                  # 全 5 镜
    .venv/Scripts/python.exe scripts/gen_ep03_real_batch.py --shots 1,3,5    # 指定
    .venv/Scripts/python.exe scripts/gen_ep03_real_batch.py --shots 3 --seed 42  # 单镜重抽
    .venv/Scripts/python.exe scripts/gen_ep03_real_batch.py --face-only      # debug: 跳过 body ref (回退 A 组)
"""
import argparse
import os
import sys

MANJU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MANJU)

from lib.comfyui_client import (  # noqa: E402
    ComfyUIClient, build_face_shot_workflow, build_face_body_shot_workflow)
from lib.storyboard import (  # noqa: E402
    character, load_script, shot, speaker)

DEFAULT_CKPT = "DreamShaper_8_pruned.safetensors"
KW, KH = 480, 832  # Wan i2v 原生桶, 沿用 ep02

FIT_PY = "../fitness-video-pipeline/.venv/Scripts/python.exe"


def build_positive(shot_id, script, extra=""):
    """构关键帧正向词: 说话人指纹 + 场景 + 画风 + 运动 (末尾不变式锁衣服).

    数据全部来自剧本 dict (Phase 3 通用化): fingerprint/shot/style/motion.
    motion 必带"the {color} top stays {color} throughout"等 3 句, 在 i2v 时锁衣服.
    """
    ch = character(script, shot_id)
    sh = shot(script, shot_id)
    return (f"{ch['fingerprint']}, {sh['shot']}, {script['style']}"
            + (f", {extra}" if extra else "") + f", {sh['motion']}")


def _face_sim_pair(ref, shot_path):
    """(sim,)。借 fitness venv insightface 子进程调 face_sim.py。None=检不出脸。"""
    import re
    import subprocess
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
    ap.add_argument("--ep", default="03", help="集数 (读剧本数据, 默认 03)")
    ap.add_argument("--shots", default=None, help="指定镜 (如 1,2,3; 默认剧本全部)")
    ap.add_argument("--out", default=None, help="关键帧输出目录 (默认 output/ep{ep}_real)")
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--weight-face", type=float, default=0.95,
                    help="脸 IPAdapter weight (沿用 ep02 单锁 0.95)")
    ap.add_argument("--weight-body", type=float, default=0.70,
                    help="全身 IPAdapter weight (调研推荐值, pilot 已验 3/3 长袖对)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--face-only", action="store_true",
                    help="强制全部走单锁 (debug: 对比 baseline, 跳过 body ref)")
    # orchestrator 接入: 角色立绘可从剧本数据透传 (剧本 characters 优先)
    ap.add_argument("--ref-face", default=None, help="脸 ref 图路径 (默认剧本 characters[主角].face_ref)")
    ap.add_argument("--ref-body", default=None, help="全身 ref 图路径 (默认剧本 characters[主角].body_ref)")
    args = ap.parse_args()

    # 读剧本数据 (自动探测 output/epXX/script.json / scripts/epXX.json / lib.epXX_shots)
    script = load_script(args.ep)
    ep = script["ep"]  # 剧本有 ep 字段用剧本
    out = args.out or os.path.join(MANJU, "output", f"ep{ep}_real")
    shots = [int(x) for x in (args.shots or ",".join(script["shots"].keys())).split(",")
             if x.strip()]
    first_spk = speaker(script, shots[0]) if shots else next(iter(script["characters"]))
    ref_face = args.ref_face or script["characters"][first_spk].get("face_ref") or ""
    ref_body = args.ref_body or script["characters"][first_spk].get("body_ref") or ""

    if not ref_face or not os.path.exists(ref_face):
        raise SystemExit(
            f"[ERR] 缺脸 ref: {ref_face!r}. 需剧本 characters[{first_spk}].face_ref 存在, "
            f"或传 --ref-face. 用 gen_script.py --char KEY=名:人设 重新生成剧本可自带 ref.")
    if not ref_body or not os.path.exists(ref_body):
        print(f"[WARN] 缺全身 ref ({ref_body!r}) → 降级 face-only 单锁", flush=True)
        args.face_only = True
    os.makedirs(out, exist_ok=True)

    c = ComfyUIClient(args.server)
    face_name = c.upload_image(ref_face)
    print(f"[upload] ref face -> {face_name}")
    body_name = c.upload_image(ref_body) if not args.face_only else None
    if body_name:
        print(f"[upload] ref body -> {body_name}")
    print(f"[cfg] {args.ckpt}  face w={args.weight_face}  body w={args.weight_body if body_name else 'N/A'}  "
          f"face-only={args.face_only}  关键帧 {KW}x{KH}  镜数 {shots}\n")

    results = []
    prefix = f"ep{ep}_shot"
    for s in shots:
        # ⭐ 2026-08-15: 按镜说话人取 ref (双角色支持) — 说话人有 face_ref 用它的,
        # 否则回退全局 (--ref-face/--ref-body 或 first_spk).
        spk = speaker(script, s)
        shot_face = args.ref_face or (script["characters"].get(spk, {}).get("face_ref") or "")
        shot_body = args.ref_body or (script["characters"].get(spk, {}).get("body_ref") or "")
        if shot_face and os.path.exists(shot_face) and shot_face != ref_face:
            ref_face = shot_face
            face_name = c.upload_image(ref_face)
            print(f"[upload] shot{s} 说话人 {spk} face -> {face_name}", flush=True)
        if shot_body and os.path.exists(shot_body) and shot_body != ref_body:
            ref_body = shot_body
            body_name = c.upload_image(ref_body)
            print(f"[upload] shot{s} 说话人 {spk} body -> {body_name}", flush=True)

        pos = build_positive(s, script)
        if body_name:
            wf = build_face_body_shot_workflow(
                ref_face_name=face_name,
                ref_body_name=body_name,
                positive=pos, negative=script["neg"],
                ckpt=args.ckpt, seed=args.seed,
                width=KW, height=KH,
                weight_face=args.weight_face, weight_body=args.weight_body,
                filename_prefix=f"{prefix}{s:02d}",
            )
            mode = "DUAL face+body"
        else:
            wf = build_face_shot_workflow(
                ref_image_name=face_name,
                positive=pos, negative=script["neg"],
                ckpt=args.ckpt, seed=args.seed,
                width=KW, height=KH,
                ipadapter_weight=args.weight_face,
                filename_prefix=f"{prefix}{s:02d}",
            )
            mode = "FACE-only"
        try:
            saved = c.generate(wf, out)
            # 改名: seed_NNNN.png 方便 contact sheet (跟 pilot 脚本同款)
            for f in saved:
                # 取 seed: 读 ksampler seed
                ks = [v for v in wf.values() if v["class_type"] == "KSampler"][0]
                seed = ks["inputs"]["seed"]
                new = os.path.join(out, f"{prefix}{s:02d}_seed{seed}.png")
                if os.path.exists(f) and not os.path.exists(new):
                    os.rename(f, new)
                    f = new
                # face_sim (用脸 ref, 同 ep02 沿用)
                sim = _face_sim_pair(ref_face, f)
                results.append((s, mode, seed, f, sim))
                print(f"[shot {s} | {mode} | seed {seed}] -> {os.path.basename(f)}  face_sim={sim}", flush=True)
        except Exception as e:
            print(f"[shot {s}] FAIL: {e}", flush=True)
            results.append((s, mode, None, None, None))

    # 汇总
    print(f"\n=== 汇总 ===")
    n_ok = sum(1 for r in results if r[3])
    n_face_ok = sum(1 for r in results if r[4] is not None and r[4] >= 0.50)
    print(f"  出图: {n_ok}/{len(results)}")
    print(f"  face_sim ≥ 0.50: {n_face_ok}/{n_ok}")
    for s, mode, seed, f, sim in results:
        if f:
            print(f"  shot{s:02d} [{mode}] seed={seed} sim={sim}  {os.path.basename(f)}")
    print(f"\n[next] vision 抽帧 + 服装属性清单逐镜打勾 (8 项/镜):")
    print(f"  1-2-3-4-5-6-7-8 项/镜: 上衣雾蓝/长袖/领口/白弧标志位置/黑高腰/灰侧线/白鞋/无伪文字")


if __name__ == "__main__":
    sys.exit(main() or 0)