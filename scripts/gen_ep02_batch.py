"""ep02 漫画版批量关键帧(10 镜)。SD1.5 Counterfeit-V3.0 + IPAdapter-Plus FACE 锁小雅漫画立绘脸。

漫画版先行(写实版回头完善)。底模 Counterfeit-V3.0(动漫专用, 替 DreamShaper 半写实 → 治"脸怪异",
2026-07-21 用户拍板 Counterfeit 候选3=seed2000)。复用 gen_ep01_batch 流程, 换 ep02 剧本(lib/ep02_shots) + 小雅立绘 REF。
IPAdapter weight 0.85(漫画版稍紧于 ep01 的 0.8, 锁女角脸); face_sim≥0.50 门; shot6 纯手部无主角脸 → skip 门。

用法:
    .venv/Scripts/python.exe scripts/gen_ep02_batch.py                  # 全 10 镜
    .venv/Scripts/python.exe scripts/gen_ep02_batch.py --shots 1,3,5    # 指定
    .venv/Scripts/python.exe scripts/gen_ep02_batch.py --shots 4 --seed 42  # 单镜重抽
"""
import argparse
import os
import sys

MANJU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MANJU)

from lib.comfyui_client import ComfyUIClient, build_face_shot_workflow  # noqa: E402
from lib.ep02_shots import CHAR, STYLE, SHOTS, NEG  # noqa: E402

REF = "../manju/assets/characters/小雅立绘.png"
OUT = "../manju/output/ep02"
# shot6 纯手部特写无主角脸, face_sim 门 skip(不测)
NO_FACE_SHOTS = {6}


def build_positive(shot, extra=""):
    return f"{CHAR}, {SHOTS[shot]}, {STYLE}" + (f", {extra}" if extra else "")


FIT_PY = "../fitness-video-pipeline/.venv/Scripts/python.exe"  # manju venv 无 insightface, 借 fitness venv


def _face_sim_pair(ref, shot_path):
    """(sim, n_ref, n_shot)。借 fitness venv insightface 跑(manju venv 缺); 子进程调 face_sim.py 取 cosine。"""
    import re
    import subprocess
    try:
        out = subprocess.check_output(
            [FIT_PY, "../manju/scripts/face_sim.py", ref, shot_path],
            cwd="../manju", text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None, 0, 0
    m = re.search(r"cosine_similarity\s*=\s*([\d.]+)", out)
    if not m:
        return None, 0, 0  # face_sim.py 检不出脸时无 cosine 行 → None
    return float(m.group(1)), 1, 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--weight", type=float, default=0.85)  # 漫画版稍紧锁女角脸(ep01 漫画用 0.8)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ckpt", default="Counterfeit-V3.0_fix_fp16.safetensors")  # 漫画专用底模(替 DreamShaper 半写实)
    args = ap.parse_args()

    assert os.path.exists(REF), f"缺小雅立绘: {REF} (先跑 gen_xiaoya_manga.py + det_score 挑脸)"
    shots = [int(x) for x in args.shots.split(",") if x.strip()]
    os.makedirs(args.out, exist_ok=True)

    c = ComfyUIClient(args.server)
    ref_name = c.upload_image(REF)
    print(f"[upload] ref(小雅立绘) -> {ref_name}\n")

    results = []
    for s in shots:
        pos = build_positive(s)
        wf = build_face_shot_workflow(
            ref_name, pos, NEG,
            ckpt=args.ckpt, seed=args.seed, ipadapter_weight=args.weight,
            width=512, height=912, filename_prefix=f"ep02_shot{s:02d}",
        )
        print(f"=== shot {s:2d} ===\n  {SHOTS[s]}", flush=True)
        outs = c.generate(wf, args.out)
        path = outs[0]
        if s in NO_FACE_SHOTS:
            print(f"  -> {os.path.basename(path)}  [skip face_sim 无主角脸镜]\n", flush=True)
            results.append((s, path, None, "[no-face-skip]"))
            continue
        sim, na, nb = _face_sim_pair(REF, path)
        if sim is None:
            flag, simstr = "[no-face]", "N/A"
        elif sim >= 0.5:
            flag, simstr = "[OK]", f"{sim:.3f}"
        else:
            flag, simstr = "[FAIL?]", f"{sim:.3f}"
        print(f"  -> {os.path.basename(path)}  face_sim={simstr}  {flag}  (ref={na} shot={nb})\n", flush=True)
        results.append((s, path, sim, flag))

    print("===== 汇总 =====")
    fails = 0
    for s, p, sim, flag in results:
        simstr = f"{sim:.3f}" if sim is not None else "N/A"
        print(f"  shot {s:2d}  sim={simstr}  {flag}  {os.path.basename(p)}")
        if flag not in ("[OK]", "[no-face-skip]"):
            fails += 1
    print(f"\n[batch done] {len(results)} 镜, {fails} 镜待复核(重抽: --shots <号> --seed <新>)")


if __name__ == "__main__":
    main()
