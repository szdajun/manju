"""ep01 批量生图(命门已过、多姿态一致性坐实后的量产)。

逐镜生 → 立刻 face_sim 抽验(防脸漂)→ 汇总打印。buffalo_l 只加载一次(批量省时)。
脸漂(sim<0.5)或漏检自动标 [FAIL?]/[no-face],事后单独重抽该镜即可。

用法:
    python scripts/gen_ep01_batch.py                          # 默认补生剩余 8 镜
    python scripts/gen_ep01_batch.py --shots 1,2,3            # 指定
    python scripts/gen_ep01_batch.py --shots 1,2 --seed 42    # 固定 seed 抽卡
"""
import argparse
import os
import sys

MANJU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MANJU)
sys.path.insert(0, os.path.join(MANJU, "scripts"))

from lib.comfyui_client import ComfyUIClient, build_face_shot_workflow  # noqa: E402
import gen_shot as gs  # noqa: E402  复用 CHAR/STYLE/SHOTS/build_positive/NEG_DEFAULT

REF = gs.REF_DEFAULT


def _face_sim_pair(ref, shot_path):
    """返回 (sim, n_faces_ref, n_faces_shot)。buffalo_l 缓存到函数属性, 整批只加载一次。"""
    import cv2
    import numpy as np
    from insightface.app import FaceAnalysis
    if not hasattr(_face_sim_pair, "_app"):
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        _face_sim_pair._app = app
    app = _face_sim_pair._app

    def rd(p):
        return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)

    def primary(p):
        fs = app.get(rd(p))
        if not fs:
            return None, 0
        fs.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                reverse=True)
        e = fs[0].embedding
        e = e / (np.linalg.norm(e) + 1e-9)
        return e, len(fs)

    ea, na = primary(ref)
    eb, nb = primary(shot_path)
    if ea is None or eb is None:
        return None, na, nb
    return float(np.dot(ea, eb)), na, nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="1,2,3,5,6,8,9,10",
                    help="逗号分隔镜号(默认补生剩余 8 镜; 4/7 已在 ep01_test 跑过)")
    ap.add_argument("--out", default="../manju/output/ep01")
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--weight", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    shots = [int(x) for x in args.shots.split(",") if x.strip()]
    os.makedirs(args.out, exist_ok=True)

    c = ComfyUIClient(args.server)
    ref_name = c.upload_image(REF)
    print(f"[upload] ref -> {ref_name}\n")

    results = []
    for s in shots:
        pos = gs.build_positive(s)
        wf = build_face_shot_workflow(
            ref_name, pos, gs.NEG_DEFAULT,
            seed=args.seed, ipadapter_weight=args.weight,
            width=512, height=912, filename_prefix=f"ep01_shot{s:02d}",
        )
        print(f"=== shot {s:2d} ===\n  {gs.SHOTS[s]}")
        outs = c.generate(wf, args.out)
        path = outs[0]
        sim, na, nb = _face_sim_pair(REF, path)
        if sim is None:
            flag = "[no-face]"
            simstr = "N/A"
        elif sim >= 0.5:
            flag = "[OK]"
            simstr = f"{sim:.3f}"
        else:
            flag = "[FAIL?]"
            simstr = f"{sim:.3f}"
        print(f"  -> {os.path.basename(path)}  face_sim={simstr}  {flag}  "
              f"(ref_faces={na} shot_faces={nb})\n")
        results.append((s, path, sim, flag))

    print("===== 汇总 =====")
    fails = 0
    for s, p, sim, flag in results:
        simstr = f"{sim:.3f}" if sim is not None else "N/A"
        print(f"  shot {s:2d}  sim={simstr}  {flag}  {os.path.basename(p)}")
        if flag != "[OK]":
            fails += 1
    print(f"\n[batch done] {len(results)} 镜, {fails} 镜待复核(重抽: --shots <号> --seed <新>)")


if __name__ == "__main__":
    main()
