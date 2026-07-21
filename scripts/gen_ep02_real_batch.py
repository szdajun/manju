"""ep02 写实版批量关键帧(10 镜)。SD1.5 DreamShaper + IPAdapter-Plus FACE 锁写实小雅立绘脸。

写实化(相对 gen_ep02_batch 漫画版):
  - REF: 漫画小雅立绘(assets/characters/小雅立绘.png) → 写实小雅立绘写实.png(#1, det0.89)
  - STYLE/NEG: manga → photoreal(ep02_shots.REAL_STYLE/REAL_NEG, 反逼漫画, 复用 ep01_real 套路)
  - ckpt: Counterfeit-V3.0(动漫) → DreamShaper_8(写实底模)
  - weight: 0.85 → 0.95(ep01 写实教训#1: 写实 IPAdapter 脸漂大于漫画, 收紧锁脸)
  - **关键帧尺寸: 512×912 → 480×832**(Wan2.2 i2v 原生竖屏桶, ÷16=30/52 均偶;
    512×912 的 912÷16=57 奇 → Wan transformer addcmul off-by-one model.py:993 崩;
    且 image dims==gen dims, 关键帧直接喂 Wan 不 resize 损质)

剧本/分镜(SHOTS 场景中性 + CHAR 小雅指纹)复用 lib/ep02_shots — 写实版只改画风不改剧本。
shot6 纯手部无主角脸 → face_sim 门 skip(不测)。产物 output/ep02_real/ep02_shotNN_*.png → gen_wan(ep02) Wan i2v。

face_sim 子进程借 fitness venv insightface(manju venv 缺), 复用 gen_ep02_batch 同款。

用法:
    .venv/Scripts/python.exe scripts/gen_ep02_real_batch.py                 # 全 10 镜
    .venv/Scripts/python.exe scripts/gen_ep02_real_batch.py --shots 1,3,5   # 指定
    .venv/Scripts/python.exe scripts/gen_ep02_real_batch.py --shots 4 --seed 42  # 单镜重抽
"""
import argparse
import os
import sys

MANJU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MANJU)

from lib.comfyui_client import ComfyUIClient, build_face_shot_workflow  # noqa: E402
from lib.ep02_shots import CHAR, SHOTS, REAL_STYLE, REAL_NEG  # noqa: E402

# 脸 crop 版(治 IPAdapter center-crop 竖图丢脸 bug, 2026-07-21 小雅实案):
# 小雅立绘脸在图上部 17%, IPAdapter center-crop 切掉脸主体 → 锁身体不锁脸 → 关键帧 face_sim 0.23-0.38 全挂。
# scripts/crop_face.py 从 小雅立绘写实.png crop 纯脸成 512×512 方图, 喂 IPAdapter 不再丢脸。face_sim 金标准也用它(同一张脸)。
REF = "../manju/output/xiaoya_real/小雅立绘写实_face.png"
OUT = "../manju/output/ep02_real"
NO_FACE_SHOTS = {6}  # shot6 纯手部特写无主角脸, face_sim 门 skip

# 写实底模(gen_ep01_real_batch 同款; Counterfeit-V3.0 是漫画专用)
DEFAULT_CKPT = "DreamShaper_8_pruned.safetensors"

# 关键帧尺寸 = Wan i2v 原生桶(480×832 ÷16=30/52 均偶)。
# 别改回 512×912: 912÷16=57 奇触发 Wan addcmul off-by-one(model.py:993)。
KW, KH = 480, 832

FIT_PY = "../fitness-video-pipeline/.venv/Scripts/python.exe"  # manju venv 无 insightface, 借 fitness venv


# 下眼睑淡化(2026-07-21 用户审片反馈: 下眼帘偏厚显疲惫; 小雅 fit 变美人设该精神)。
# 正面词强明眸/平滑下眼睑 + REAL_NEG 反眼袋, 双向夹。arcface 不敏感 → face_sim 不受影响。
EYE_FIX = "bright fresh energetic eyes, smooth under-eye area, no eye bags, well-rested fresh look"


def build_positive(shot, extra=""):
    return f"{CHAR}, {SHOTS[shot]}, {REAL_STYLE}, {EYE_FIX}" + (f", {extra}" if extra else "")


def _face_sim_pair(ref, shot_path):
    """(sim,)。借 fitness venv insightface 子进程调 face_sim.py(manju venv 缺)。None=检不出脸。"""
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
    ap.add_argument("--shots", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--weight", type=float, default=0.95)  # 写实脸漂大, 收紧(ep01 教训#1)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    args = ap.parse_args()

    assert os.path.exists(REF), f"缺写实小雅立绘: {REF} (先 gen_xiaoya_real_lihui.py + det_score 挑脸定档)"
    shots = [int(x) for x in args.shots.split(",") if x.strip()]
    os.makedirs(args.out, exist_ok=True)

    c = ComfyUIClient(args.server)
    ref_name = c.upload_image(REF)
    print(f"[upload] ref(写实小雅立绘) -> {ref_name}")
    print(f"[cfg] {args.ckpt}  IPAdapter w={args.weight}  关键帧 {KW}x{KH} (Wan 原生桶, 喂 Wan 不 resize)\n")

    results = []
    for s in shots:
        pos = build_positive(s)
        wf = build_face_shot_workflow(
            ref_name, pos, REAL_NEG,
            ckpt=args.ckpt, seed=args.seed, ipadapter_weight=args.weight,
            width=KW, height=KH, filename_prefix=f"ep02_shot{s:02d}",
        )
        print(f"=== shot {s:2d} ===\n  {SHOTS[s]}", flush=True)
        outs = c.generate(wf, args.out)
        path = outs[0]
        if s in NO_FACE_SHOTS:
            print(f"  -> {os.path.basename(path)}  [skip face_sim 无主角脸镜]\n", flush=True)
            results.append((s, path, None, "[no-face-skip]"))
            continue
        sim = _face_sim_pair(REF, path)
        if sim is None:
            flag, simstr = "[no-face]", "N/A"
        elif sim >= 0.5:
            flag, simstr = "[OK]", f"{sim:.3f}"
        else:
            flag, simstr = "[FAIL?]", f"{sim:.3f}"
        print(f"  -> {os.path.basename(path)}  face_sim={simstr}  {flag}\n", flush=True)
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
