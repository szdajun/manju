"""ep02 写实版 i2v 全量换 Seedance (云端 doubao-seedance-1-0-pro-250528)。

根因(2026-07-22 系统调试 Phase 4.5, 用户"是不是考虑换/升级模型"): 厚眼帘/浓妆是
Wan2.2-5B 在 12GB 死路下的固有倾向, 正/负 prompt 都压不住("控制不住"), 立绘/关键帧
干净但 Wan i2v 出妆浓。本地升级被 12GB 挡死(14B/Hunyuan 需 ~24GB)。→ 换模型族:
Seedance 云端 i2v (零 VRAM, 不同先验, ep01 已验证整片可组装)。

shot05 A/B 已证: Seedance 能生成本镜, 脸不崩不漂, 格式 480×864@24fps 与 ep01 一致
(组装 proven)。vision 判 Wan/Seed 都"干净"但 Wan 正是用户 v4 拒掉的源 → vision 非
eye-gate, 用户才是 → 跑全片组装 v5 让用户 eye-gate 决生死。

正/负 prompt: 用 I2V_MOTION 运动意图 + FACE_POS 正向脸锚(ep01 教训#1 Seedance 头动
大脸漂, 拼后缀降漂; 再加薄眼帘/素颜正向, 因正向 > 负向)。shot06 纯手部无脸 → 不拼
FACE_POS。

增量: 输出已存在则 skip。

跑 manju venv (纯 stdlib + urllib, 复用 lib.video_gen.seedance_i2v):
  /f/wkspace/manju/.venv/Scripts/python.exe scripts/gen_ep02_seedance.py
  /f/wkspace/manju/.venv/Scripts/python.exe scripts/gen_ep02_seedance.py --shots 3,5,7
产物: output/ep02_i2v_real/_seedance/ep02_shotNN.mp4
"""
import argparse
import glob
import os
import sys

MANJU = "../manju"
sys.path.insert(0, MANJU)
from lib.video_gen import seedance_i2v  # noqa: E402
from lib.ep02_shots import I2V_MOTION, I2V_MOTION_REAL_OVERRIDE  # noqa: E402

KF_DIR = f"{MANJU}/output/ep02_real"
OUT = f"{MANJU}/output/ep02_i2v_real/_seedance"

# 正向脸锚: 降 Seedance 头动脸漂(ep01 教训#1) + 薄眼帘/素颜正向(正向 > 负向, 文本能压的极限)。
FACE_POS = (
    ", face stable and centered, minimal head movement, face clearly visible throughout, "
    "bare fresh natural face, no makeup, nude natural lips, no eyeliner, no eyeshadow, "
    "thin smooth natural eyelids, wide open bright alert eyes, clean smooth under-eye area"
)

# 纯手部/无脸镜不拼脸锚(无眼帘问题, 脸锚反而矛盾)
NO_FACE_SHOTS = {6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--duration", type=int, default=5)
    ap.add_argument("--resolution", default="480p")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    shots = [int(x) for x in args.shots.split(",") if x.strip()]
    ok, skip, fail = [], [], []
    for s in shots:
        out = os.path.join(OUT, f"ep02_shot{s:02d}.mp4")
        if os.path.exists(out):
            print(f"[skip] shot{s:02d} 已存在: {os.path.basename(out)}", flush=True)
            skip.append(s)
            continue
        cands = sorted(glob.glob(os.path.join(KF_DIR, f"ep02_shot{s:02d}_*.png")))
        if not cands:
            print(f"[ERR] shot{s:02d} 关键帧缺: ep02_shot{s:02d}_*.png in {KF_DIR}", flush=True)
            fail.append(s)
            continue
        src = cands[-1]
        # 懒求值(2026-07-23): .get 的默认参会被 eagerly 求值, shot11/12 只在 override 里 →
        # `get(s, I2V_MOTION[s])` 会先 KeyError 在 I2V_MOTION[11]。改 `or` 链惰性兜底。
        motion = I2V_MOTION_REAL_OVERRIDE.get(s) or I2V_MOTION.get(s)
        if not motion:
            print(f"[ERR] shot{s:02d} 无 motion 意图(override + I2V_MOTION 都缺)", flush=True)
            fail.append(s)
            continue
        prompt = motion if s in NO_FACE_SHOTS else motion + FACE_POS
        print(f"[shot{s:02d}] KF={os.path.basename(src)} face_anchor={'N(手部)' if s in NO_FACE_SHOTS else 'Y'}",
              flush=True)
        try:
            seedance_i2v(src, prompt, out, duration=args.duration,
                         ratio="9:16", resolution=args.resolution)
            ok.append(s)
        except Exception as e:
            print(f"[FAIL] shot{s:02d}: {e}", flush=True)
            fail.append(s)
    print(f"\n[done] ok={ok} skip={skip} fail={fail}", flush=True)
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
