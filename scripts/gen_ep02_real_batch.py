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

from lib.comfyui_client import (  # noqa: E402
    ComfyUIClient, build_face_shot_workflow, build_dual_face_shot_workflow)
from lib.ep02_shots import (  # noqa: E402
    CHAR, TINGTING, SHOTS, SHOTS_REAL_OVERRIDE, REAL_STYLE, REAL_NEG)

# 脸 crop 版(治 IPAdapter center-crop 竖图丢脸 bug, 2026-07-21 小雅实案):
# 小雅立绘脸在图上部 17%, IPAdapter center-crop 切掉脸主体 → 锁身体不锁脸 → 关键帧 face_sim 0.23-0.38 全挂。
# scripts/crop_face.py 从 小雅立绘写实.png crop 纯脸成 512×512 方图, 喂 IPAdapter 不再丢脸。face_sim 金标准也用它(同一张脸)。
REF = "../manju/output/xiaoya_real/小雅立绘写实_face.png"
# 婷婷脸 ref(2026-07-22 Phase 2 新生立绘, 双人镜双锁脸用)
REF_TINGTING = "../manju/output/tingting_real/婷婷立绘写实_face.png"
OUT = "../manju/output/ep02_real"
NO_FACE_SHOTS = {6}  # shot6 纯手部特写无主角脸, face_sim 门 skip
# 双人镜(2026-07-22): 婷婷清晰可见且构图可左右分 → 双 IPAdapter 各锁一边脸。
#   2026-07-23 正反打: shot4/9 改"婷婷 solo 正面说话镜"(单锁婷婷, 见 SOLO_TINGTING),
#   双锁只剩 shot10 挽臂(两人都正面)。
DUAL_SHOTS = {10}
# 正反打(2026-07-23 治互动感+口型靶子): 婷婷台词镜改 solo 正面、视线偏画外(看小雅)。
#   单锁婷婷脸(REF_TINGTING) + SHOTS_REAL_OVERRIDE 婷婷正面构图。原 2=身后/4=侧身/9=身旁。
SOLO_TINGTING = {2, 4, 9, 12}  # +12: 收尾 tag 婷婷 power pose(2026-07-23)
MASK_LEFT = "../manju/output/ep02_real/_masks/mask_left.png"   # 小雅区
MASK_RIGHT = "../manju/output/ep02_real/_masks/mask_right.png"  # 婷婷区

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


def _strip_solo(c):
    """去掉 '1girl, solo, ' 前缀, 留角色体貌特征(双人镜两人并列不该有 solo)。"""
    return c.replace("1girl, solo, ", "")


def build_dual_positive(shot):
    """双人镜正向词: 显式左=小雅 / 右=婷婷, 配合左/右 mask 让双 IPAdapter 各锁一边。

    左右站位写死进 prompt(SD1.5 先把两人放对位置) + attn_mask 强制脸身份到对应半区,
    两层夹逼避免"婷婷生成在左边"和 mask 错位。
    """
    return (f"two young Chinese women standing side by side facing the camera, "
            f"on the left {_strip_solo(CHAR)}, on the right {_strip_solo(TINGTING)}, "
            f"{SHOTS[shot]}, "
            f"the slim fit long-straight-black-haired woman positioned on the left side of the frame, "
            f"the colleague with shoulder-length wavy dark brown hair positioned on the right side, "
            f"both women fully visible from head to waist and completely inside the frame, "
            f"centered wide composition showing both people entirely, "
            f"neither person cut off or partially out of frame, medium shot framing, "
            f"{REAL_STYLE}, {EYE_FIX}")


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
    ap.add_argument("--weight", type=float, default=0.95)  # 单人镜写实脸漂大, 收紧(ep01 教训#1)
    ap.add_argument("--dual-weight", type=float, default=0.9)  # 双人镜每个 adapter 略降, 叠加不过饱和
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--no-dual", action="store_true",
                    help="强制全部走单锁(调试用, 跳过双 IPAdapter)")
    ap.add_argument("--swap-masks", action="store_true",
                    help="左右 mask 对调喂(调试 mask 方向用)")
    args = ap.parse_args()

    assert os.path.exists(REF), f"缺写实小雅立绘: {REF} (先 gen_xiaoya_real_lihui.py + det_score 挑脸定档)"
    shots = [int(x) for x in args.shots.split(",") if x.strip()]
    os.makedirs(args.out, exist_ok=True)

    c = ComfyUIClient(args.server)
    ref_name = c.upload_image(REF)
    print(f"[upload] ref 小雅 -> {ref_name}")
    # 双人镜需要婷婷 ref + 左右 mask(只在有双人镜时上传)
    need_dual = any(s in DUAL_SHOTS for s in shots) and not args.no_dual
    need_tingting = need_dual or any(s in SOLO_TINGTING for s in shots)
    ref_t_name = mask_l_name = mask_r_name = mask_a_name = mask_b_name = None
    if need_tingting:
        assert os.path.exists(REF_TINGTING), f"缺婷婷立绘: {REF_TINGTING} (先 gen_tingting_real_lihui.py)"
        ref_t_name = c.upload_image(REF_TINGTING)
        print(f"[upload] ref 婷婷 -> {ref_t_name}")
    if need_dual:
        assert os.path.exists(MASK_LEFT) and os.path.exists(MASK_RIGHT), \
            f"缺 mask, 先跑 scripts/_bake_dual_masks.py"
        mask_l_name = c.upload_image(MASK_LEFT)   # 白在左(直觉命名)
        mask_r_name = c.upload_image(MASK_RIGHT)  # 白在右
        # ⚠ IPAdapter attn_mask 约定反向(empirical 2026-07-22 shot10 控制实验):
        #   白(alpha255)区 → adapter 把身份施到对侧(黑/alpha0 区)。
        #   故想让小雅在左 → 喂"白在右"的 mask_right; 婷婷在右 → 喂 mask_left。
        mask_a_name, mask_b_name = mask_r_name, mask_l_name   # 小雅←mask_right, 婷婷←mask_left
        if args.swap_masks:
            mask_a_name, mask_b_name = mask_b_name, mask_a_name
            print("[diag] swap-masks ON: 反转默认(诊断用)")
        print(f"[upload] 小雅mask={mask_a_name} 婷婷mask={mask_b_name} (双人镜)")
    print(f"[cfg] {args.ckpt}  单镜 w={args.weight} 双镜 w={args.dual_weight}  "
          f"关键帧 {KW}x{KH} (Wan 原生桶)  双人镜={sorted(s for s in shots if s in DUAL_SHOTS)}\n")

    results = []
    for s in shots:
        is_dual = s in DUAL_SHOTS and not args.no_dual
        is_tingting = s in SOLO_TINGTING   # 正反打: 婷婷 solo 正面说话镜
        # ⚠ SHOTS_REAL_OVERRIDE.get(s, SHOTS[s]) 会提前求值 SHOTS[s] 作默认参数 →
        # shot11/12 只在 override 里(SHOTS 没有)就先 KeyError(2026-07-23 tag 实案)。改懒求值。
        shot_text = SHOTS_REAL_OVERRIDE.get(s, SHOTS.get(s))
        if is_dual:
            pos = build_dual_positive(s)
            wf = build_dual_face_shot_workflow(
                ref_name, ref_t_name, mask_a_name, mask_b_name,
                pos, REAL_NEG,
                ckpt=args.ckpt, seed=args.seed, ipadapter_weight=args.dual_weight,
                width=KW, height=KH, filename_prefix=f"ep02_shot{s:02d}",
            )
            tag = " [DUAL 双锁: 左小雅/右婷婷]"
            sim_ref = REF  # 双人镜主脸是小雅(左)
        elif is_tingting:
            pos = f"{TINGTING}, {shot_text}, {REAL_STYLE}, {EYE_FIX}"
            wf = build_face_shot_workflow(
                ref_t_name, pos, REAL_NEG,
                ckpt=args.ckpt, seed=args.seed, ipadapter_weight=args.weight,
                width=KW, height=KH, filename_prefix=f"ep02_shot{s:02d}",
            )
            tag = " [SOLO 婷婷正面(正反打)]"
            sim_ref = REF_TINGTING
        else:
            pos = f"{CHAR}, {shot_text}, {REAL_STYLE}, {EYE_FIX}"
            wf = build_face_shot_workflow(
                ref_name, pos, REAL_NEG,
                ckpt=args.ckpt, seed=args.seed, ipadapter_weight=args.weight,
                width=KW, height=KH, filename_prefix=f"ep02_shot{s:02d}",
            )
            tag = ""
            sim_ref = REF
        print(f"=== shot {s:2d} ==={tag}\n  {shot_text}", flush=True)
        outs = c.generate(wf, args.out)
        path = outs[0]
        if s in NO_FACE_SHOTS:
            print(f"  -> {os.path.basename(path)}  [skip face_sim 无主角脸镜]\n", flush=True)
            results.append((s, path, None, "[no-face-skip]", is_dual))
            continue
        sim = _face_sim_pair(sim_ref, path)
        if sim is None:
            flag, simstr = "[no-face]", "N/A"
        elif sim >= 0.5:
            flag, simstr = "[OK]", f"{sim:.3f}"
        else:
            flag, simstr = "[FAIL?]", f"{sim:.3f}"
        # 双人镜: 小雅主脸 face_sim 仍要过; 婷婷局部眼睑只能肉眼(contact sheet)
        extra = "  婷婷脸=肉眼判" if is_dual else ""
        print(f"  -> {os.path.basename(path)}  小雅face_sim={simstr}  {flag}{extra}\n", flush=True)
        results.append((s, path, sim, flag, is_dual))

    print("===== 汇总 =====")
    fails = 0
    for s, p, sim, flag, is_dual in results:
        simstr = f"{sim:.3f}" if sim is not None else "N/A"
        dual_tag = " [DUAL]" if is_dual else ""
        print(f"  shot {s:2d}  sim={simstr}  {flag}{dual_tag}  {os.path.basename(p)}")
        if flag not in ("[OK]", "[no-face-skip]"):
            fails += 1
    print(f"\n[batch done] {len(results)} 镜, {fails} 镜待复核(重抽: --shots <号> --seed <新>)")
    print("[注] 双人镜(4/9/10)婷婷脸 + 全镜眼睑肉眼判: 跑 scripts/_ep02_keyframe_contact.py 出图")


if __name__ == "__main__":
    main()
