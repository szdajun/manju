"""i2v 输出人脸一致性校验 (2026-08-15, P0).

逐镜抽帧 + insightface 嵌入, 检查各镜输出是否为同一个人.

指标:
  - **跨镜一致性 (主)**: 各镜嵌入两两余弦; 每镜对其它镜的平均相似度 < 阈值 → [DRIFT] 漂移镜.
    ⚠ H3 重渲脸, vs 角色 ref 天然 ~0-0.2, 所以主判据是跨镜是否同一人 (本会话校准:
    H3 健康输出跨镜 sim ~0.54-0.75, 默认阈值 0.45 低于健康下限留余量).
  - **vs 角色 ref (辅, 信息性)**: 各镜 vs face_ref 余弦 (H3 会 ~0, 仅参考; SD/FLUX 系可用).

用法 (fitness venv, 有 insightface):
  ../fitness-video-pipeline/.venv/Scripts/python.exe scripts/check_i2v_face.py --ep 04
  ../fitness-video-pipeline/.venv/Scripts/python.exe scripts/check_i2v_face.py --ep 04 --i2v-dir ... --threshold 0.5

退出码: 有漂移镜 → 1; 否则 0.
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

MANJU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MANJU)

from lib.storyboard import character, load_script  # noqa: E402

# 校准值 (2026-08-15, ep03/ep04 H3 输出): 健康跨镜 sim 均值 0.66-0.70, min 0.54.
DEFAULT_THRESHOLD = 0.45


def _make_app():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def _imread_unicode(p):
    """CJK 安全读图 (np.fromfile + imdecode)."""
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def _img_embedding(app, img):
    """图 → 主脸 normed embedding (无脸返 None)."""
    faces = app.get(img)
    if not faces:
        return None
    return max(faces, key=lambda f: f.det_score).normed_embedding


def _frame_embedding(app, mp4, frame_idx=None):
    """取 mp4 中段帧 → 主脸 normed embedding (无脸返 None)."""
    cap = cv2.VideoCapture(mp4)
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx is None:
        frame_idx = max(0, n // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_idx, max(0, n - 1)))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return _img_embedding(app, frame)


def cross_shot_consistency(embs):
    """跨镜一致性: 每镜 vs 其它镜平均余弦 → dict{shot: sim}."""
    keys = list(embs)
    result = {}
    for i, k in enumerate(keys):
        others = [np.dot(embs[k], embs[j]) for j in keys if j != k]
        result[k] = float(np.mean(others)) if others else 1.0
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", default="03")
    ap.add_argument("--i2v-dir", default=None,
                    help="i2v 目录 (默认 output/ep{ep}_i2v_real/_seedance)")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"跨镜漂移阈值 (默认 {DEFAULT_THRESHOLD}; H3 健康 ~0.54-0.75)")
    ap.add_argument("--frame", type=int, default=None, help="抽帧索引 (默认中段)")
    ap.add_argument("--no-ref", action="store_true", help="不对比角色 ref")
    args = ap.parse_args()

    script = load_script(args.ep)
    ep = script["ep"]
    i2v_dir = args.i2v_dir or os.path.join(MANJU, "output", f"ep{ep}_i2v_real", "_seedance")
    prefix = f"ep{ep}_shot"
    shots = [int(x) for x in script["shots"].keys()]

    app = _make_app()
    # 角色 ref (默认首个有 face_ref 的角色)
    ref_emb, ref_name = None, None
    if not args.no_ref:
        for ck, ch in script["characters"].items():
            fr = ch.get("face_ref")
            if fr and os.path.exists(fr):
                ref_emb = _img_embedding(app, _imread_unicode(fr))
                ref_name = ck
                break
    embs = {}
    for s in shots:
        cands = sorted(glob.glob(os.path.join(i2v_dir, f"{prefix}{s:02d}.mp4")))
        if not cands:
            print(f"[skip] shot{s:02d} 缺 i2v 输出: {prefix}{s:02d}.mp4")
            continue
        e = _frame_embedding(app, cands[-1], args.frame)
        if e is None:
            print(f"[skip] shot{s:02d} 无脸 (中段帧)")
            continue
        embs[s] = e

    if len(embs) < 2:
        print("[ERR] 有效镜 < 2, 无法跨镜校验")
        return 2

    consis = cross_shot_consistency(embs)
    print(f"\n=== ep{ep} i2v 人脸一致性 ({len(embs)} 镜, 跨镜阈值 {args.threshold}) ===")
    drift = []
    for s in shots:
        if s not in embs:
            continue
        sim = consis[s]
        flag = "  [DRIFT]" if sim < args.threshold else ""
        ref_s = ""
        if ref_emb is not None:
            ref_s = f"  vs-ref={np.dot(embs[s], ref_emb):.2f}"
        print(f"  shot{s:02d}: 跨镜={sim:.3f}{ref_s}{flag}")
        if flag:
            drift.append(s)
    avg = np.mean([consis[s] for s in embs])
    print(f"\n  平均跨镜={avg:.3f}  漂移镜: {drift or '无'}")
    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
