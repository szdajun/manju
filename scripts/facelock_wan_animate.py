"""Wan-Animate 换脸锁脸 (facelock) — 接 fitness 验证过的后处理 (color_match + skin_and_lip_fix).

manju 旧 facelock (face_swap_batch 裸 inswapper paste_back 模式) 缺后处理 → 脸偏暖/塑料感/
浮在画面上 ("贴上去"). 本脚本 port fitness tools/face_swap.py 两道后处理 (LOCKED recipe):
  1. color_match_face: 换脸区 LAB Reinhard 色彩迁向换脸前原脸 (场景光), 治暖脸/冷颈
  2. skin_and_lip_fix: 脸区亮度 L 对齐脸框下方现场皮肤 (脖子/胸) + 唇色淡化, 治亮度突兀

源照默认client_a四视图正面 (Krea2 identity-lock; 用户拍板 "一直用client_a的四视图"). 四视图是扩散
模型生成、本身偏软, 比 high-res 真人照更贴合 Wan-Animate 软身体, 缓解"脸比身体清晰"塑料感.

跑在 fitness venv (insightface + inswapper_128 模型):
  /f/wkspace/fitness-video-pipeline/.venv/Scripts/python.exe scripts/facelock_wan_animate.py --input <video>
产物: output/wan_animate/{stem}_facelock.mp4
"""
import argparse
import os
import sys

import cv2
import numpy as np

BASE = "../manju"
WAN_DIR = f"{BASE}/output/wan_animate"
SWAPPER_MODEL = os.path.expanduser("~/.insightface/models/inswapper_128.onnx")
DET_SIZE = 640
DEFAULT_SRC = f"{BASE}/assets/characters/client_a/krea2id_client_a_正面_00002_.png"


def imread_unicode(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def get_app():
    import insightface
    app = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(DET_SIZE, DET_SIZE))
    return app


def get_swapper():
    import insightface
    if not os.path.exists(SWAPPER_MODEL):
        raise FileNotFoundError(f"缺 inswapper: {SWAPPER_MODEL}")
    return insightface.model_zoo.get_model(SWAPPER_MODEL, providers=["CPUExecutionProvider"])


def primary_face(app, img):
    fs = app.get(img)
    if not fs:
        return None
    return max(fs, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


# ---- ported verbatim from fitness tools/face_swap.py (LOCKED recipe, 勿改) ----
def color_match_face(frame, bbox, ref_roi, strength=0.8):
    """把 frame[bbox] 换脸区域的色彩向 ref_roi(换脸前的原肤色)迁移, 消除偏色/过白.

    用 LAB 空间 Reinhard 迁移 (匹配均值+方差), 只改颜色不改五官/纹理 → 保留换脸身份,
    但肤色/光感回归原场景. strength: 0=不迁移, 1=完全用原肤色. 默认 0.8.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = y2 - y1, x2 - x1
    if h < 10 or w < 10 or ref_roi is None:
        return frame
    tgt = frame[y1:y2, x1:x2]
    # inswapper paste_back=True 偶尔改 1-2 px → tgt/ref 尺寸不齐 broadcast 崩; 加 size 防御.
    if tgt.shape[0] < 10 or tgt.shape[1] < 10:
        return frame
    if ref_roi.shape[0] != tgt.shape[0] or ref_roi.shape[1] != tgt.shape[1]:
        ref = cv2.resize(ref_roi, (tgt.shape[1], tgt.shape[0]))
    else:
        ref = ref_roi
    tgt_lab = cv2.cvtColor(tgt, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = tgt_lab.copy()
    for c in range(3):
        tm, ts = tgt_lab[..., c].mean(), tgt_lab[..., c].std() + 1e-6
        rm, rs = ref_lab[..., c].mean(), ref_lab[..., c].std() + 1e-6
        migrated = (tgt_lab[..., c] - tm) / ts * rs + rm
        out[..., c] = tgt_lab[..., c] * (1 - strength) + migrated * strength
    out_bgr = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    # 羽化 mask (中心全迁移, 边缘渐变回换脸结果)
    mask = np.zeros((h, w), dtype=np.float32)
    im_ = max(5, min(h, w) // 12)
    cv2.rectangle(mask, (im_, im_), (w - im_, h - im_), 1.0, -1)
    bk = max(3, min(h, w) // 6)
    bk += bk % 2 == 0
    mask = cv2.GaussianBlur(mask, (bk, bk), 0)
    m3 = np.stack([mask] * 3, axis=-1)
    frame[y1:y2, x1:x2] = (
        tgt.astype(np.float32) * (1.0 - m3) + out_bgr.astype(np.float32) * m3
    ).astype(np.uint8)
    return frame


def skin_and_lip_fix(frame, lead_bbox, app, lip_desat=0.35, lum_blend=0.6):
    """换脸后皮肤和谐 + 唇色淡化.

    在换脸后的整帧上, 用 lead_bbox 定位人脸:
      1. 皮肤: 脸区亮度(L)对齐脸框下方周边肤色(脖子/胸=现场皮肤) → 边缘宽羽化 → 自然过渡.
      2. 唇色: 用 landmark_3d_68 定位嘴唇, 降饱和度 + 微提亮.
    """
    if lead_bbox is None:
        return frame
    x1, y1, x2, y2 = [int(v) for v in lead_bbox]
    h, w = frame.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    if x2 - x1 < 20 or y2 - y1 < 20:
        return frame
    faces = app.get(frame[y1:y2, x1:x2])
    if not faces:
        return frame
    lead = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    fx1, fy1, fx2, fy2 = [int(v) for v in lead.bbox]
    gx1, gy1 = x1 + fx1, y1 + fy1
    gx2, gy2 = x1 + fx2, y1 + fy2
    pw = (gx2 - gx1) // 2
    ox1, oy1 = max(0, gx1 - pw), max(0, gy1 - pw)
    ox2, oy2 = min(w, gx2 + pw), min(h, gy2 + pw)
    if ox2 - ox1 < 30 or oy2 - oy1 < 30:
        return frame
    region = frame[oy1:oy2, ox1:ox2].copy()
    rh, rw = region.shape[:2]
    lx1, ly1 = gx1 - ox1, gy1 - oy1
    lx2, ly2 = gx2 - ox1, gy2 - oy1

    # 1) 皮肤亮度对齐 (脸区 L 对齐脸框下方现场肤色)
    lab_all = cv2.cvtColor(region, cv2.COLOR_BGR2LAB).astype(np.float32)
    mask_in = np.zeros((rh, rw), bool)
    mask_in[ly1:ly2, lx1:lx2] = True
    outer = ~mask_in
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    H = hsv[..., 0].astype(int); S = hsv[..., 1].astype(int); V = hsv[..., 2].astype(int)
    skin = ((H < 25) | (H > 165)) & (S > 20) & (S < 180) & (V > 40)
    # 周边肤色参考取【脸框下方】(脖子/胸 = 现场皮肤), 不用脸框紧邻一圈(主要是头发/阴影/背景).
    outer_skin = (outer & skin) & (np.arange(rh)[:, None] >= ly2 - 5)
    if outer_skin.sum() < 50:
        outer_skin = outer & skin  # 下方无足够皮肤(仰头/遮挡) fallback 紧邻一圈
    inner_skin = mask_in & skin
    if outer_skin.sum() > 50 and inner_skin.sum() > 30:
        outer_L = lab_all[..., 0][outer_skin].mean()
        inner_L = lab_all[..., 0][inner_skin].mean()
        delta = (outer_L - inner_L) * lum_blend
        lab_all[..., 0][mask_in] = np.clip(lab_all[..., 0][mask_in] + delta, 0, 255)
        maskf = mask_in.astype(np.float32)
        maskf = cv2.GaussianBlur(maskf, (61, 61), 0)
        m3 = np.stack([maskf] * 3, axis=-1)
        adjusted = cv2.cvtColor(np.clip(lab_all, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        region = (region.astype(np.float32) * (1 - m3) + adjusted.astype(np.float32) * m3).astype(np.uint8)

    # 2) 唇色淡化 (landmark 定位唇, 降饱和 + 微提亮)
    if hasattr(lead, "landmark_3d_68") and lead.landmark_3d_68 is not None:
        lm = lead.landmark_3d_68
        lip = np.array([(p[0], p[1]) for p in lm[48:60]], dtype=np.int32)
        lip_region = lip + np.array([x1 - ox1, y1 - oy1])
        lip_region[:, 0] = np.clip(lip_region[:, 0], 0, rw - 1)
        lip_region[:, 1] = np.clip(lip_region[:, 1], 0, rh - 1)
        lip_mask = np.zeros((rh, rw), np.uint8)
        cv2.fillConvexPoly(lip_mask, lip_region.reshape(-1, 1, 2), 255)
        lip_mask = cv2.GaussianBlur(lip_mask, (15, 15), 0)
        hsv_r = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).astype(np.float32)
        lm3 = lip_mask.astype(np.float32) / 255.0 * lip_desat
        hsv_r[..., 1] = np.clip(hsv_r[..., 1] - hsv_r[..., 1] * lm3, 0, 255)
        hsv_r[..., 2] = np.clip(hsv_r[..., 2] + 20 * lm3, 0, 255)
        region = cv2.cvtColor(np.clip(hsv_r, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

    frame[oy1:oy2, ox1:ox2] = region
    return frame


# ---- end ported ----


def facelock(app, swapper, src_face, infile, outfile, color_match_strength=0.8):
    """逐帧: 检测最大脸 → 存换脸前原脸 → inswapper paste_back → color_match → skin_and_lip_fix."""
    cap = cv2.VideoCapture(infile)
    if not cap.isOpened():
        print(f"[ERR] 打不开 {infile}", file=sys.stderr)
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # manju 短片预览已知可用; face_swap_batch 同款
    writer = cv2.VideoWriter(outfile, fourcc, fps, (w, h))
    n = 0
    n_swap = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces = app.get(frame)
        if faces:
            lead = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            x1, y1, x2, y2 = [int(v) for v in lead.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            orig_face_roi = frame[y1:y2, x1:x2].copy()  # 换脸前原脸 (color_match 场景光参考)
            out = swapper.get(frame, lead, src_face, paste_back=True)
            if color_match_strength > 0 and orig_face_roi.size > 0:
                out = color_match_face(out, lead.bbox, orig_face_roi, color_match_strength)
            out = skin_and_lip_fix(out, lead.bbox, app)
            n_swap += 1
        else:
            out = frame
        writer.write(out)
        n += 1
    cap.release()
    writer.release()
    print(f"  {n}帧 换脸{n_swap}({100*n_swap//max(n,1)}%) -> {os.path.basename(outfile)}", flush=True)
    return outfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=f"{WAN_DIR}/client_a_768x1152_fullbody_00001_.mp4",
                    help="Wan-Animate 原始输出视频 (未换脸)")
    ap.add_argument("--src", default=DEFAULT_SRC, help="源脸照 (默认client_a四视图正面)")
    ap.add_argument("--out", default=None, help="输出路径 (默认 {stem}_facelock.mp4)")
    ap.add_argument("--color-match", type=float, default=0.8, help="color_match 强度 0-1 (默认 0.8)")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERR] 输入视频不存在: {args.input}", file=sys.stderr); sys.exit(2)
    stem = os.path.splitext(os.path.basename(args.input))[0]
    out = args.out or os.path.join(WAN_DIR, f"{stem}_facelock.mp4")

    app = get_app()
    swapper = get_swapper()
    src_img = imread_unicode(args.src)
    src_face = primary_face(app, src_img)
    assert src_face is not None, f"源照检出脸失败: {args.src}"
    print(f"[src] {os.path.basename(args.src)} (client_a四视图正面)  color_match={args.color_match}\n", flush=True)

    facelock(app, swapper, src_face, args.input, out, args.color_match)
    print(f"\n[done] {out}")


if __name__ == "__main__":
    sys.exit(main() or 0)
