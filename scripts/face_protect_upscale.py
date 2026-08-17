# -*- coding: utf-8 -*-
"""脸区保护: 把"用户认可的原始脸"贴回超分帧, 治 Real-ESRGAN 把脸超成僵尸.

根因 (用户证词钉死, 2026-08-11):
- 原始生成脸 (448×672) 用户认可 "不错".
- Real-ESRGAN realesr-animevideov3 超分后脸变 "僵尸/难看" — anime 模型用动漫先验
  重建小脸区 → 死皮/空洞/非人感. 身/背景的锐化增益是真的想要的, 不能放弃超分.
→ 解法: 保留超分的身/背景, **只把脸区换回原始脸** (lanczos 放大 + mild unsharp).

输入必须帧对齐: 用裁首2帧后的原始 (.._t2.mp4, 35帧) + 同源超分 (..._t2_2x.mp4, 35帧).
逐帧: insightface 测超分帧脸框 → 椭圆 feather mask → 贴原始脸. 测不到脸 (背身) 的帧
保持超分原样 (背身脸不可见, 僵尸感无所谓).

跑 (fitness venv 有 insightface):
  /f/wkspace/fitness-video-pipeline/.venv/Scripts/python.exe scripts/face_protect_upscale.py \
      --orig output/wan_animate/client_a_448x672_matchsrc_00001__t2.mp4 \
      --upscaled output/wan_animate/client_a_896x1344_matchsrc_t2_2x.mp4 \
      --out output/wan_animate/client_a_896x1344_matchsrc_t2_2x_faceprot.mp4
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

FFMPEG = r"~/ffmpeg/ffmpeg.exe"


def make_face_app():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def feather_mask(H, W, bbox, expand=0.35):
    """脸框→椭圆羽化 mask (脸中心 1, 边缘渐 0). expand 放大覆盖额头/下颌."""
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    ax = max((x2 - x1) / 2.0 * (1 + expand), 8.0)
    ay = max((y2 - y1) / 2.0 * (1 + expand), 8.0)
    mask = np.zeros((H, W), np.float32)
    cv2.ellipse(mask, (int(cx), int(cy)), (int(ax), int(ay)), 0, 0, 360, 1.0, -1)
    k = max(int(min(ax, ay) * 0.5) * 2 + 1, 11)
    return cv2.GaussianBlur(mask, (k, k), 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True, help="裁后原始低分 mp4 (脸好), 与 upscaled 帧对齐")
    ap.add_argument("--upscaled", required=True, help="超分 mp4 (身锐脸僵尸)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--unsharp", type=float, default=0.5,
                    help="贴回脸的 mild unsharp 强度 (治 lanczos 放大偏软; 0=关)")
    ap.add_argument("--crf", type=int, default=16)
    args = ap.parse_args()

    cap_o = cv2.VideoCapture(args.orig)
    cap_u = cv2.VideoCapture(args.upscaled)
    if not cap_o.isOpened() or not cap_u.isOpened():
        print("[ERR] 打不开输入", file=sys.stderr); sys.exit(2)
    W = int(cap_u.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap_u.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_u.get(cv2.CAP_PROP_FPS) or 16.0

    app = make_face_app()
    tmp = tempfile.mkdtemp(prefix="faceprot_")
    outdir = os.path.join(tmp, "out"); os.makedirs(outdir)
    n = prot = 0
    try:
        while True:
            ok_o, fo = cap_o.read()
            ok_u, fu = cap_u.read()
            if not (ok_o and ok_u):
                break
            # 原始脸放大到超分尺寸 + mild unsharp (lanczos 偏软, 补一点锐度别太软)
            fo_big = cv2.resize(fo, (W, H), interpolation=cv2.INTER_LANCZOS4)
            if args.unsharp > 0:
                blur = cv2.GaussianBlur(fo_big, (0, 0), 1.2)
                fo_big = cv2.addWeighted(fo_big, 1 + args.unsharp, blur, -args.unsharp, 0)
            # insightface 在超分帧测脸 (RGB)
            faces = app.get(fu[:, :, ::-1])
            if faces:
                f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                m = feather_mask(H, W, f.bbox)[..., None]
                out = (fu * (1 - m) + fo_big * m).astype(np.uint8)
                prot += 1
            else:
                out = fu  # 背身/漏检: 保持超分 (脸不可见)
            cv2.imwrite(os.path.join(outdir, f"f{n:08d}.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 96])
            n += 1
            if n % 10 == 0:
                print(f"  ...{n} 帧 ({prot} 贴脸)")
        cap_o.release(); cap_u.release()
        subprocess.run([FFMPEG, "-y", "-framerate", str(fps),
                        "-i", os.path.join(outdir, "f%08d.jpg"),
                        "-c:v", "libx264", "-crf", str(args.crf),
                        "-pix_fmt", "yuv420p", "-r", str(fps), args.out],
                       check=True, capture_output=True)
        print(f"[done] {n}帧, {prot}帧贴回原始脸 ({prot / max(n, 1) * 100:.0f}%) -> {os.path.basename(args.out)}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
