"""即梦/可灵 等 AI 出图右下角水印清洗。

用法:
    python clean_watermark.py <src> <dst> [w_pct=0.16] [h_pct=0.08]
默认抹掉右下角 宽16%×高8% 区域(即梦水印常见尺寸,含余量)。
背景纯色区 inpaint 干净;若人物延伸到该区,会糊,此时缩小 w_pct/h_pct 重跑。
支持中文路径(cv2 原生 imread/imwrite 不支持)。
"""
import os
import sys

import numpy as np
import cv2


def imread_unicode(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def imwrite_unicode(p, img):
    ext = os.path.splitext(p)[1] or ".png"
    cv2.imencode(ext, img)[1].tofile(p)


def clean_watermark(src, dst, w_pct=0.16, h_pct=0.08):
    img = imread_unicode(src)
    if img is None:
        raise SystemExit(f"[ERR] 读不了图: {src}")
    H, W = img.shape[:2]
    mask = np.zeros((H, W), np.uint8)
    x0, x1 = int(W * (1 - w_pct)), W
    y0, y1 = int(H * (1 - h_pct)), H
    mask[y0:y1, x0:x1] = 255
    if img.ndim == 3 and img.shape[2] == 4:
        out_bgr = cv2.inpaint(img[:, :, :3], mask, 3, cv2.INPAINT_TELEA)
        out = np.dstack([out_bgr, img[:, :, 3]])
    else:
        out = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    imwrite_unicode(dst, out)
    print(f"[OK] 去水印 -> {dst}  (mask 右下角 {w_pct*100:.0f}%×{h_pct*100:.0f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(1)
    clean_watermark(
        sys.argv[1], sys.argv[2],
        float(sys.argv[3]) if len(sys.argv) > 3 else 0.16,
        float(sys.argv[4]) if len(sys.argv) > 4 else 0.08,
    )
