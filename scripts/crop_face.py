"""crop 立绘的脸成正方形, 喂 IPAdapter-plus-face。

治 IPAdapter center-crop 竖图丢脸 bug: 竖图立绘(512×912)被 CLIPImageProcessor
center-crop 成方图, 若脸在图上部(头肩/半身构图脸 cy<30%), 脸主体被切 → IPAdapter
锁身体不锁脸 → 关键帧 face_sim 0.23-0.38 全挂(小雅实案 2026-07-21)。
预 crop 纯脸成正方形喂 IPAdapter, center-crop 不再丢脸 → 锁脸恢复。

insightface 取最大脸 → 脸长边×1.8 做正方形(含发际/下巴/耳) → 以脸中心 crop → resize 512。

用法:
    python scripts/crop_face.py F:/.../小雅立绘写实.png            # -> 同目录 _face.png
    python scripts/crop_face.py F:/.../小雅立绘写实.png out.png
"""
import sys

import cv2
import numpy as np
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))

src = sys.argv[1]
dst = sys.argv[2] if len(sys.argv) > 2 else src.replace(".png", "_face.png")
img = cv2.imdecode(np.fromfile(src, dtype=np.uint8), cv2.IMREAD_COLOR)
h, w = img.shape[:2]
fs = app.get(img)
assert fs, f"no face in {src}"
fs.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
x1, y1, x2, y2 = [int(v) for v in fs[0].bbox]
fw, fh = x2 - x1, y2 - y1
# 正方形边长 = 脸长边×1.8(含发际/下巴/耳), 以脸中心为中心
side = int(max(fw, fh) * 1.8)
cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
x0 = max(0, cx - side // 2)
y0 = max(0, cy - side // 2)
x1n = min(w, x0 + side)
y1n = min(h, y0 + side)
side = min(x1n - x0, y1n - y0)  # clamp 到边界后取实际正方形
x1n, y1n = x0 + side, y0 + side
crop = img[y0:y1n, x0:x1n]
out = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA)
ok, buf = cv2.imencode(".png", out)
buf.tofile(dst)  # CJK 路径 tofile 替代 imwrite
print(f"[OK] {dst}  face[{x1},{y1},{x2},{y2}] side={side} -> 512x512")
