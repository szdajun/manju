"""临时: face_sim命门 — 写实立绘 vs Seedance i2v 片段帧 arcface cosine。

≥0.45 (i2v clip 阈值) = 脸撑住运镜, 命门过。逐帧前向读(不 seek)避 h264 fast-seek
抽错帧坑(per memory ffmpeg-fast-seek-frame-extraction-artifact)。用完即删。
"""
import os
import sys

import cv2
import numpy as np
from insightface.app import FaceAnalysis

LIHUA = "../manju/output/lihui_real/主角立绘写实.png"
CLIP = "../manju/output/ep01_i2v_real/validate_lihua.mp4"
THRESH = 0.45

assert os.path.exists(LIHUA), f"缺立绘: {LIHUA}"
assert os.path.exists(CLIP), f"缺片段(Seedance 还没跑完?): {CLIP}"

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1, det_size=(640, 640))


def cv2_imread(path):
    # cv2.imread 不认 Windows CJK 路径(主角立绘写实.png)→ 走 bytes+imdecode
    with open(path, "rb") as f:
        arr = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def emb_of(bgr):
    fs = app.get(bgr)
    if not fs:
        return None, 0.0
    b = max(fs, key=lambda f: f.det_score)
    return b.normed_embedding, float(b.det_score)


# 立绘脸 embedding
e0, s0 = emb_of(cv2_imread(LIHUA))
if e0 is None:
    sys.exit("[fail] 立绘本身检不出脸, face_sim 无意义")
print(f"[lihua] det={s0:.4f} emb_dim={len(e0)}")

# 前向读整段, 按比例抽 9 帧(首/四分/中/四三/尾 + 中段密采)
cap = cv2.VideoCapture(CLIP)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
print(f"[clip] frames={total} fps={fps:.2f} dur={total/fps:.2f}s")
idxs = sorted(set([0, total // 4, total // 2, 3 * total // 4, max(0, total - 1)]
                  + [total // 2 + d for d in range(-8, 9, 2)]))
idxs = [i for i in idxs if 0 <= i < total]
targets = set(idxs)

kept = {}
fi = 0
while len(kept) < len(targets):
    ok, fr = cap.read()
    if not ok:
        break
    if fi in targets:
        kept[fi] = fr
    fi += 1
cap.release()

print(f"\n{'frame':>6} {'t(s)':>6} {'cos':>8} {'det':>7}")
coses = []
for fi in sorted(kept):
    e, s = emb_of(kept[fi])
    if e is None:
        print(f"{fi:>6} {fi/fps:>6.2f} {'no-face':>8} {0.0:>7.4f}")
        continue
    cos = float(np.dot(e0, e))
    coses.append(cos)
    print(f"{fi:>6} {fi/fps:>6.2f} {cos:>8.4f} {s:>7.4f}")

if not coses:
    print("\n[face_sim] i2v 帧全程无脸 — Seedance 没保住人脸, 命门爆")
    sys.exit(1)

mn, mean, mx = min(coses), float(np.mean(coses)), max(coses)
print(f"\n[face_sim] min={mn:.4f}  mean={mean:.4f}  max={mx:.4f}  (thresh={THRESH})")
verdict = "PASS [OK] 脸撑住运镜, 命门过" if mn >= THRESH else "FAIL [X] 脸漂移/丢, 命门爆"
print(f"[verdict] {verdict}  (gate=min, 最差帧也要 ≥{THRESH})")
if mn < THRESH <= mean:
    print("[note] min 挂但 mean 过 — 多数帧 OK, 个别帧脸偏; 看是否可接受或加锁脸")
