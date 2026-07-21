"""临时: Seedance i2v 片段 face_sim 命门批量验(每段 vs 写实立绘, ≥0.45 过)。
前向读抽 5 帧(首/四分/中/四三/尾)避 h264 fast-seek 抽错帧。CJK 路径 imdecode。用完即删。"""
import glob
import os

import cv2
import numpy as np
from insightface.app import FaceAnalysis

LIHUA = "../manju/output/lihui_real/主角立绘写实.png"
CLIPS_DIR = "../manju/output/ep01_i2v_real"
THRESH = 0.45

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1, det_size=(640, 640))


def cv2_imread(p):
    with open(p, "rb") as f:
        arr = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def emb_of(bgr):
    fs = app.get(bgr)
    if not fs:
        return None
    b = max(fs, key=lambda f: f.det_score)
    return b.normed_embedding


e0 = emb_of(cv2_imread(LIHUA))
assert e0 is not None, "立绘无脸?!"

clips = sorted(glob.glob(os.path.join(CLIPS_DIR, "ep01_shot*.mp4")))
assert clips, f"无片段 in {CLIPS_DIR} (i2v 还没跑完?)"
print(f"[found] {len(clips)} 片段\n")
print(f"{'clip':<26} {'min':>6} {'mean':>6} {'max':>6}  verdict")

allpass = True
for cp in clips:
    cap = cv2.VideoCapture(cp)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = sorted(set([0, total // 4, total // 2, 3 * total // 4, max(0, total - 1)]))
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
    coses = []
    for k in sorted(kept):
        e = emb_of(kept[k])
        if e is not None:
            coses.append(float(np.dot(e0, e)))
    nm = os.path.basename(cp)
    if not coses:
        print(f"{nm:<26} {'noface':>6}                    FAIL")
        allpass = False
        continue
    mn, mean, mx = min(coses), float(np.mean(coses)), max(coses)
    v = "PASS" if mn >= THRESH else "FAIL"
    if v == "FAIL":
        allpass = False
    print(f"{nm:<26} {mn:>6.3f} {mean:>6.3f} {mx:>6.3f}  {v}")

print(f"\n[overall] {'ALL PASS [OK]' if allpass else 'SOME FAIL [X]'} @ i2v thresh {THRESH}")
