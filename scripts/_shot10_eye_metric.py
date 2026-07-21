"""shot10 双人镜眼睛稳定性客观测法 (2026-07-22)。

face_sim 命门对局部眼睛瑕疵不敏感(arcface 余弦看整体身份), 用户报的"末段左脸眼睛坏"
是局部 artifact, 需另测。本脚本: insightface 检两人脸, 按bbox中心x分左/右, 取每张脸的
眼睛横带(脸高 18-46%), 算灰度向量余弦: ① vs 关键帧该脸眼带(漂离源) ② 逐帧自稳定。
正常眨眼高稳定; 末段融化/扭曲/不对称 → 余弦掉。跑在 fitness venv(有 insightface)。

用法: FIT_PY scripts/_shot10_eye_metric.py <frames_dir>  (dir 内 keyframe.png + fNN.Ns.png)
"""
import os
import sys
import glob
import re

import cv2
import numpy as np
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))


def eye_strip_vec(img, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    h = y2 - y1
    ey1 = y1 + int(h * 0.18)
    ey2 = y1 + int(h * 0.46)
    crop = img[max(0, ey1):ey2, max(0, x1):x2]
    if crop.size == 0:
        return None
    g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (32, 16))
    v = g.flatten().astype(np.float32)
    v -= v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cos(a, b):
    return float(np.dot(a, b))


def detect_lr(img):
    """返 (left_face, right_face) 各自 (box, eye_vec); 按中心x分。"""
    faces = app.get(img)
    if len(faces) < 2:
        return None, None
    faces.sort(key=lambda f: (f.bbox[0] + f.bbox[2]) / 2)
    left, right = faces[0], faces[-1]
    lv = eye_strip_vec(img, left.bbox)
    rv = eye_strip_vec(img, right.bbox)
    return (left.bbox, lv), (right.bbox, rv)


def t_of(name):
    m = re.search(r"f([\d.]+)s", name)
    return float(m.group(1)) if m else -1


def main():
    d = sys.argv[1]
    kf = os.path.join(d, "keyframe.png")
    frames = sorted(glob.glob(os.path.join(d, "f*s.png")), key=lambda p: t_of(os.path.basename(p)))
    kimg = cv2.imread(kf)
    kimg = cv2.cvtColor(kimg, cv2.COLOR_BGR2RGB) if kimg is not None else None
    kL, kR = detect_lr(kimg) if kimg is not None else (None, None)
    print(f"关键帧脸数: 左={'有' if kL and kL[1] is not None else '缺'} 右={'有' if kR and kR[1] is not None else '缺'}")
    prevL = prevR = None
    print(f"\n{'frame':>9}  | {'左眼vsKF':>8} {'左眼逐帧':>8} | {'右眼vsKF':>8} {'右眼逐帧':>8}")
    print("-" * 62)
    for p in frames:
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        L, R = detect_lr(img)
        name = os.path.basename(p)
        l_kf = cos(L[1], kL[1]) if (L and L[1] is not None and kL and kL[1] is not None) else float("nan")
        r_kf = cos(R[1], kR[1]) if (R and R[1] is not None and kR and kR[1] is not None) else float("nan")
        l_seq = cos(L[1], prevL) if (L and L[1] is not None and prevL is not None) else float("nan")
        r_seq = cos(R[1], prevR) if (R and R[1] is not None and prevR is not None) else float("nan")
        print(f"{name:>9}  | {l_kf:>8.3f} {l_seq:>8.3f} | {r_kf:>8.3f} {r_seq:>8.3f}")
        if L and L[1] is not None:
            prevL = L[1]
        if R and R[1] is not None:
            prevR = R[1]


if __name__ == "__main__":
    main()
