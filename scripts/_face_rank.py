"""临时: insightface buffalo_l 扫 6 张立绘候选, 客观排"脸最干净"
(单脸 + 正脸 + det_score 最高)。vision 不可用时作客观兜底挑脸。用完即删。

排脸准则 = face_sim命门 所需: 一张能被 arcface 稳检出、单脸、正脸、高分
(det_score 高 ≈ 正脸 + 光照好 + 五官清晰) 的图, 才能给 Seedance i2v 输出
帧做 face_sim 比对 (立绘 vs i2v 帧 ≥0.45)。0 脸/多脸直接判废。
"""
import glob
import os
import sys

import cv2
from insightface.app import FaceAnalysis

# argv[1]=候选目录, argv[2]=glob 前缀(如 manju_xiaoya_real_); 默认李辉向后兼容
SRC = sys.argv[1] if len(sys.argv) > 1 else "../manju/output/lihui_real"
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "manju_lihui_real_"
files = sorted(glob.glob(os.path.join(SRC, PREFIX + "*.png")))
assert files, f"no candidates under {SRC} ({PREFIX}*)"

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1, det_size=(640, 640))  # CPU, 稳

rows = []
for f in files:
    img = cv2.imread(f)
    faces = app.get(img)
    n = len(faces)
    if not faces:
        rows.append({"f": f, "n": 0, "score": 0.0, "bbox": None, "g": "?", "age": None})
        continue
    # 单脸优先: 取 det_score 最高(并列取面积大)那张当主脸
    best = max(faces, key=lambda fc: (fc.det_score,
              (fc.bbox[2] - fc.bbox[0]) * (fc.bbox[3] - fc.bbox[1])))
    bx = [int(round(x)) for x in best.bbox]
    g = {1: "M", 0: "F"}.get(getattr(best, "gender", None), "?")
    rows.append({"f": f, "n": n, "score": float(best.det_score),
                 "bbox": bx, "g": g, "age": int(getattr(best, "age", 0) or 0)})

# 排序: 单脸优先, 再 det_score
rows.sort(key=lambda r: (r["n"] == 1, r["score"]), reverse=True)

print(f"{'candidate':<42} {'n':>2} {'det_score':>9} {'bbox':>24} {'g':>2} {'age':>3}")
for r in rows:
    nm = os.path.basename(r["f"])
    bx = str(r["bbox"]) if r["bbox"] else "-"
    print(f"{nm:<42} {r['n']:>2} {r['score']:>9.4f} {bx:>24} {r['g']:>2} {r['age'] or '-':>3}")

top = rows[0]
print(f"\n[top] {os.path.basename(top['f'])}  det={top['score']:.4f}  nfaces={top['n']}  "
      f"gender={top['g']}  age={top['age']}")
if top["n"] != 1:
    print("[warn] top 非 1 脸, 作脸源不理想 — 可能需重抽或调 prompt")
