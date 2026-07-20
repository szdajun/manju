"""manju 命门硬验证: 立绘 vs 生成镜 脸 embedding 余弦相似度。

借 fitness venv 的 insightface buffalo_l arcface(512d)。纯 CPU,不抢 ComfyUI 的 GPU。
中文路径用 imdecode 绕 cv2 坑。

判据(insightface arcface 经验阈, 真人脸训练):
  >=0.50 同一人(脸命门硬过) / 0.30~0.50 模糊 / <0.30 不同人(脸漂)

⚠ caveat: buffalo_l 真人脸训练, 漫画风立绘可能漏检/嵌入不稳 —— 检不出本身也是信息。
"""
import os
import sys

import cv2
import numpy as np


def imread_unicode(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def cosine(ea, eb):
    """两向量的余弦相似度(内部归一化, 输入未归一化也可)。纯 numpy, manju venv 可单测。"""
    ea = np.asarray(ea, dtype=np.float64)
    eb = np.asarray(eb, dtype=np.float64)
    na = np.linalg.norm(ea) + 1e-9
    nb = np.linalg.norm(eb) + 1e-9
    return float(np.dot(ea / na, eb / nb))


def _app():
    """lazy 建 insightface app(借 fitness venv 的 insightface)。纯 CPU, 不抢 ComfyUI GPU。"""
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def primary(app, p):
    """图 p 的最大脸 embedding(归一化)+ 脸数 + top det_score。没检出脸返 (None, n, 0.0)。"""
    img = imread_unicode(p)
    faces = app.get(img)
    if not faces:
        return None, len(faces), 0.0
    faces.sort(
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True,
    )
    f = faces[0]
    emb = f.embedding / (np.linalg.norm(f.embedding) + 1e-9)
    return emb, len(faces), float(f.det_score)


def face_cosine(a, b):
    """两张图最大脸的 arcface 余弦; 任一没检出脸返 None。

    insightface 在 fitness venv; manju venv 缺时 import 处抛 ImportError,
    调用方(fitness venv 跑的脚本)才有 insightface。
    """
    app = _app()
    ea, _na, _sa = primary(app, a)
    eb, _nb, _sb = primary(app, b)
    if ea is None or eb is None:
        return None
    return cosine(ea, eb)


def main(a, b):
    app = _app()
    ea, na, sa = primary(app, a)
    eb, nb, sb = primary(app, b)
    print(f"[{os.path.basename(a)}] faces={na} top_det={sa:.3f}")
    print(f"[{os.path.basename(b)}] faces={nb} top_det={sb:.3f}")
    if ea is None or eb is None:
        print("[FAIL] 至少一张图 buffalo_l 没检出脸 -> 无法客观比对")
        print("       (漫画脸 stylized 可能超出真人脸检测器范围, 见 caveat)")
        return 2

    sim = cosine(ea, eb)
    print(f"\ncosine_similarity = {sim:.4f}")
    if sim >= 0.50:
        verdict = "同一人 (脸命门硬过 [OK])"
    elif sim >= 0.30:
        verdict = "模糊 (气质像, 介于同一人/不同人间 [?])"
    else:
        verdict = "不同人 (脸漂了, 命门未过 [FAIL])"
    print(f"判定: {verdict}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: face_sim.py <图A> <图B>")
        sys.exit(1)
    sys.exit(main(sys.argv[1], sys.argv[2]))
