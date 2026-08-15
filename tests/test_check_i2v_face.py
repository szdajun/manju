"""check_i2v_face 跨镜一致性纯计算单测 (manju venv, 无需 insightface).

check_i2v_face.py 的 insightface 是懒加载 (`_make_app` 内), 模块级只有 cv2/numpy/storyboard,
manju venv 可跑 `cross_shot_consistency`.
"""
import numpy as np

from scripts import check_i2v_face as cf


def _unit_vec(d, seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=d)
    return v / np.linalg.norm(v)


def test_cross_shot_consistency_similar_faces():
    """5 个相近嵌入 → 每镜跨镜均值都高 (>0.7)."""
    d = 512
    base = _unit_vec(d, 1)
    embs = {}
    for i in range(5):
        # 轻微扰动 (夹角 ~10°)
        v = base + 0.1 * _unit_vec(d, 100 + i)
        embs[i + 1] = v / np.linalg.norm(v)
    consis = cf.cross_shot_consistency(embs)
    assert all(v > 0.7 for v in consis.values())
    # 自一致: 和自身 sim=1
    k = list(embs)[0]
    assert abs(np.dot(embs[k], embs[k]) - 1.0) < 1e-6


def test_cross_shot_consistency_flags_outlier():
    """1 个完全不同的嵌入 → 它被标为低跨镜 (<0.3), 其它正常."""
    d = 512
    a = _unit_vec(d, 1)
    b = _unit_vec(d, 2)   # 正交 → sim≈0
    embs = {1: a, 2: a, 3: a, 4: b}
    consis = cf.cross_shot_consistency(embs)
    # 3 个正常 (跨镜均值被离群拉低到 0.667) 仍 > 阈值 0.45
    assert all(consis[i] > 0.45 for i in (1, 2, 3))
    assert consis[4] < 0.45   # 漂移镜 (正交 ~0)


def test_cross_shot_single_shot_returns_1():
    """单镜时无"其它"→ 返 1.0 (不崩)."""
    assert cf.cross_shot_consistency({1: _unit_vec(512, 1)})[1] == 1.0


def test_drift_threshold_contract():
    """阈值语义: 跨镜 < 阈值 → DRIFT."""
    embs = {1: _unit_vec(512, 1), 2: _unit_vec(512, 2)}  # 正交, sim~0
    consis = cf.cross_shot_consistency(embs)
    assert consis[1] < 0.45 and consis[2] < 0.45
