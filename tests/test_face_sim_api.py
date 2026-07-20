import numpy as np

from scripts.face_sim import cosine


def test_cosine_orthogonal_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine(a, b)) < 1e-9


def test_cosine_identical_is_one():
    a = np.array([0.6, 0.8])
    assert abs(cosine(a, a) - 1.0) < 1e-6
