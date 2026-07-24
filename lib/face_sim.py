"""face_sim wrapper. Used by T3+ tests + CP1.

Bridges scripts/_face_sim_check.py (real buffalo_l detector from insightface)
to a simple pair API: pair_face_sim(img_path, ref_path) -> float [0, 1].

Returns 0.0 on detection failure (does not raise). Mirrors the CJK-safe
cv2.imdecode read pattern from scripts/_face_sim_check.py.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None  # type: ignore

_APP = None
_REF_EMB_CACHE: dict[str, np.ndarray] = {}


def _cv2_imread(path: str) -> np.ndarray | None:
    """cv2.imread fails on Windows CJK paths — read bytes + imdecode."""
    try:
        with open(path, "rb") as f:
            buf = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _get_app():
    """Lazy-init the buffalo_l FaceAnalysis singleton."""
    global _APP
    if _APP is None:
        if FaceAnalysis is None:
            return None
        try:
            _APP = FaceAnalysis(name="buffalo_l")
            _APP.prepare(ctx_id=-1, det_size=(640, 640))
        except Exception:
            _APP = None
    return _APP


def _face_embedding(bgr: np.ndarray) -> np.ndarray | None:
    app = _get_app()
    if app is None or bgr is None:
        return None
    try:
        faces = app.get(bgr)
    except Exception:
        return None
    if not faces:
        return None
    face = max(faces, key=lambda f: f.det_score)
    emb = getattr(face, "normed_embedding", None)
    if emb is None:
        return None
    return np.asarray(emb, dtype=np.float32)


def _get_ref_embedding(ref_path: str) -> np.ndarray | None:
    if ref_path in _REF_EMB_CACHE:
        return _REF_EMB_CACHE[ref_path]
    bgr = _cv2_imread(ref_path)
    emb = _face_embedding(bgr)
    if emb is not None:
        _REF_EMB_CACHE[ref_path] = emb
    return emb


def pair_face_sim(img_path: str, ref_path: str) -> float:
    """Cosine similarity [0, 1] between dominant face in img_path and ref_path.

    Returns 0.0 on any detection failure. buffalol embeddings are L2-normalized,
    so dot product == cosine similarity in [-1, 1]; clipped to [0, 1].
    """
    ref_emb = _get_ref_embedding(ref_path)
    if ref_emb is None:
        return 0.0
    bgr = _cv2_imread(img_path)
    img_emb = _face_embedding(bgr)
    if img_emb is None:
        return 0.0
    cos = float(np.dot(ref_emb, img_emb))
    if cos < 0.0:
        return 0.0
    if cos > 1.0:
        return 1.0
    return cos