# -*- coding: utf-8 -*-
"""tests/test_assemble_cache.py — 组装段缓存失效规则 (2026-08-20 ep26 镜2 教训).

seg/av 缓存键原只有 sub_style, 不感知 i2v/voice 更新:
i2v 重生成后重组装静默沿用旧画面+旧音轨 (说话人修复被缓存吞掉)。
规则: i2v 比 seg 新 → seg 重建; seg/voice 比 av 新 → av 重 mux。
不跑 ffmpeg/不生成 — 只验证判定式 (与 make_ep03_real 内联条件同构)。
"""
import os

import pytest


def _seg_is_stale(src_mtime: float, seg_mtime: float | None) -> bool:
    """与 make_ep03_real.py seg 缓存判定同构."""
    return seg_mtime is None or src_mtime > seg_mtime


def _av_is_stale(seg_mtime: float, av_mtime: float | None,
                 voice_mtime: float | None = None) -> bool:
    """与 make_ep03_real.py av 缓存判定同构."""
    if av_mtime is None:
        return True
    if seg_mtime > av_mtime:
        return True
    return voice_mtime is not None and voice_mtime > av_mtime


def test_seg_missing_is_stale():
    assert _seg_is_stale(src_mtime=1.0, seg_mtime=None)


def test_seg_fresh_when_newer_than_i2v():
    assert not _seg_is_stale(src_mtime=100.0, seg_mtime=200.0)


def test_seg_stale_when_i2v_regenerated():
    """ep26 镜2 病灶: i2v 12:20 重生成 > seg 09:33 缓存 → 必须重建."""
    assert _seg_is_stale(src_mtime=200.0, seg_mtime=100.0)


def test_av_missing_is_stale():
    assert _av_is_stale(seg_mtime=1.0, av_mtime=None)


def test_av_fresh_when_newest():
    assert not _av_is_stale(seg_mtime=100.0, av_mtime=200.0,
                            voice_mtime=150.0)


def test_av_stale_when_seg_rebuilt():
    """seg 因 i2v 更新重建后 → 旧 av 必须重 mux."""
    assert _av_is_stale(seg_mtime=300.0, av_mtime=200.0)


def test_av_stale_when_voice_regenerated():
    """voice mp3 重新生成 → 旧 av (H3 自合成音轨) 必须重 mux."""
    assert _av_is_stale(seg_mtime=100.0, av_mtime=200.0, voice_mtime=300.0)


def test_mtime_judgement_matches_script():
    """判定式与 make_ep03_real.py 内联条件逐字一致 (防漂移)."""
    src = "scripts/make_ep03_real.py"
    code = open(src, encoding="utf-8").read()
    assert "os.path.getmtime(src) > os.path.getmtime(seg_v)" in code
    assert "os.path.getmtime(seg_v) > os.path.getmtime(dst_av)" in code
    assert ("os.path.getmtime(voice) > os.path.getmtime(dst_av)" in code)
