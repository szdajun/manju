"""tests/test_post_styles.py — 后处理风格库 (2026-08-17 spec)."""
import os
import sys

import pytest
from PIL import Image

MANJU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MANJU)
sys.path.insert(0, os.path.join(MANJU, "scripts"))

from lib.post_styles import (  # noqa: E402
    SUBTITLE_STYLES, resolve_style, TRANSITIONS, pick_transitions)


def test_subtitle_styles_schema():
    """4 套预设, 每套必填字段齐全, 值类型正确."""
    assert set(SUBTITLE_STYLES) == {"classic", "clean", "variety", "minimal"}
    for key, st in SUBTITLE_STYLES.items():
        assert isinstance(st["font_size"], int), key
        assert len(st["fill"]) == 3, key                    # RGB
        assert len(st["stroke"]) in (3, 4), key             # RGB/RGBA
        assert isinstance(st["stroke_w"], int), key
        assert st["box"] is None or len(st["box"]) == 4, key  # RGBA 或无底条
        assert isinstance(st["radius"], int), key


def test_classic_is_current_behavior():
    """classic = 现黄字黑边硬编码值 (默认零破坏)."""
    st = resolve_style("classic")
    assert st["fill"] == (255, 218, 70)
    assert st["stroke"] == (0, 0, 0, 255)
    assert st["stroke_w"] == 3
    assert st["box"] == (0, 0, 0, 165)
    assert st["radius"] == 14


def test_resolve_style_unknown_raises():
    with pytest.raises(ValueError, match="classic"):
        resolve_style("neon")


def test_render_subtitle_styles(tmp_path):
    """4 风格各出 PNG: 720x1280, 有台词 → 非全透明; 无台词 → 全透明."""
    import make_ep01 as m
    for key in SUBTITLE_STYLES:
        out = str(tmp_path / f"sub_{key}.png")
        m.render_subtitle("第一组哑铃，就从现在开始！", out, style=key)
        img = Image.open(out)
        assert img.size == (720, 1280)
        alphas = img.getchannel("A").getextrema()
        assert alphas[1] > 0, key            # 有字/有底条
    # 无台词: 透明占位
    out0 = str(tmp_path / "sub_empty.png")
    m.render_subtitle("", out0, style="variety")
    assert Image.open(out0).getchannel("A").getextrema()[1] == 0


def test_render_subtitle_unknown_style(tmp_path):
    import make_ep01 as m
    with pytest.raises(ValueError, match="classic"):
        m.render_subtitle("x", str(tmp_path / "s.png"), style="neon")


def _mini_script(pairs):
    """pairs = [(line_type),...] → 最小剧本 dict (pick_transitions 只看 line_type)."""
    return {"shots": {str(i + 1): {"line_type": lt, "line": "x"}
                      for i, lt in enumerate(pairs)}}


def test_transitions_table_covers_rules():
    """规则表引用的转场名都在 ffmpeg xfade 白名单表里."""
    from lib.post_styles import TRANSITION_RULES
    for tr in TRANSITION_RULES.values():
        assert tr in TRANSITIONS


def test_pick_transitions_rules():
    d_vo_d = _mini_script(["dialog", "voiceover", "dialog", "ambient"])
    got = pick_transitions(d_vo_d)
    assert got == ["circleopen", "slideup", "blackfade"]


def test_pick_transitions_dialog_chain_and_fallback():
    # dialog→dialog 快切; 未知组合兜底 fade; 末镜前 blackfade
    got = pick_transitions(_mini_script(["dialog", "dialog", "ambient"]))
    assert got == ["wipeleft", "blackfade"]
    got2 = pick_transitions(_mini_script(["ambient", "ambient", "ambient"]))
    assert got2 == ["fade", "blackfade"]


def test_pick_transitions_single_shot():
    assert pick_transitions(_mini_script(["dialog"])) == []
