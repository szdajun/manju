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
    assert got == ["circleopen", "slideup", "fadeblack"]


def test_pick_transitions_dialog_chain_and_fallback():
    # dialog→dialog 快切; 未知组合兜底 fade; 末镜前 fadeblack
    got = pick_transitions(_mini_script(["dialog", "dialog", "ambient"]))
    assert got == ["wipeleft", "fadeblack"]
    got2 = pick_transitions(_mini_script(["ambient", "ambient", "ambient"]))
    assert got2 == ["fade", "fadeblack"]


def test_pick_transitions_single_shot():
    assert pick_transitions(_mini_script(["dialog"])) == []


def test_transitions_all_valid_ffmpeg():
    """白名单全部是 ffmpeg xfade 合法名 (blackfade 是未实现的错误名, 教训 2026-08-17)."""
    valid = {"fade", "wipeleft", "wiperight", "slideup", "slidedown",
             "circleopen", "circleclose", "fadeblack", "fadewhite", "dissolve"}
    assert set(TRANSITIONS) <= valid


# ---- BGM 调性预设测试 ----

from lib.post_styles import BGM_PRESETS, resolve_bgm, bgm_cache_path  # noqa: E402


def test_bgm_presets_schema():
    assert set(BGM_PRESETS) == {"pump", "funny", "warm", "chill"}
    for key, p in BGM_PRESETS.items():
        assert isinstance(p["caption"], str) and len(p["caption"]) > 50, key
        assert 0.0 < p["volume"] <= 0.5, key


def test_pump_caption_is_current_default():
    """pump = 现 music3.DEFAULT_BGM_CAPTION 原文迁入 (零破坏)."""
    from lib import music3
    assert BGM_PRESETS["pump"]["caption"] == music3.DEFAULT_BGM_CAPTION


def test_resolve_bgm_unknown():
    with pytest.raises(ValueError, match="pump"):
        resolve_bgm("epic")


def test_bgm_cache_path_stable(tmp_path):
    a = bgm_cache_path("funny", 42, str(tmp_path))
    b = bgm_cache_path("funny", 42, str(tmp_path))
    assert a == b and a.endswith(".flac") and "funny" in a
    assert bgm_cache_path("warm", 42, str(tmp_path)) != a   # preset 进路径


# ---- 封面渲染器测试 ----

from lib.post_styles import render_cover  # noqa: E402


def _make_kf(kf_dir, name, size=(768, 1344)):
    from PIL import Image
    Image.new("RGB", size, (200, 120, 90)).save(os.path.join(kf_dir, name))


def test_render_cover_makes_1080x1920(tmp_path):
    kf_dir = tmp_path / "kf"
    kf_dir.mkdir()
    _make_kf(str(kf_dir), "ep22_shot01_seed42.png")
    script = {"ep": "22", "title": "自律的开始"}
    out = render_cover(script, str(kf_dir), str(tmp_path / "cover.png"))
    assert os.path.exists(out)
    assert Image.open(out).size == (1080, 1920)


def test_render_cover_uses_specified_shot(tmp_path):
    kf_dir = tmp_path / "kf"
    kf_dir.mkdir()
    _make_kf(str(kf_dir), "ep22_shot03_seed7.png")
    script = {"ep": "22", "title": "自律的开始"}
    render_cover(script, str(kf_dir), str(tmp_path / "c.png"), shot_id=3)
    assert (tmp_path / "c.png").exists()


def test_render_cover_missing_kf_raises(tmp_path):
    script = {"ep": "22", "title": "自律的开始"}
    with pytest.raises(FileNotFoundError):
        render_cover(script, str(tmp_path / "nope"), str(tmp_path / "c.png"))


def test_render_cover_gradient_not_solid_black(tmp_path):
    """渐变区不是实心黑 (RGBA 画布不混 alpha 的回归, 2026-08-17 教训)."""
    from PIL import Image
    kf_dir = tmp_path / "kf"
    kf_dir.mkdir()
    Image.new("RGB", (768, 1344), (255, 0, 0)).save(
        str(kf_dir / "ep22_shot01_seed1.png"))
    script = {"ep": "22", "title": "自律的开始"}
    out = str(tmp_path / "c.png")
    render_cover(script, str(kf_dir), out)
    img = Image.open(out).convert("RGB")
    # 渐变起点附近 (y≈1255): 纯红源混少量黑, 必须仍是显著红色而非黑
    px = img.getpixel((540, 1255))
    assert px[0] > 120 and px[2] < 80, px
    # 渐变中段 y≈1600: 深红但非纯黑
    px2 = img.getpixel((540, 1600))
    assert px2 != (0, 0, 0) and px2[0] > 0, px2
