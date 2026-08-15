"""tests/test_storyboard.py — 分镜数据抽象层 (manju Phase 1, 2026-08-06).

验证 load_script 双来源 (Python 模块 / JSON) 输出同一规范化 dict,
+ schema 校验. 不碰 ComfyUI/Seedance/网络.
"""
import json

import pytest

from lib.storyboard import (
    dump_json, load_json_script, load_script, module_to_dict, validate,
)


def test_load_script_module_ep03():
    """从 lib.ep03_shots 模块读取, 返回规范化 dict."""
    script = load_script("03")
    assert script["ep"] == "03"
    assert set(script["shots"].keys()) == {"1", "2", "3", "4", "5"}
    assert script["shots"]["1"]["speaker"] == "LINXIA"
    # 角色资产 (orchestrator 透传用)
    assert "LINXIA" in script["characters"]
    assert script["characters"]["LINXIA"]["face_ref"].endswith(".png")
    assert script["characters"]["LINXIA"]["body_ref"].endswith(".png")
    assert script["characters"]["LINXIA"]["voice_id"] == "female-shaonv"
    # 样式/负词非空
    assert script["style"]
    assert script["neg"]
    # 每镜有完整字段
    for sh in script["shots"].values():
        assert sh["shot"] and sh["motion"]


def test_json_roundtrip():
    """dump_json → load_json_script 读回 == 原 dict (JSON/模块路径等价)."""
    script = load_script("03")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = f"{d}/script.json"
        dump_json(script, p)
        back = load_json_script(p)
    assert back == script


def test_load_script_explicit_json(tmp_path):
    """load_script(source=json路径) 显式走 JSON."""
    script = load_script("03")
    p = tmp_path / "x.json"
    dump_json(script, str(p))
    back = load_script("03", source=str(p))
    assert back == script


def test_load_script_source_module_name():
    """load_script(source="ep03_shots") 按模块名加载."""
    script = load_script("03", source="ep03_shots")
    assert set(script["shots"].keys()) == {"1", "2", "3", "4", "5"}


def test_missing_ep_raises():
    """不存在的 ep → FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_script("999")


def test_validate_rejects_bad_schema():
    """缺 shots → ValueError."""
    with pytest.raises(ValueError):
        validate({"ep": "03", "title": "x", "style": "", "neg": "",
                  "characters": {"A": {"fingerprint": "x", "voice_id": "y"}}})


def test_validate_empty_shots_rejects():
    """shots 为空 → ValueError."""
    bad = {
        "ep": "03", "title": "x", "style": "", "neg": "",
        "characters": {"A": {"fingerprint": "x", "voice_id": "y"}},
        "shots": {},
    }
    with pytest.raises(ValueError):
        validate(bad)


def test_module_to_dict_ep03():
    """module_to_dict 直接测: 映射 CHAR→characters/SHOTS→shots 等."""
    import lib.ep03_shots as mod
    script = module_to_dict(mod)
    assert script["characters"]["LINXIA"]["fingerprint"].startswith("1girl")
    assert script["shots"]["1"]["shot"]
    assert script["duration_s"] == 5


# ---- Phase 3: 剧本数据访问器 (shot/speaker/character) ----

def test_shot_helpers():
    """shot/speaker/character 访问器 (Phase 3 通用化: 阶段脚本按镜取字段)."""
    from lib.storyboard import character, shot, speaker
    script = load_script("03")
    assert shot(script, 1) == script["shots"]["1"]
    assert shot(script, "1") == script["shots"]["1"]
    assert speaker(script, 1) == "LINXIA"
    assert character(script, 1) == script["characters"]["LINXIA"]
    assert character(script, 1)["voice_id"] == "female-shaonv"
    assert character(script, 1)["fingerprint"].startswith("1girl")


# ---- line_type (镜头台词类型, 2026-08-06 端到端整合) ----

def test_line_type_backward_compat_dialog():
    """旧剧本无 line_type → 有 line 推断 dialog (向后兼容)."""
    from lib.storyboard import LINE_TYPE_DIALOG, line_type
    script = load_script("03")   # ep03 每镜有 line, 无 line_type 字段
    assert line_type(script, 1) == LINE_TYPE_DIALOG


def test_line_type_explicit():
    """显式 line_type 优先于推断."""
    from lib.storyboard import (LINE_TYPE_AMBIENT, LINE_TYPE_DIALOG,
                                LINE_TYPE_VOICEOVER, line_type)
    script = load_script("03")
    script["shots"]["1"]["line_type"] = "voiceover"
    assert line_type(script, 1) == LINE_TYPE_VOICEOVER
    script["shots"]["2"]["line_type"] = "ambient"
    assert line_type(script, 2) == LINE_TYPE_AMBIENT
    # 非法值回落推断
    script["shots"]["3"]["line_type"] = "bogus"
    assert line_type(script, 3) == LINE_TYPE_DIALOG


def test_line_type_ambient_when_empty_line():
    """无 line → ambient (纯环境音)."""
    from lib.storyboard import LINE_TYPE_AMBIENT, line_type
    script = load_script("03")
    script["shots"]["1"]["line"] = ""
    assert line_type(script, 1) == LINE_TYPE_AMBIENT


def test_line_type_module_passthrough(tmp_path):
    """module_to_dict 透传模块 LINE_TYPE dict."""
    from lib.storyboard import LINE_TYPE_VOICEOVER, line_type, module_to_dict
    import lib.ep03_shots as mod
    mod.LINE_TYPE = {1: "voiceover"}
    try:
        script = module_to_dict(mod)
        assert line_type(script, 1) == LINE_TYPE_VOICEOVER
    finally:
        delattr(mod, "LINE_TYPE")
