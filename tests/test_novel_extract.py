"""lib/novel_extract 实体提取单测 (注入 fake ask, 不调真实 API)."""
from lib import novel_extract


def _fake_ask(json_text):
    return lambda prompt, max_tokens=1500, temperature=0.4: json_text


def test_extract_entities_parses():
    fake = _fake_ask('''```json
{"subject": "健身女孩第一天去健身房立誓自律却先拍照",
 "characters": {"LINXIA": {"name": "林夏", "appearance": "黑色高马尾, 雾蓝色运动上衣", "personality": "元气自律"},
                "COACH": {"name": "教练", "appearance": "肌肉型健身教练, 黑色背心", "personality": "严肃专业"}},
 "scenes": ["现代健身房"]}
```''')
    r = novel_extract.extract_entities("章节内容", ask=fake)
    assert r["subject"].startswith("健身女孩")
    assert r["scenes"] == ["现代健身房"]
    # characters → characters_spec 形状
    assert set(r["characters"]) == {"LINXIA", "COACH"}
    assert r["characters"]["LINXIA"]["name"] == "林夏"
    assert "雾蓝色运动上衣" in r["characters"]["LINXIA"]["brief"]
    assert "元气自律" in r["characters"]["LINXIA"]["brief"]


def test_extract_entities_plain_json():
    fake = _fake_ask('{"subject":"S","characters":{"A":{"name":"甲","appearance":"红衣","personality":"活泼"}},"scenes":["家"]}')
    r = novel_extract.extract_entities("x", ask=fake)
    assert r["subject"] == "S"
    assert r["characters"]["A"]["brief"] == "红衣, 活泼"


def test_extract_entities_empty_chapter():
    r = novel_extract.extract_entities("", ask=lambda *a, **k: "{}")
    assert r == {"subject": "", "characters": {}, "scenes": []}


def test_extract_entities_missing_fields_tolerated():
    fake = _fake_ask('{"subject":"S"}')  # 无 characters/scenes
    r = novel_extract.extract_entities("x", ask=fake)
    assert r["characters"] == {}
    assert r["scenes"] == []
