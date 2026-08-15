"""lib/novel_series 分集规划单测."""
from lib import novel_series as ns


def _chapters(n):
    return [{"title": f"第{i}章", "text": "内容" * 100} for i in range(1, n + 1)]


def test_group_n_episodes_even():
    eps = ns.group_into_episodes(_chapters(6), episodes_n=3)
    assert len(eps) == 3
    assert [len(e["chapters"]) for e in eps] == [2, 2, 2]
    assert eps[0]["title"] == "第1章"


def test_group_auto_by_chars():
    # 每章 100 字, 4000/集 → 40 章才够一组; 用 max_chapters 限制
    eps = ns.group_into_episodes(_chapters(5), max_chapters=2)
    assert len(eps) == 3
    assert [len(e["chapters"]) for e in eps] == [2, 2, 1]


def test_group_empty():
    assert ns.group_into_episodes([]) == []


def test_extract_series_characters_parses():
    fake = lambda prompt, max_tokens=1500, temperature=0.4: '''{"characters": {
        "LINXIA": {"name": "林夏", "appearance": "黑色高马尾, 雾蓝运动上衣", "personality": "元气自律"},
        "COACH": {"name": "李教练", "appearance": "肌肉型, 黑背心", "personality": "专业"}}}'''
    chars = ns.extract_series_characters(_chapters(2), ask=fake)
    assert set(chars) == {"LINXIA", "COACH"}
    assert chars["LINXIA"]["name"] == "林夏"
    assert "雾蓝运动上衣" in chars["LINXIA"]["brief"]


def test_extract_series_characters_empty():
    assert ns.extract_series_characters([], ask=lambda *a, **k: "{}") == {}


def test_episode_title_from_first_chapter():
    eps = ns.group_into_episodes([{"title": "开场", "text": "x"}, {"title": "转折", "text": "y"}], episodes_n=2)
    assert eps[0]["title"] == "开场"
    assert eps[1]["title"] == "转折"
