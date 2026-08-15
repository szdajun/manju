"""lib/novel 章节拆分单测 (P1)."""
import pytest

from lib import novel


def test_split_chinese_chapters():
    t = ("前言。\n\n第一章 初入健身房\n她推开门。\n\n"
         "第二章 遇见教练\n肌肉男走来。")
    chs = novel.split_chapters(t)
    assert [c["title"] for c in chs] == ["全文", "初入健身房", "遇见教练"]
    assert "她推开门" in chs[1]["text"]


def test_split_english_chapters():
    chs = novel.split_chapters("Chapter 1 Intro\nhello.\n\nChapter 2 More\nworld.")
    assert [c["title"] for c in chs] == ["Intro", "More"]


def test_split_no_marker_fallback():
    t = ("第一段" + "很长文字" * 500 + "\n\n" + "第二段" + "长文字" * 500)
    chs = novel.split_chapters(t)
    # fallback 应切出多块
    assert len(chs) >= 2
    assert all(c["title"].startswith("第") for c in chs)


def test_split_empty():
    assert novel.split_chapters("") == []
    assert novel.split_chapters(None) == []


def test_split_long_chapter_truncated():
    # 单章超长 → 截断到 max_chars
    t = "第一章 很长\n" + "段落文字内容" * 2000
    chs = novel.split_chapters(t, max_chars=3000)
    assert len(chs) == 1
    assert len(chs[0]["text"]) <= 3000


def test_read_novel_utf8(tmp_path):
    p = tmp_path / "n.txt"
    p.write_text("第一章 测试\n内容", encoding="utf-8")
    assert "第一章" in novel.read_novel(str(p))


def test_read_novel_missing():
    with pytest.raises(ValueError):
        novel.read_novel("Z:/nonexistent/novel.txt")
