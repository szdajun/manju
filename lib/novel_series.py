"""lib/novel_series.py — 长篇小说 → 多集规划 (P1-1, 2026-08-15).

章节分组 + 系列角色锁定:
  - group_into_episodes: 章节 → 若干集 (N 组均等 或 auto 按字数)
  - extract_series_characters: 前 1-2 章 → 全系列固定角色 characters_spec (跨集复用, 定妆只生成一次)
"""
import json

from .novel_extract import _default_ask
from .scriptgen import parse_llm_json

# 每集目标字数 (auto 分组用)
DEFAULT_CHARS_PER_EP = 4000
DEFAULT_MAX_CHAPTERS = 3

_SERIES_CHAR_PROMPT = """你是短剧编剧。下面是某部小说的开头章节。请提取**贯穿全剧的主要角色** (2-4 个),
只输出 JSON, 不要其它文字。

JSON 格式:
{{
  "characters": {{
    "KEY1": {{"name": "中文名", "appearance": "外观特征(中文, 具体到服装颜色款式/发型/身材)", "personality": "性格特征(中文)"}},
    "KEY2": {{...}}
  }}
}}

要求:
- 只放全剧反复出现的主要角色; KEY 用大写英文标识
- appearance 具体到可画程度 (服装颜色+款式, 发型, 体型), 这是后续生成定妆图的依据
- 跨集一致性靠它: 每集剧本都复用这份角色设定

小说开头章节:
---
{chapters}
---
只输出 JSON:"""


def group_into_episodes(chapters, episodes_n=None, max_chapters=DEFAULT_MAX_CHAPTERS,
                        chars_per_ep=DEFAULT_CHARS_PER_EP):
    """章节 → 若干集.

    episodes_n 给定 → 分成 N 个大致均等组.
    否则 → 按累计字数 (~chars_per_ep) + 最多 max_chapters/集 分组.
    返回 [{title, chapters: [{title, text}]}].
    """
    chapters = list(chapters or [])
    if not chapters:
        return []
    if episodes_n is not None and episodes_n > 0:
        # N 组均等 (按章节数切分)
        n = min(episodes_n, len(chapters))
        groups = [[] for _ in range(n)]
        for i, ch in enumerate(chapters):
            groups[i * n // len(chapters)].append(ch)
    else:
        # auto: 按字数 + 最多 max_chapters 分组
        groups, cur, size = [], [], 0
        for ch in chapters:
            if cur and (len(cur) >= max_chapters or size + len(ch["text"]) > chars_per_ep):
                groups.append(cur)
                cur, size = [], 0
            cur.append(ch)
            size += len(ch["text"])
        if cur:
            groups.append(cur)
    return [_episode(g) for g in groups if g]


def _episode(chapter_group):
    """章节组 → 集 dict (标题 = 首章标题)."""
    return {"title": chapter_group[0]["title"], "chapters": chapter_group}


def extract_series_characters(chapters, ask=None, max_tokens=1500, temperature=0.4):
    """前 1-2 章 → 系列角色 characters_spec ({KEY: {name, brief}}).

    brief = "appearance, personality" 让 scriptgen 展开成英文指纹 (跨集锁定).
    ask 注入可测 (默认 lib.llm.ask_llm / MiniMax).
    """
    sample = "\n\n".join(c["text"] for c in list(chapters)[:2]) or ""
    if not sample:
        return {}
    prompt = _SERIES_CHAR_PROMPT.format(chapters=sample[:6000] if len(sample) > 6000 else sample)
    caller = ask or _default_ask
    text = caller(prompt, max_tokens=max_tokens, temperature=temperature)
    obj = parse_llm_json(text)
    characters = {}
    for key, ch in (obj.get("characters") or {}).items():
        if not isinstance(ch, dict):
            continue
        name = ch.get("name") or key
        appearance = str(ch.get("appearance", "")).strip()
        personality = str(ch.get("personality", "")).strip()
        brief = ", ".join(p for p in (appearance, personality) if p)
        characters[str(key).upper()] = {"name": name, "brief": brief or name}
    return characters


def characters_json(characters_spec: dict) -> str:
    return json.dumps(characters_spec, ensure_ascii=False, indent=2)
