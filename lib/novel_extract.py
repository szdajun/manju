"""lib/novel_extract.py — 小说章节 → 实体提取 (P1, 2026-08-15).

单轮 LLM 提取: 章节文本 → {subject, characters, scenes}.
characters → characters_spec 形状 (喂 lib.scriptgen.gen_script):
  {KEY: {"name": name, "brief": "外观, 性格"}} — brief 让 scriptgen 展开成英文指纹.
"""
import json

from .scriptgen import parse_llm_json

_EXTRACT_PROMPT = """你是短剧编剧。从下面的小说章节里提取短剧剧本需要的实体, 只输出 JSON, 不要其它文字。

JSON 格式:
{{
  "subject": "本集一句话梗概 (中文, 供剧本 LLM 当题材)",
  "characters": {{
    "KEY1": {{"name": "中文名", "appearance": "外观特征(中文, 具体到服装颜色款式/发型/身材)", "personality": "性格特征(中文)"}},
    "KEY2": {{...}}
  }},
  "scenes": ["场景1(中文)", "场景2(中文)"]
}}

要求:
- characters 只放本章实际出现的 2-4 个主要角色; KEY 用大写英文标识 (如 LINXIA)
- appearance 要具体到可画的程度 (服装颜色+款式, 发型, 体型), 这是后续生成角色立绘的依据
- subject 一句话概括本章主要情节和冲突, 供生成单集剧本

章节文本:
---
{chapter}
---
只输出 JSON:"""


def _default_ask(prompt, *, max_tokens, temperature):
    from .llm import ask_llm  # 可插拔 LLM (MANJU_LLM 切换 minimax/doubao)
    return ask_llm(prompt, max_tokens=max_tokens, temperature=temperature)


def extract_entities(chapter_text: str, ask=None, max_tokens=2500,
                     temperature=0.4) -> dict:
    """章节 → {subject, characters, scenes}.

    characters 已是 characters_spec 形状 ({KEY: {name, brief}}), 直接喂 gen_script.
    ask: 注入可测 (默认用豆包 ask_doubao).
    """
    chapter = (chapter_text or "").strip()
    if not chapter:
        return {"subject": "", "characters": {}, "scenes": []}
    # 章节超长先截 (控制 token)
    if len(chapter) > 6000:
        chapter = chapter[:6000] + "……(截断)"

    prompt = _EXTRACT_PROMPT.format(chapter=chapter)
    caller = ask or _default_ask
    text = caller(prompt, max_tokens=max_tokens, temperature=temperature)
    obj = parse_llm_json(text)

    subject = str(obj.get("subject", "")).strip()
    scenes = obj.get("scenes") or []
    if isinstance(scenes, str):
        scenes = [scenes]
    scenes = [str(s).strip() for s in scenes if str(s).strip()]

    # characters → characters_spec
    characters = {}
    for key, ch in (obj.get("characters") or {}).items():
        if not isinstance(ch, dict):
            continue
        name = ch.get("name") or key
        appearance = str(ch.get("appearance", "")).strip()
        personality = str(ch.get("personality", "")).strip()
        brief = ", ".join(p for p in (appearance, personality) if p)
        characters[str(key).upper()] = {"name": name, "brief": brief or name}

    return {"subject": subject, "characters": characters, "scenes": scenes}


def characters_to_json(characters_spec: dict) -> str:
    """characters_spec → JSON 字符串 (CLI 打印用)."""
    return json.dumps(characters_spec, ensure_ascii=False, indent=2)
