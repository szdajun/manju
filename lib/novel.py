"""lib/novel.py — 小说读取 + 章节拆分 (P1, 纯 stdlib 零依赖, 2026-08-15).

支持纯 TXT (UTF-8/GBK/BOM 兜底). docx/epub 需后续加依赖.
"""
import os
import re

# 章节标题: 第X章/回/节 + 可选标题文字, 或英文 Chapter N
_CHAPTER_RE = re.compile(
    r"^(?:第\s*[一二三四五六七八九十百千0-9０-９]+\s*[章节回卷集篇]|Chapter\s+\d+)(.*)$",
    re.IGNORECASE)

# fallback 块大小 (无章节标记时按空行切 ~2000 字)
_FALLBACK_CHARS = 2000


def read_novel(path: str) -> str:
    """读小说文本, 依次尝试 utf-8 / utf-8-sig / gbk. 失败抛 ValueError."""
    if not os.path.exists(path):
        raise ValueError(f"小说文件不存在: {path}")
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法解码小说 (试了 utf-8/gbk): {path}")


def split_chapters(text: str, max_chars: int = 12000) -> list[dict]:
    """拆章节 → [{title, text}].

    优先识别 `第X章/回/节` / `Chapter N` 标题行; 无任何标记 → 按空行/字数 fallback 切块.
    max_chars: 单章超长时按段截断 (控制后续 LLM token).
    """
    text = (text or "").strip()
    if not text:
        return []
    lines = text.split("\n")
    chapters, cur_title, cur = [], None, []
    found_marker = False
    for ln in lines:
        m = _CHAPTER_RE.match(ln.strip())
        if m:
            found_marker = True
            if cur_title is not None and cur:
                chapters.append({"title": cur_title, "text": _join(cur)})
            # 标题 = 标记行剩余部分 (如 "初入健身房"); 无 → 用标记本身
            rest = (m.group(1) or "").strip()
            cur_title = rest if rest else ln.strip()
            cur = []
        else:
            if cur_title is None:
                cur_title = "全文"  # 首个标记前的内容
            cur.append(ln)

    if cur_title is not None and cur:
        chapters.append({"title": cur_title, "text": _join(cur)})

    # fallback: 完全没有章节标记 → 按空行分块 ~2000 字
    if not found_marker:
        chapters = _fallback_split(text)
    # 超长章截断 (保段落完整)
    for ch in chapters:
        if len(ch["text"]) > max_chars:
            ch["text"] = _truncate_paragraphs(ch["text"], max_chars)
    return chapters


def _join(paragraphs: list[str]) -> str:
    """段落 join: 空行保留为段落分隔 (LLM 可读)."""
    return "\n".join(p for p in paragraphs if p.strip())


def _fallback_split(text: str) -> list[dict]:
    """无章节标记 → 按空行段落累积 ~2000 字切块."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        return [{"title": "第1部分", "text": text}]
    chapters, cur, size = [], [], 0
    idx = 0
    for p in paras:
        cur.append(p)
        size += len(p)
        if size >= _FALLBACK_CHARS:
            idx += 1
            chapters.append({"title": f"第{idx}部分", "text": "\n\n".join(cur)})
            cur, size = [], 0
    if cur:
        idx += 1
        chapters.append({"title": f"第{idx}部分", "text": "\n\n".join(cur)})
    return chapters


def _truncate_paragraphs(text: str, max_chars: int) -> str:
    """按段落截断到 ≤max_chars (保段落完整, 留尾部给 LLM 收束)."""
    if len(text) <= max_chars:
        return text
    kept, size = [], 0
    for p in re.split(r"\n\s*\n", text):
        if size + len(p) > max_chars:
            break
        kept.append(p)
        size += len(p) + 2
    # 保留至少一段 (即使第一段就超长)
    if not kept and text:
        kept = [text[:max_chars]]
    return "\n\n".join(kept)
