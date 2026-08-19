"""lib/series_runner.py — 多集批量编排核心 (循环复用 orchestrator, 失败隔离)."""
from __future__ import annotations

import re

_TOKEN = re.compile(r"^(\d+)(?:-(\d+))?$")


def parse_eps(spec: str) -> list[str]:
    """'20-25' / '20,21' / '20-22,25' → 补零排序去重的 ep 列表. 非法 token ValueError."""
    if not spec.strip():
        raise ValueError("eps 不能为空")
    out: set[int] = set()
    for tok in spec.split(","):
        tok = tok.strip()
        m = _TOKEN.match(tok)
        if not m:
            raise ValueError(f"eps token 非法: {tok!r} (期望 N 或 N-M, 逗号分隔)")
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        if b < a:
            raise ValueError(f"eps 区间倒置: {tok!r}")
        out.update(range(a, b + 1))
    return [str(e).zfill(2) for e in sorted(out)]
