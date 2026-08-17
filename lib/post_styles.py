"""lib/post_styles.py — 后处理风格库 (2026-08-17): 字幕/转场/BGM/封面单一真相源.

纯数据 + 纯函数 + 封面渲染器。make_ep01 渲染函数按 style key 取参,
默认 classic = 现黄字黑边行为 (零破坏)。
"""
from __future__ import annotations

# ---- 字幕预设 (只差色/描边/底条; 字体/换行/Y位不动) ----
SUBTITLE_STYLES: dict[str, dict] = {
    "classic": {  # 现黄字版 (默认)
        "font_size": 42, "fill": (255, 218, 70),
        "stroke": (0, 0, 0, 255), "stroke_w": 3,
        "box": (0, 0, 0, 165), "radius": 14,
    },
    "clean": {    # 白字高对比
        "font_size": 44, "fill": (255, 255, 255),
        "stroke": (0, 0, 0, 220), "stroke_w": 4,
        "box": (0, 0, 0, 190), "radius": 18,
    },
    "variety": {  # 综艺花字
        "font_size": 46, "fill": (255, 255, 255),
        "stroke": (255, 90, 30, 255), "stroke_w": 6,
        "box": (255, 90, 30, 200), "radius": 22,
    },
    "minimal": {  # 裸字细描边
        "font_size": 40, "fill": (255, 255, 255),
        "stroke": (0, 0, 0, 160), "stroke_w": 2,
        "box": None, "radius": 0,
    },
}


def resolve_style(key: str) -> dict:
    """取字幕预设。未知 key → ValueError 带合法 key 列表."""
    if key not in SUBTITLE_STYLES:
        raise ValueError(
            f"未知字幕风格 {key!r}, 可选: {sorted(SUBTITLE_STYLES)}")
    return SUBTITLE_STYLES[key]


# ---- 转场 (ffmpeg xfade 内置类型白名单 + 节奏规则) ----
TRANSITIONS: dict[str, str] = {
    "fade": "交叉溶解 (通用)",
    "wipeleft": "左擦快切 (叙事推进)",
    "slideup": "上滑 (回现实)",
    "circleopen": "圆形展开 (进内心)",
    "fadeblack": "黑场 (收尾/章节)",
}

# (上镜 line_type, 下镜 line_type) → 转场; 未命中兜底 fade
TRANSITION_RULES: dict[tuple[str, str], str] = {
    ("dialog", "dialog"): "wipeleft",
    ("dialog", "voiceover"): "circleopen",
    ("voiceover", "dialog"): "slideup",
}


def pick_transitions(script: dict) -> list[str]:
    """按镜头 line_type 查节奏规则 → 每衔接一个转场 (len = 镜数-1).

    末镜前固定 fadeblack (收尾感), 其余未命中规则兜底 fade。
    """
    ids = sorted(script["shots"].keys(), key=int)
    lts = []
    for sid in ids:
        lt = (script["shots"][sid] or {}).get("line_type") or "dialog"
        lts.append(lt)
    out = []
    for i in range(len(lts) - 1):
        if i == len(lts) - 2:                    # 末镜前
            out.append("fadeblack")
        else:
            out.append(TRANSITION_RULES.get((lts[i], lts[i + 1]), "fade"))
    return out
