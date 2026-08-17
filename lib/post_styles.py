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


# ---- BGM 调性预设 (Music3 caption, 英文 prompt) ----
# pump = music3.DEFAULT_BGM_CAPTION 原文迁入 (import 复用, 不复制粘贴)
def _bgm_presets() -> dict[str, dict]:
    from lib import music3
    return {
        "pump": {  # 燃/训练 (现默认)
            "caption": music3.DEFAULT_BGM_CAPTION, "volume": 0.30,
        },
        "funny": {  # 滑稽轻快 (打工牛马喜剧向)
            "caption": (
                "Global Metadata: Playful comedic background for a funny "
                "office-worker fitness short, 118 BPM, G major, bouncy "
                "pizzicato strings, quirky woodwind stabs, slide whistle "
                "accents, light shuffle rhythm, cheerful and clumsy "
                "mood.\n\n"
                "Audio Prompt: pizzicato strings, comedic woodwinds, light "
                "percussion, shuffle groove, playful and bouncy, no vocals"),
            "volume": 0.25,
        },
        "warm": {   # 温情
            "caption": (
                "Global Metadata: Warm emotional underscore for an "
                "inspirational fitness journey short, 90 BPM, C major, soft "
                "piano arpeggios, gentle strings pad, subtle acoustic "
                "guitar, hopeful and heartwarming.\n\n"
                "Audio Prompt: soft piano, warm strings, acoustic guitar, "
                "slow build, emotional and hopeful, no vocals"),
            "volume": 0.22,
        },
        "chill": {  # 丧系自嘲
            "caption": (
                "Global Metadata: Laid-back lo-fi track for a self-deprecating "
                "office-worker comedy short, 82 BPM, A minor, mellow Rhodes "
                "chords, dusty vinyl drums, lazy bassline, tired but cozy "
                "vibe.\n\n"
                "Audio Prompt: lo-fi rhodes, vinyl crackle, laid-back drums, "
                "mellow bass, chill and slightly melancholic, no vocals"),
            "volume": 0.22,
        },
    }


BGM_PRESETS: dict[str, dict] = _bgm_presets()


def resolve_bgm(key: str) -> dict:
    """取 BGM 预设。未知 key → ValueError 带合法 key 列表."""
    if key not in BGM_PRESETS:
        raise ValueError(f"未知 BGM 预设 {key!r}, 可选: {sorted(BGM_PRESETS)}")
    return BGM_PRESETS[key]


def bgm_cache_path(preset: str, seed: int, root: str) -> str:
    """BGM 产物缓存路径: root/{preset}_{md5(caption)[:8]}.flac. 同 caption+seed 免生成."""
    import hashlib
    import os
    h = hashlib.md5(BGM_PRESETS[preset]["caption"].encode()).hexdigest()[:8]
    return os.path.join(root, f"bgm_{preset}_{h}.flac")


# ---- 组装选项三级优先合并 (make_ep03_real CLI 接线, Task 5) ----
DEFAULT_POST = {"sub_style": "classic", "transition_mode": "fade",
                "bgm_preset": None, "cover": False, "cover_shot": None}


def resolve_post_opts(script: dict, **cli) -> dict:
    """三级优先合并: 剧本 post 字段 > CLI > 默认。非法 key ValueError.

    返回 {sub_style, transition_mode, bgm_preset, cover, cover_shot}。
    (CLI 值 None = 未传, 让位剧本/默认; post 值 None 同理视为未设。)
    """
    post = dict(script.get("post") or {})
    merged = dict(DEFAULT_POST)
    merged.update({k: v for k, v in cli.items() if v is not None and k in DEFAULT_POST})
    merged.update({k: v for k, v in post.items()
                   if k in DEFAULT_POST and v is not None})
    resolve_style(merged["sub_style"])          # 校验, 未知 key ValueError
    if merged["transition_mode"] not in ("auto", "fade"):
        raise ValueError(f"未知 transition_mode {merged['transition_mode']!r}")
    if merged["bgm_preset"] is not None:
        resolve_bgm(merged["bgm_preset"])
    return merged


# ---- 封面渲染器 ----
COVER_W, COVER_H = 1080, 1920


def render_cover(script: dict, kf_dir: str, out: str,
                 shot_id: int | None = None, series_tag: str = "打工牛马·健身篇") -> str:
    """封面: 关键帧居中裁 9:16 + 底部渐变黑 + title 大字 + 系列角标 + 集数标.

    关键帧缺 → FileNotFoundError (调用方 warn 跳过, 不阻塞成片)。
    """
    import glob
    import os

    from PIL import Image, ImageDraw, ImageFont

    ep = str(script.get("ep", "")).zfill(2)
    sid = 1 if shot_id is None else int(shot_id)
    cands = sorted(glob.glob(os.path.join(
        kf_dir, f"ep{ep}_shot{sid:02d}_*.png")))
    if not cands:
        raise FileNotFoundError(
            f"封面关键帧缺: ep{ep}_shot{sid:02d}_*.png in {kf_dir}")
    kf = Image.open(cands[-1]).convert("RGB")

    # 居中裁 9:16 → 覆盖画布
    tw, th = COVER_W, COVER_H
    scale = max(tw / kf.width, th / kf.height)
    kf = kf.resize((int(kf.width * scale) + 1, int(kf.height * scale) + 1))
    x0 = (kf.width - tw) // 2
    y0 = (kf.height - th) // 2
    img = kf.crop((x0, y0, x0 + tw, y0 + th)).convert("RGB")

    d = ImageDraw.Draw(img, "RGBA")
    # 底部 35% 线性渐变黑 (title 可读性)
    gh = int(th * 0.35)
    for i in range(gh):
        a = int(220 * i / gh)
        d.line([(0, th - gh + i), (tw, th - gh + i)], fill=(0, 0, 0, a))

    font_path = "C:/Windows/Fonts/msyhbd.ttc"  # = make_ep01.FONT
    f_tag = ImageFont.truetype(font_path, 32)
    f_ep = ImageFont.truetype(font_path, 44)

    # title 大字: 底部 18% 处, 白字黑描边 (超宽缩到 84px)
    title = str(script.get("title") or f"EP.{ep}")
    size = 96
    f_t = ImageFont.truetype(font_path, size)
    while d.textlength(title, font=f_t) > tw - 120 and size > 60:
        size -= 8
        f_t = ImageFont.truetype(font_path, size)
    tw_text = d.textlength(title, font=f_t)
    d.text(((tw - tw_text) / 2, th - int(th * 0.18)), title,
           font=f_t, fill=(255, 255, 255, 255),
           stroke_width=6, stroke_fill=(0, 0, 0, 255))

    # 左上系列角标: 半透明黑底条 + 白字
    tag_w = d.textlength(series_tag, font=f_tag)
    d.rounded_rectangle([24, 40, 24 + tag_w + 36, 40 + 56], radius=12,
                        fill=(0, 0, 0, 140))
    d.text((42, 52), series_tag, font=f_tag, fill=(255, 255, 255, 230))

    # 右下集数标
    ep_s = f"EP.{ep.lstrip('0') or ep}"
    ep_w = d.textlength(ep_s, font=f_ep)
    d.text((tw - ep_w - 40, th - 130), ep_s, font=f_ep,
           fill=(255, 218, 70, 255), stroke_width=4,
           stroke_fill=(0, 0, 0, 255))

    img.save(out)
    return out
