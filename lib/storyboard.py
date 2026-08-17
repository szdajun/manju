"""lib/storyboard.py — 分镜数据抽象层 (orchestrator 唯一消费格式).

motion: 让 orchestrator 与剧本来源解耦 — 底层可以是 Python 模块 (ep03 现状)
或 JSON (Phase 2 LLM 产物). 统一输出"规范化 dict" (见 schema).

背景 (2026-08-06, manju Phase 1 一键出片):
  用户拍板真人短剧管线, Phase 1 先做 orchestrator. 剧本将来 LLM 生成 JSON
  (Phase 2), 所以 load_script 自动探测: output/ep{ep}/script.json →
  scripts/ep{ep}.json → lib.ep{ep}_shots (Python 模块). 探测到 JSON 即消费,
  orchestrator 零改动.

权威: 本文件是"分镜数据"的唯一解析入口. schema 即本文件的规范化 dict
结构 (Phase 2 LLM 只需按此结构产出 JSON). 改结构同步改 validate + tests.
"""
from __future__ import annotations

import importlib
import json
import os

MANJU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 规范化 dict 必填键 (顶层)
_REQUIRED_TOP = {"ep", "title", "style", "neg", "characters", "shots"}
# 每角色必填键
_REQUIRED_CHAR = {"fingerprint", "voice_id"}
# 每镜必填键
_REQUIRED_SHOT = {"shot", "line", "motion", "speaker"}

# 镜头台词类型 (可选字段, 缺省时 line_type() 向后兼容推断)
LINE_TYPE_DIALOG = "dialog"        # 角色面向镜头说话 → H3 驱动口型
LINE_TYPE_VOICEOVER = "voiceover"  # 画外音解说 → H3 环境音 + 后期叠旁白
LINE_TYPE_AMBIENT = "ambient"      # 无台词 → H3 纯环境音
LINE_TYPES = {LINE_TYPE_DIALOG, LINE_TYPE_VOICEOVER, LINE_TYPE_AMBIENT}


def normalize_ep(ep: str | int) -> str:
    """规范化集数: "3"/3 → "03"."""
    ep_str = str(ep)
    if ep_str.isdigit():
        return ep_str.zfill(2)
    return ep_str


def _getattr(module, name, default=None):
    return getattr(module, name, default)


def module_to_dict(module) -> dict:
    """lib.epXX_shots Python 模块 → 规范化 dict.

    映射 (见 ep03_shots.py 约定):
      CHAR → characters[主角].fingerprint
      REAL_STYLE → style; REAL_NEG → neg; EYE_FIX → eye_fix (可选)
      SHOTS[N] → shots[N].shot; LINES[N] → shots[N].line
      I2V_MOTION[N] → shots[N].motion; SPEAKER[N] → shots[N].speaker
      TITLE → title; FACE_REF/BODY_REF → characters[主角].face_ref/body_ref
      DURATION_S → duration_s; VOICE_ID → characters[主角].voice_id

    多角色 (ep02): 扫 SPEAKER 收集所有角色名, 逐一建 character 条目;
    非主角角色以模块同名常量 (如 TINGTING) 作 fingerprint, 缺则只留 voice_id.
    """
    speakers = _getattr(module, "SPEAKER", {}) or {}
    shot_ids = sorted(set(speakers.keys()) | set((_getattr(module, "SHOTS", {}) or {}).keys()))
    main_char = speakers.get(min(shot_ids), "NARRATOR") if shot_ids else "NARRATOR"

    characters = {}
    for spk in sorted(set(speakers.values())):
        if spk == main_char:
            fingerprint = _getattr(module, "CHAR", "")
            face_ref = _getattr(module, "FACE_REF", None)
            body_ref = _getattr(module, "BODY_REF", None)
            voice_id = _getattr(module, "VOICE_ID", None)
        else:
            # 其他角色: 模块同名常量作指纹 (如 TINGTING), 缺则空
            fingerprint = _getattr(module, spk, None) or ""
            face_ref = _getattr(module, f"{spk}_FACE_REF", None)
            body_ref = _getattr(module, f"{spk}_BODY_REF", None)
            voice_id = _getattr(module, f"{spk}_VOICE", None)
        characters[spk] = {
            "fingerprint": fingerprint,
            "face_ref": face_ref,
            "body_ref": body_ref,
            "voice_id": voice_id,
            "lora": None,
        }

    shots = {}
    line_types = _getattr(module, "LINE_TYPE", {}) or {}
    for s in shot_ids:
        shots[str(s)] = {
            "shot": (_getattr(module, "SHOTS", {}) or {}).get(s, ""),
            "line": (_getattr(module, "LINES", {}) or {}).get(s, ""),
            "motion": (_getattr(module, "I2V_MOTION", {}) or {}).get(s, ""),
            "speaker": speakers.get(s, main_char),
            "line_type": line_types.get(s),
        }

    script = {
        "ep": normalize_ep(_getattr(module, "EP", "XX")),
        "title": _getattr(module, "TITLE", ""),
        "duration_s": _getattr(module, "DURATION_S", 5),
        "style": _getattr(module, "REAL_STYLE", ""),
        "neg": _getattr(module, "REAL_NEG", ""),
        "eye_fix": _getattr(module, "EYE_FIX", None),
        "characters": characters,
        "shots": shots,
    }
    return script


def validate(script: dict) -> None:
    """校验规范化 dict 必填结构, 不合法抛 ValueError."""
    missing = _REQUIRED_TOP - set(script.keys())
    if missing:
        raise ValueError(f"剧本缺顶层键: {missing}")
    if not script["characters"]:
        raise ValueError("剧本 characters 为空 (需至少一个角色)")
    if not script["shots"]:
        raise ValueError("剧本 shots 为空 (需至少一个镜头)")
    for spk, ch in script["characters"].items():
        m = _REQUIRED_CHAR - set(ch.keys())
        if m:
            raise ValueError(f"角色 {spk} 缺键: {m}")
    for sid, sh in script["shots"].items():
        m = _REQUIRED_SHOT - set(sh.keys())
        if m:
            raise ValueError(f"镜 {sid} 缺键: {m}")


def load_json_script(path: str) -> dict:
    """读 JSON 剧本文件 → 校验 → 规范化 dict."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON 剧本必须是 dict: {path}")
    # 规范化 shots key 为 str
    if isinstance(data.get("shots"), dict):
        data["shots"] = {str(k): v for k, v in data["shots"].items()}
    validate(data)
    return data


def dump_json(script: dict, path: str) -> None:
    """规范化 dict → JSON (Phase 2 LLM 落点 + roundtrip 测试)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(script, f, ensure_ascii=False, indent=2)


def _module_name_for_ep(ep: str) -> str:
    return f"lib.ep{ep}_shots"


def load_script(ep: str | int, *, source: str | None = None) -> dict:
    """读取一集分镜数据, 返回规范化 dict (orchestrator 唯一消费格式).

    Args:
        ep: "03" 或 3, 规范化成两位 "03".
        source: 显式来源. 以 ".json" 结尾 → 按 JSON 读; 否则当模块名
            (如 source="ep05_shots" → import lib.ep05_shots).
            None → 自动探测:
              1) output/ep{ep}/script.json 存在 → JSON (Phase 2 LLM 产物落点)
              2) scripts/ep{ep}.json 存在     → JSON (人工校验版, 入 git)
              3) 否则 import lib.ep{ep}_shots (Phase 1 ep03 现状)

    Returns:
        规范化 dict (validate 通过).
    Raises:
        FileNotFoundError: 探测不到来源 / JSON 缺失.
        ValueError: schema 不合法.
    """
    ep_norm = normalize_ep(ep)

    if source:
        if source.endswith(".json"):
            return load_json_script(source)
        # 模块名: 允许 "ep05_shots" 或 "lib.ep05_shots"
        mod_name = source if source.startswith("lib.") else f"lib.{source}"
        return module_to_dict(importlib.import_module(mod_name))

    # 自动探测
    candidates = [
        os.path.join(MANJU, "output", f"ep{ep_norm}", "script.json"),
        os.path.join(MANJU, "scripts", f"ep{ep_norm}.json"),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return load_json_script(cand)

    # 回退到 Python 模块
    try:
        mod = importlib.import_module(_module_name_for_ep(ep_norm))
    except ModuleNotFoundError:
        raise FileNotFoundError(
            f"找不到 ep{ep_norm} 的分镜数据: 无 JSON ({candidates}) 也无 "
            f"{_module_name_for_ep(ep_norm)} 模块")
    script = module_to_dict(mod)
    script["ep"] = ep_norm
    validate(script)
    return script


# ---- 剧本数据访问器 (Phase 3 通用化: 阶段脚本按每镜说话人取字段) ----

_INNER_MONOLOGUE_RE = None  # lazy compile (见 clean_line)


def clean_line(line: str) -> str:
    """台词清洗: 剥"（内心：...）"标注壳, 只留独白内容 (TTS 念出+字幕双防线).

    2026-08-18 ep22 镜3 教训: LLM 违规把 voiceover 台词写成 "（内心：说好的不喘气呢！）",
    脏 line 直通 TTS (念出"内心") 和字幕 (显示括注)。规范在 scriptgen prompt 里
    早有, 但纯靠 LLM 自觉 — 此函数是程序化兜底, 喂 TTS/字幕前过一遍。
    兼容全角/半角括号、全角/半角冒号; 普通台词 (无标注) 原样返回。
    """
    import re
    text = str(line or "").strip()
    m = re.fullmatch(r"[（(]\s*内心\s*[:：]\s*(.+?)[）)]", text)
    return m.group(1).strip() if m else text


def shot(script: dict, sid: int | str) -> dict:
    """规范化镜头访问: sid 传 str 或 int 均可."""
    return script["shots"][str(sid)]


def speaker(script: dict, sid: int | str) -> str:
    """sid 对应镜的说话人角色 key."""
    return shot(script, sid)["speaker"]


def character(script: dict, sid: int | str) -> dict:
    """sid 对应镜的说话人角色条目 (fingerprint/face_ref/body_ref/voice_id/lora)."""
    return script["characters"][speaker(script, sid)]


def frame_chars(script: dict, sid: int | str) -> list[str] | None:
    """该镜画面里的角色列表 (混合方案 2026-08-07 新增).

    - 显式标注 shot dict 的 frame_chars → 用它 (控制单人/双人取景).
    - 缺省 None → 默认: 剧本多角色 = 双人同框, 单角色 = 单人.
    """
    fc = shot(script, sid).get("frame_chars")
    if fc:
        return [str(c) for c in fc]
    if len(script["characters"]) >= 2:
        return list(script["characters"].keys())
    return None


def line_type(script: dict, sid: int | str) -> str:
    """镜的台词类型: dialog/voiceover/ambient.

    显式标注 (shot dict 里 line_type) 优先; 缺省向后兼容推断:
      有非空 line → dialog (角色说台词, 默认历史行为)
      无 line → ambient (纯环境音, 无 TTS)
    voiceover 只能由剧本显式标注 (画外音是创作意图, 无法从数据推断).

    Returns:
        "dialog" | "voiceover" | "ambient"
    """
    lt = shot(script, sid).get("line_type")
    if lt in LINE_TYPES:
        return lt
    return LINE_TYPE_DIALOG if (shot(script, sid).get("line") or "").strip() \
        else LINE_TYPE_AMBIENT
