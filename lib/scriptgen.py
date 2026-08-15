"""lib/scriptgen.py — Phase 2 剧本 LLM 引擎 (manju, 2026-08-06).

从「题材 + 角色设定」→ 豆包 LLM → 规范化分镜 JSON → output/epXX/script.json.
orchestrator 经 lib.storyboard.load_script 自动探测消费 (零改动).

用法 (CLI):
    python scripts/gen_script.py --ep 04 --title "..." --subject "..." [--dry-run]

关键设计 (Plan agent 2026-08-06 定):
  - ask_doubao 只支持单轮 user prompt → 任务+角色+few-shot+格式规范全塞一个字符串
  - LLM 只输出子集 title/duration_s/characters/shots; style/neg/eye_fix/ep 由 merge_script 注入
  - 角色指纹命门: 已知角色(带 fingerprint)逐字复用不得改服装; 新角色(只有 brief)让 LLM 展开
  - few-shot: 嵌入 ep03 shot1 (最通用钩子镜, 含完整 3 句服装不变式)
  - 鲁棒解析: 去 ```json 围栏 → first{/last} → 尾逗号清洗 → json.loads → raw_decode 降级
"""
from __future__ import annotations

import json
import os
import re
import sys

from lib.ep03_shots import EYE_FIX, REAL_NEG, REAL_STYLE  # 复用写实风格块 (无循环依赖)
from lib.storyboard import dump_json, normalize_ep, validate

MANJU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 风格块: LLM 不需要输出 style/neg, merge_script 从这里注入
STYLES = {
    "real": {
        "style": REAL_STYLE,
        "neg": REAL_NEG,
        "eye_fix": EYE_FIX,
    },
    # cartoon 是占位 (下游关键帧/i2v 是写实路径, 未验证, 慎用)
    "cartoon": {
        "style": ("anime style, cel shading, clean lineart, vibrant colors, "
                  "digital illustration, high detail"),
        "neg": ("photorealistic, photograph, 3d render, realistic skin, blurry, "
                "lowres, extra fingers"),
        "eye_fix": None,
    },
}

# 可选声线白名单 (voice_id 只从这里选, 否则 merge 回退默认)
# ⚠ 2026-08-07 修: "male-qingnian" 不是有效 MiniMax 声线 (TTS 2054 voice id not exist),
#   正确男声 = "male-qn-qingse" (lib/tts.py 默认).
VOICE_OPTIONS = [
    "female-shaonv",    # 少女活泼 (林夏 ep03 已验证)
    "female-chengshu",  # 成熟女声
    "female-wenrou",    # 温柔
    "male-qn-qingse",   # 青年男声 (MiniMax 有效 ID)
]
VOICE_DEFAULT = "female-shaonv"


# ---- few-shot (从 scripts/ep03.json 读, 避免手抄漂移) ----

def _load_few_shot_shots(*shot_ids: int) -> dict:
    """读 scripts/ep03.json 的角色指纹 + 指定 shot, 作 prompt few-shot 示例.

    只取 fingerprint + shots, 不带 name/face_ref/body_ref/lora (贴近 LLM 输出子集).
    """
    path = os.path.join(MANJU, "scripts", "ep03.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = {"characters": {}, "shots": {}}
    for k, ch in data.get("characters", {}).items():
        out["characters"][k] = {"fingerprint": ch["fingerprint"]}
    for sid in shot_ids:
        s = str(sid)
        if s in data.get("shots", {}):
            out["shots"][s] = data["shots"][s]
    return out


# ---- prompt ----

_PROMPT_TEMPLATE = """你是 AI 真人短剧的分镜编剧。根据给定的题材与角色设定，产出一集结构化分镜剧本。你的输出必须是一个严格 JSON 对象。

【你的任务】
- 题材/本集内容: {subject}
- 集标题(可润色或沿用): {title}
- 镜头数: {n_shots}

【画风】(由管线注入, 你不要输出 style/neg 字段; shot 场景描述要贴合此写实画风)
style: {style}
neg: {neg}

【输出格式】
只输出一个 JSON 对象。不要输出 markdown 代码块围栏(```)，不要输出任何解释文字。第一个字符必须是 {{，最后一个字符必须是 }}。

{{
  "title": "中文集标题(8-14字, 一句话点出本集幽默点或反转)",
  "duration_s": 5,
  "characters": {{
    "角色KEY": {{
      "fingerprint": "英文角色指纹(规范见下)",
      "voice_id": "从可选列表选一个"
    }}
  }},
  "shots": {{
    "1": {{
      "shot": "英文关键帧静态场景 prompt",
      "line": "中文台词",
      "motion": "英文 i2v 运动描述",
      "speaker": "角色KEY",
      "line_type": "dialog"
    }},
    ... 共 {n_shots} 镜 ...
  }}
}}

【字段规范】

1. title: 中文, 8-14 字, 幽默/生活化, 概括本集核心冲突或反转。

2. characters: 每个角色一条
   - 角色KEY 用大写拉丁字母(如 LINXIA)。
   - fingerprint 是英文角色指纹(喂给生图/视频模型), 必须:
     a) 以 "1girl, solo, " 或 "1boy, solo, " 开头;
     b) 含姓名: 拉丁名(拼音) + 括号中文名, 如 "Linxia (林夏)";
     c) 含年龄体型, 如 "a slim-fit 26-30 year-old Chinese woman";
     d) 含发型, 如 "black hair in a high ponytail";
     e) 含完整服装: 上衣(颜色+款式+可辨识标志, 如 "misty-blue long-sleeve athletic top with a small white arc logo on the left chest") + 下装(颜色+款式+细节) + 鞋;
     f) 含妆容/气质/表情, 如 "natural makeup, bright energetic eyes, friendly confident expression";
     g) 含场景标签, 如 "modern athletic wear, gym setting"。
   服装是角色一致性的命门: 颜色/款式/标志必须具体, 且每个 shot 都要重复引用。
   - voice_id 从 ["female-shaonv"(少女活泼), "female-chengshu"(成熟), "female-wenrou"(温柔), "male-qn-qingse"(青年男)] 选一个。

3. shots: 恰好 {n_shots} 个镜头, key 从 "1" 到 "{n_shots}"。
   叙事弧: 钩子(开场抓眼) → 起势(展开) → 转折(小波折) → 反转(幽默反差) → 收尾(金句/约定)。
   每镜四字段:
   - shot (英文关键帧 prompt, 描述性静态场景), 必须:
     a) 以景别开头, **优先全身/大半身 (性感健身内容要秀身材, 2026-08-07 用户拍板)**: full body shot / medium full shot / three-quarter body shot 为主, 少用 medium close-up, 尽量不用 close-up (近景看不到身材);
     b) 含场景(如 "modern gym interior with treadmill in the background");
     c) 含角色名 + 完整服装指纹(每镜重复, 不得省略);
     d) 含角色当前动作姿态 + 表情;
     e) 含灯光(如 "bright overhead gym lighting") + 构图(如 "vertical composition") + 情绪 mood(如 "modern fitness atmosphere")。
   - line (中文台词): 12-25 字, 短句, 口语化, 幽默/生活化, 与镜内容强关联, 适合配音。
   - motion (英文 i2v 运动描述): 描述单一低幅度动作(i2v 模型做不了大动作), 必须:
     a) 主语 + 动作 + 幅度(如 "subtle step motion");
     b) 含表情变化(如 "satisfied smirk gradually forming");
     c) 句尾必带 3 句服装不变式, 颜色/标志必须与 fingerprint 完全一致:
        "the {{上衣颜色}} top stays {{颜色}} throughout, the {{裤色}} pants stay {{颜色}} throughout, the {{标志}} stays in the same position on the left chest"
      不变式示例: "the misty-blue top stays misty-blue throughout, the black pants stay black throughout, the white arc logo stays in the same position on the left chest"。
   - speaker: 该镜说话的角色KEY。
   - line_type (可选, 三选一, 缺省= "dialog"):
       "dialog"    角色面向镜头说台词, 唇型由台词音频驱动 (默认, 有 line 的镜都是);
       "voiceover" 画外音解说, 角色闭嘴不开口, 后期叠旁白 (旁白镜, 仅此值需显式标注);
       "ambient"   无台词纯环境音 (舞蹈/健身等动作镜, 此时 line 留空字符串)。
   创作提示: 想让角色"开口说台词"→ dialog; 想让镜头有"解说感"但角色不说话 → voiceover。

【角色设定(用户给定, 必须遵守)】
{character_instructions}

【格式示例(参考已上线 ep03《装备先到位》, 严格模仿其结构; 这是唯一格式参考)】
{few_shot}
"""


def _character_instructions(characters_spec: dict | None) -> str:
    """从角色 spec 生成"角色设定"段落指令."""
    lines = []
    for key, ch in (characters_spec or {}).items():
        name = ch.get("name") or key
        if ch.get("fingerprint"):
            lines.append(
                f"- 【{key} {name}】已定稿指纹(必须逐字复用, 不得改服装/发型/气质): "
                f"{ch['fingerprint']}")
        else:
            brief = ch.get("brief") or "未给, 请按题材自拟"
            lines.append(
                f"- 【{key} {name}】人设: {brief} → 请你按上面 fingerprint 规范"
                f"把它展开成完整英文指纹(服装必须具体到颜色+款式+标志)")
    if not lines:
        lines.append("- 无额外角色设定, 请按题材自拟 1-2 个主要角色。")
    return "\n".join(lines)


def build_prompt(
    subject: str,
    title: str,
    n_shots: int,
    style: str,
    neg: str,
    characters_spec: dict | None = None,
    few_shot_shots: tuple[int, ...] = (1,),
) -> str:
    """构造完整单轮 prompt (任务+画风+格式规范+角色+few-shot)."""
    few = _load_few_shot_shots(*few_shot_shots)
    return _PROMPT_TEMPLATE.format(
        subject=subject,
        title=title,
        n_shots=n_shots,
        style=style,
        neg=neg,
        character_instructions=_character_instructions(characters_spec),
        few_shot=json.dumps(few, ensure_ascii=False, indent=2),
    )


# ---- LLM 输出解析 (鲁棒) ----

def parse_llm_json(text: str) -> dict:
    """鲁棒解析 LLM 输出 → dict. 失败抛 ValueError.

    覆盖: markdown 围栏包裹 / 纯 JSON / 前后杂文 / 尾逗号 / 截断.
    """
    # 1) 去 ```json ... ``` / ``` ... ``` 围栏
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    # 2) 找第一个 { 到最后一个 }
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"LLM 输出中找不到 JSON 对象: {text[:200]!r}")
    text = text[start:end + 1]
    # 3) 去尾逗号 (json.loads 拒绝), 然后标准解析
    text = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # 4) 退化: 逐 { 位置 raw_decode, 容错前缀杂物/截断
        dec = json.JSONDecoder()
        obj = None
        for i in range(len(text)):
            if text[i] == "{":
                try:
                    obj, _ = dec.raw_decode(text[i:])
                    break
                except json.JSONDecodeError:
                    continue
        if obj is None:
            raise ValueError(f"LLM JSON 解析失败: {text[:300]!r}")
    if not isinstance(obj, dict):
        raise ValueError(f"LLM JSON 不是对象: {type(obj).__name__}")
    return obj


# ---- merge (LLM 子集 → 规范化 dict) ----

def _canonical_key(k: str) -> str:
    """角色 key 归一化: 大写拉丁字母 (LINXIA / linxia → LINXIA)."""
    s = "".join(ch for ch in str(k).strip().upper() if ch.isalnum())
    return s if s else str(k).strip()


def merge_script(
    parsed: dict,
    *,
    ep: str,
    style: str,
    neg: str,
    eye_fix: str | None = None,
    duration_s: int = 5,
    char_overrides: dict | None = None,      # {KEY: {face_ref?, body_ref?, voice_id?}}
    expected_n_shots: int | None = None,
    voice_default: str = VOICE_DEFAULT,
) -> dict:
    """LLM 子集 → 规范化 dict (补 ep/style/neg/eye_fix/char 默认), validate 通过."""
    ep_norm = normalize_ep(ep)

    # 1. characters: key 归一化 + 补默认
    raw_chars = parsed.get("characters", {}) or {}
    if not raw_chars:
        raise ValueError("LLM 未产出 characters")
    characters = {}
    key_map = {}
    for old_key, ch in raw_chars.items():
        new_key = _canonical_key(old_key)
        key_map[old_key] = new_key
        fingerprint = str(ch.get("fingerprint") or "").strip()
        if not fingerprint:
            raise ValueError(f"角色 {old_key} 缺 fingerprint")
        overrides = (char_overrides or {}).get(new_key, {}) or {}
        characters[new_key] = {
            "fingerprint": fingerprint,
            "face_ref": overrides.get("face_ref"),
            "body_ref": overrides.get("body_ref"),
            "voice_id": overrides.get("voice_id") or str(ch.get("voice_id") or voice_default),
            "lora": None,
        }

    # 2. shots: 排序 + speaker 重映射
    raw_shots = parsed.get("shots", {}) or {}
    if not raw_shots:
        raise ValueError("LLM 未产出 shots")
    first_char = next(iter(characters))
    shots = {}
    for sid in sorted(raw_shots.keys(), key=lambda x: int(str(x)) if str(x).isdigit() else 0):
        s = raw_shots[sid]
        spk = _canonical_key(s.get("speaker") or first_char)
        if spk not in characters:
            spk = first_char
        lt = str(s.get("line_type") or "").strip().lower()
        shots[str(sid)] = {
            "shot": str(s.get("shot") or ""),
            "line": str(s.get("line") or ""),
            "motion": str(s.get("motion") or ""),
            "speaker": spk,
            "line_type": lt or None,
        }

    if expected_n_shots is not None and len(shots) != expected_n_shots:
        raise ValueError(
            f"LLM 产出 {len(shots)} 镜, 期望 {expected_n_shots} 镜. "
            f"考虑提高 max_tokens 或重试")

    script = {
        "ep": ep_norm,
        "title": str(parsed.get("title") or f"第{ep_norm}集"),
        "duration_s": int(parsed.get("duration_s") or duration_s),
        "style": style,
        "neg": neg,
        "eye_fix": eye_fix,
        "characters": characters,
        "shots": shots,
    }
    validate(script)
    return script


# ---- 主入口 ----

def _default_ask(prompt: str, *, max_tokens: int, temperature: float) -> str:
    """默认走可插拔 LLM (lib.llm, MANJU_LLM 切换 minimax/doubao). 惰性 import 让测试可注入."""
    from .llm import ask_llm
    return ask_llm(prompt, max_tokens=max_tokens, temperature=temperature)


def gen_script(
    subject: str,
    title: str,
    n_shots: int = 5,
    style: str = "real",
    ep: str = "04",
    characters_spec: dict | None = None,
    *,
    temperature: float = 0.4,
    max_tokens: int = 3000,
    dry_run: bool = False,
    out_dir: str | None = None,
    ask: callable | None = None,
) -> dict:
    """主入口: 题材+角色 → 豆包 → 规范化剧本 → output/epXX/script.json.

    dry_run=True → 返回 {"dry_run", "prompt", "path"}, 不调 API 不写文件.
    返回规范化 script dict (已 dump).
    """
    if style not in STYLES:
        raise KeyError(f"未知风格 {style!r}, 可选 {list(STYLES)}")
    sb = STYLES[style]

    prompt = build_prompt(
        subject, title, n_shots,
        sb["style"], sb["neg"], characters_spec,
    )
    out_path = os.path.join(
        out_dir or os.path.join(MANJU, "output", f"ep{normalize_ep(ep)}"),
        "script.json")

    if dry_run:
        return {"dry_run": True, "prompt": prompt, "path": out_path}

    text = (ask or _default_ask)(prompt, max_tokens=max_tokens, temperature=temperature)
    parsed = parse_llm_json(text)

    # char_overrides: 从 characters_spec 补 face_ref/body_ref/voice_id
    char_overrides = {}
    for k, v in (characters_spec or {}).items():
        kk = _canonical_key(k)
        ov = {f: v[f] for f in ("face_ref", "body_ref", "voice_id") if v.get(f)}
        if ov:
            char_overrides[kk] = ov

    script = merge_script(
        parsed, ep=ep, style=sb["style"], neg=sb["neg"],
        eye_fix=sb.get("eye_fix"),
        char_overrides=char_overrides, expected_n_shots=n_shots,
    )
    dump_json(script, out_path)
    return script
