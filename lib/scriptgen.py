"""lib/scriptgen.py — Phase 2 剧本 LLM 引擎 (manju, 2026-08-06).

从「题材 + 角色设定」→ 豆包 LLM → 规范化分镜 JSON → output/epXX/script.json.
orchestrator 经 lib.storyboard.load_script 自动探测消费 (零改动).

用法 (CLI):
    python scripts/gen_script.py --ep 04 --title "..." --subject "..." [--dry-run]

关键设计 (Plan agent 2026-08-06 定):
  - ask_doubao 只支持单轮 user prompt → 任务+角色+few-shot+格式规范全塞一个字符串
  - LLM 只输出子集 title/duration_s/characters/shots; style/neg/eye_fix/ep 由 merge_script 注入
  - 角色指纹命门: 已知角色(带 fingerprint)逐字复用不得改服装; 新角色(只有 brief)让 LLM 展开
  - few-shot: 嵌入 ep03_niuma shot1 (牛马梗钩子镜, 含完整 3 句服装不变式)
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

# 可选声线白名单 (prompt 让 LLM 只从这里选; merge_script 校验 — 白名单外值 print 告警并回退
# VOICE_DEFAULT, 防非法值直达下游 TTS 报 2054; 2026-08-19 终审跟进落地, 原"原样透传"行为已改)
# ⚠ 2026-08-07 修: "male-qingnian" 不是有效 MiniMax 声线 (TTS 2054 voice id not exist),
#   正确男声 = "male-qn-qingse" (lib/tts.py 默认).
VOICE_OPTIONS = [
    "female-shaonv",    # 少女活泼 (林夏 ep03 已验证)
    "female-chengshu",  # 成熟女声
    "female-wenrou",    # 温柔
    "male-qn-qingse",   # 青年男声 (MiniMax 有效 ID)
]
VOICE_DEFAULT = "female-shaonv"


def _sanitize_voice_id(key: str, value: str | None, default: str) -> str:
    """voice_id 白名单校验: 非法值 print 告警 (含原值/回退值) 并回退 default.

    背景 (2026-08-19 Task3 r3): LLM 偶发拼错 (如 "female-shaonun"), 旧逻辑原样透传,
    直到下游 MiniMax TTS 才报 2054 (voice id not exist), 排障链路长。
    注意: char_overrides (上游代码/配置给定) 与 LLM 输出走同一校验, 口径一致。
    """
    v = str(value or default)
    if v not in VOICE_OPTIONS:
        try:   # Windows GBK 控制台打不出 ⚠ → reconfigure 失败也不炸校验本身
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
        print(f"[scriptgen] ⚠ 角色 {key} voice_id {v!r} 不在白名单 "
              f"{VOICE_OPTIONS}, 回退 {default!r} (防 TTS 2054)")
        return default
    return v


# ---- few-shot (从 scripts/ 素材 json 读, 避免手抄漂移) ----

def _load_few_shot_shots(*shot_ids: int, path: str | None = None) -> dict:
    """读 few-shot 素材 json 的角色指纹 + 指定 shot, 作 prompt few-shot 示例.

    ⚠ 素材 json 仅是 few-shot 素材, 不是可跑剧本 — 缺 style/neg,
    不满足 load_json_script/validate 的完整 schema, 勿喂 orchestrator.

    2026-08-20: 默认切 ep27_duet.json (双人轮转黄金示例 — 织体四律
    轮转/接话/反应镜/训练镜 + 每集 ≤1 双人同框 dialog 镜的完整示范;
    2026-08-21 缓冲空镜废除, 镜3 改训练镜);
    ep03_niuma.json (单人自嘲版) 保留为 path= 显式回退.

    只取 fingerprint + shots, 不带 name/face_ref/body_ref/lora (贴近 LLM 输出子集).
    """
    if path is None:
        path = os.path.join(MANJU, "scripts", "ep27_duet.json")
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

1. title: 中文, 8-14 字, 钩子式 — 制造好奇缺口 (如"加班三年，我第一次准点下班"),
   禁止平铺描述式标题 (如"林夏健身遇猛男教练")。

2. characters: 每个角色一条
   - 角色KEY 用大写拉丁字母(如 LINXIA)。
   - fingerprint 是英文角色指纹(喂给生图/视频模型), 必须:
     a) 以 "1girl, solo, " 或 "1boy, solo, " 开头;
     b) 含姓名: 拉丁名(拼音) + 括号中文名, 如 "Linxia (林夏)";
     c) 含年龄体型, 如 "a slim-fit 26-30 year-old Chinese woman";
     d) 含发型, 如 "black hair in a high ponytail";
     e) 含完整服装: 上衣(颜色+款式+可辨识标志, 如 "misty-blue sports bra with a small white arc logo on the left chest") + 下装(颜色+款式+细节, 如 "black high-waist spandex booty shorts") + 鞋;
     f) 含妆容/气质/表情, 如 "natural makeup, bright energetic eyes, friendly confident expression";
     g) 含场景标签, 如 "modern athletic wear, gym setting"。
   服装是角色一致性的命门: 颜色/款式/标志必须具体, 且每个 shot 都要重复引用。
   - voice_id 从 ["female-shaonv"(少女活泼), "female-chengshu"(成熟), "female-wenrou"(温柔), "male-qn-qingse"(青年男)] 选一个。

3. shots: 恰好 {n_shots} 个镜头, key 从 "1" 到 "{n_shots}"。
   叙事结构 (竖屏短剧完播率规则, 必须遵守):
   - 镜1 (钩子): 冲突前置。开场即反差画面 + 悬念台词, 3 秒内让观众想知道"接下来怎样"。
     禁止: 宣言式开场("我要做XX的人"), 问候式开场, 环境铺陈开场。
     ⚠ 宣言变体也算违规: "这铁我举定了/今天不喘气就算成功/我一定要XX" 都是立 flag, 同样禁止。
     镜1 台词只写"已经发生的困境/反差", 不写"想要达成的目标"。合格思路 (⚠ 只借思路, 严禁照抄任何示例原句):
     a) 窘境反差/数字对比 — 如"体检报告说我二十六岁的腰，六十岁"的句式 (自嘲式数字反差; 与 few-shot 的"时间+发现"构型刻意不同);
     b) 新手困境吐槽 — 如"器械比领导还难搞"的比喻 (健身物+职场角色对比)。
     ⚠ 照抄判定 (对 few-shot 示例句与本题所有示例句都生效): 与任一示例句连续 6 字以上相同即算照抄, 直接不合格。必须用本集剧情重写。
   - 镜2-3 (展开+梗位): 每镜至少一个留人点 — 牛马自嘲梗 / 反差 / 误会 / 吐槽。
     梗从打工生活取材: 加班/周报/领导语录/KPI/工资条/通勤/出租屋/同事/老板画饼。
     健身是壳: 梗必须挂在举铁/跑步机/教练等健身场景上, 不脱离画面。
   - 镜4 (转折): 期望违背, 情绪反转 (惨→爽 / 爽→更惨)。
   - 镜5 (追更钩): 下集悬念, 悬念必须具体 ("明天教练要带我练腿"), 让观众想看下一集。
     禁止: 鸡汤总结, "加油/坚持/明天会更好"式台词。
     每集结尾固定喊本剧 sign-off "牛马，明天见。" (紧跟悬念句后, 形成品牌记忆点), 不得改写。
     ⚠ 例外说明: few-shot 镜5 未含 sign-off, 属于少数省略场合 — 新产剧本镜5 必须带 (sign-off
     是本剧固定收尾词, 不计入"照抄示例"查重线, 固定句不算抄袭)。

   对话织体四律 (2026-08-20 用户拍板: 情节与人物互动太简单, 对话感是第一提升点;
   同日追加第④律: 分镜衔接太仓促, 需缓冲位):
   ① 轮转律: 相邻有台词的镜 speaker 必须不同 (林夏说→教练接→林夏说), 同一角色
     禁止连续 2 镜有台词。双人剧本必须让两个角色都说上话。
   ② 接话律: 后说者的台词必须**直接接前一句的话头** — 引用对方的词、反驳、毒舌
     补刀、顺着梗往下翻 (甲:"器械比领导还难搞" 乙:"领导至少不会让你当场社死")。
     禁止各说各的: 两人台词像两条平行线、去掉谁都不影响另一句 = 不合格。
   ③ 反应镜律: 全片至少 1 镜为**无台词反应 beat** (line_type=ambient, line 留空):
     被怼到失语 / 扶额 / 瞳孔地震 / 无语望天 — 用表情演戏, 是喜剧的"停顿拍"。
     motion 里写足表情动作 (没有台词的镜, 表情就是全部台词)。
   ④ (已废除 2026-08-21) 缓冲镜律曾强制全片 ≥1 无人空镜 (frame_chars=[]):
     用户反馈 ep29 镜3 空镜"没内容, 没人没动作" → 拍板废除强制空镜。
     现每镜必须有人有内容 (训练动作/对白/反应表情); 镜头呼吸靠组装层
     慢溶转场 + 反应镜承担, 不再插纯环境空镜。历史: 空镜是 2026-08-20
     "衔接太仓促" 的旧解法, 已被组装呼吸位取代。
   六镜织体参考模板 (镜位功能不变, 只换织体):
     镜1 林夏 dialog 单人(钩子) → 镜2 教练 dialog 单人(接话毒舌) →
     镜3 林夏训练镜 ambient 无台词有人(训练动作, 替代旧缓冲空镜位) →
     镜4 反应 beat ambient 无台词有人(表情演戏) → 镜5 双人同框 教练 dialog(转折)
     → 镜6 林夏 dialog 单人(悬念+sign-off)。
   每镜四字段:
   - shot (英文关键帧 prompt, 描述性静态场景), 必须:
     a) 以景别开头, **优先全身/大半身 (性感健身内容要秀身材, 2026-08-07 用户拍板)**: full body shot / medium full shot / three-quarter body shot 为主, 少用 medium close-up, 尽量不用 close-up (近景看不到身材);
     a') 景别轮换 (2026-08-21 视觉语言模块): 五大景别 EWS(极远景)/WS(全景)/MS(中景)/
     MCU(近景)/CU(特写)。全片至少覆盖 3 种, 相邻镜尽量不同景别。训练镜优先 MS/WS
     (秀身材+动作), 表情/反应镜用 MCU/CU (情绪), 环境镜用 EWS/WS (氛围)。禁止全片
     同景别 (反例 ep29 全 medium full 呆板)。
     b) 含场景(如 "modern gym interior with treadmill in the background");
     c) 含角色名 + 完整服装指纹(每镜重复, 不得省略);
     d) 含角色当前动作姿态 + 表情;
     e) 含灯光(如 "bright overhead gym lighting") + 构图(如 "vertical composition") + 情绪 mood(如 "modern fitness atmosphere")。
   - line (中文台词): 12-25 字, 短句, 口语化, 幽默/生活化, 与镜内容强关联, 适合配音。
     趣味标准: 台词要"让人想截图/发朋友"。优先自嘲式比喻("这杠铃比我的KPI还重")、
     职场黑话错位("深蹲算不算向下管理")、反差吐槽; 禁止正经鼓励/教学腔/舔狗文学。
     ⚠ **语法必须完整** (主谓宾齐全, 如 "你比很多人先迈出第一步" 而非 "你比很多人第一步了");
     ⚠ **单句单意** (一个陈述/一句问候, 不要两个分句挤一句 — H3 音频会重复后半句);
     ⚠ 不要括注/内心标注 (内心独白用 voiceover 另写)。
   - motion (英文 i2v 运动描述): 描述单一低幅度动作(i2v 模型做不了大动作), 必须:
     a) 主语 + 动作 + 幅度(如 "subtle step motion");
     b) 含表情变化(如 "satisfied smirk gradually forming");
     c) 句尾必带 3 句服装不变式, 颜色/标志必须与 fingerprint 完全一致:
        "the {{上衣颜色}} top stays {{颜色}} throughout, the {{裤色}} pants stay {{颜色}} throughout, the {{标志}} stays in the same position on the left chest"
      不变式示例: "the misty-blue sports bra stays misty-blue throughout, the black booty shorts stay black throughout, the white arc logo stays in the same position on the left chest"。
     d) 运镜 (2026-08-21 视觉语言模块): 每镜带轻运镜描述, 打破固定视野。运镜写在
        动作后, 格式 "camera slowly pushes in" / "camera tracks her movement"。
        训练镜用 Push In / Tracking (跟随力量动作), 对话镜用 Slow Push In,
        反应镜用 Subtle Dolly / Arc。幅度轻 (slow/slight), 不眩晕。禁止 Static 全片。
     ⚠ **训练镜强制规则 (题材含力量训练时, 本集必须遵守)**: 至少 4 镜的 motion 必须把**真实训练动作**写成
       动作主体 — deep squat 深蹲 / deadlift 硬拉 / overhead barbell press 杠铃推举 / barbell curl 弯举 /
       dumbbell row 划船 / walking lunge 箭步蹲 / bench press 卧推 任选。motion 的主语是"正在练"的动作,
       不是"站着说话/只比划手势"; 动作必须写进 motion 主谓, 不能只出现在 shot 背景里。
       训练镜 motion 示例: "Linxia does a deep squat holding the barbell on her shoulders, back straight,
       knees tracking over toes, hips sinking below parallel, a strained comedic grin, subtle squat motion,
       the misty-blue sports bra stays misty-blue throughout, the black booty shorts stay black throughout,
       the white arc logo stays in the same position on the left chest"。
   - speaker: 该镜说话的角色KEY。
   - line_type (可选, 三选一, 缺省= "dialog"):
       "dialog"    角色面向镜头说台词, 唇型由台词音频驱动 (默认, 有 line 的镜都是);
       "voiceover" 画外音解说, 角色闭嘴不开口, 后期叠旁白 (旁白镜, 仅此值需显式标注);
       "ambient"   无台词纯环境音 (舞蹈/健身等动作镜, 此时 line 留空字符串)。
   - frame_chars (可选, 该镜画面中的角色KEY数组, 缺省= [speaker]):
      单人镜 = [speaker]; 双人同框镜 = [两个角色KEY]。
      ⚠ **双人同框对话镜每集 ≤1 个** (2026-08-20 解禁, 依据 ep26 镜2 修复验证):
        双人对话镜已可靠 (说话者声线随 voice_id, 倾听者闭嘴), 但求稳为主 —
        有台词的镜默认单人正反打, 每集最多 1 个双人同框 dialog 镜 (建议放镜4
        转折位, 两人同框说话画面信息量最大)。其余双人同框镜仍必须无台词。
      ⚠ 单人镜优先正反打 (工业标准): 说话者单人入画面, 对话感靠织体三律的
        轮转+接话撑, 不靠同框。
   ⚠ **内心独白必须用 voiceover, 不能是 dialog**: 角色"心里想/内心OS/没说出来"的内容
     (如 "（内心：说好的不喘气呢！）") 要写成 `line_type: "voiceover"` — 角色闭嘴不出声,
     独白作为画外音后期叠入。line 只写独白内容本身 (去掉"（内心：...）"标注), 不要带括注。
   ⚠ **能说出声的才写 dialog**: 角色真的开口说的话才 dialog; 动作镜可加 "啊/呀" 等
     感叹词 (可 dialog 或 ambient, 别把整句内心独白塞进 dialog)。
   创作提示: 想让角色"开口说台词"→ dialog; 想让镜头有"解说感"但角色不说话 → voiceover。

【角色设定(用户给定, 必须遵守)】
{character_instructions}

【格式示例(双人轮转黄金结构 ep27_duet6《教练说我的动作像在打卡》— 织体四律的完整示范:
轮转 (林夏→教练→林夏) / 接话 (教练台词直接接林夏话头) / 训练镜 (镜3 有人有动作的
训练 beat, 2026-08-21 起替代旧缓冲空镜) / 反应镜 (镜4 无台词表情 beat) /
镜5 双人同框 dialog。严格模仿其结构与趣味; 这是唯一格式参考)】
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
    few_shot_shots: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
    few_shot_path: str | None = None,
) -> str:
    """构造完整单轮 prompt (任务+画风+格式规范+角色+few-shot).

    few_shot_shots 默认全 6 镜: ep27_duet 双人轮转示例必须整集展示
    (织体四律是跨镜结构, 只看单镜学不到轮转/接话)。
    """
    few = _load_few_shot_shots(*few_shot_shots, path=few_shot_path)
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
            "voice_id": _sanitize_voice_id(
                new_key,
                overrides.get("voice_id") or ch.get("voice_id"),
                voice_default),
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
        # frame_chars: LLM 显式给出 (单人=[speaker] / 双人=[两角色]) → 透传; 缺省 None (storyboard 推断)
        fc = s.get("frame_chars")
        if isinstance(fc, (list, tuple)) and fc:
            fc = [_canonical_key(c) for c in fc]
            fc = [c for c in fc if c in characters] or None
        else:
            fc = None
        shots[str(sid)] = {
            "shot": str(s.get("shot") or ""),
            "line": str(s.get("line") or ""),
            "motion": str(s.get("motion") or ""),
            "speaker": spk,
            "line_type": lt or None,
            "frame_chars": fc,
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
    n_shots: int = 6,
    style: str = "real",
    ep: str = "04",
    characters_spec: dict | None = None,
    *,
    scenes: list | None = None,     # P1-2: 场景列表 (novel_extract 产出, 供场景资产)
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
    if scenes:
        script["scenes"] = [str(s) for s in scenes if str(s).strip()]  # P1-2 场景资产
    dump_json(script, out_path)
    return script
