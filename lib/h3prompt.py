"""官方 MiniMax H3 六段式 prompt 构建器 (h3-prompt-writing Ref2VA 规范).

基于 MiniMax 官方 skill `h3-prompt-writing` (references/ref-en.txt) 的六段式格式:
  subject_definitions → summary → retention_analysis → detailed_description
  → overall_soundscape → non_diegetic_music

2026-08-14/15 在本地 ComfyUI H3 GGUF (768×1344 turbo) 上四模式对照验证通过:
  - dialog:     六段式 + `<Subject 1>`(from `<Picture 1>`=关键帧) + `<Audio 1>` + `<d>[中文]台词</d>` + (S1)
                 → 中文正确、无乱码前奏 (现有 hack 的 768 乱码问题消失)
  - voiceover:  六段式 + 明确闭嘴沉默, **不写旁白** (TTS 旁白由组装叠加) → 干净静音
  - ambient:    六段式 + 沉默 + overall_soundscape → 干净静音
  - 双人对话:   `<Subject 1>`=倾听者(<Picture 1>/ref0), `<Subject 2>`=说话者(<Picture 2>/ref1),
                `<Audio 1>`=说话者(S1)声线, 倾听者闭嘴 → 一人说话一人倾听

⛔ 核心规律 (实验钉死): **prompt 里绝不能出现 voiceover/narration/off-screen 字样**
(否则本地 H3 会自己生成乱语)。台词保留中文在 `<d>[中文]...</d>`, prompt 其余英文。
motion/shot/line 原样插入 (不虚构); fingerprint 去掉 danbooru 头标签后原样用。
"""
from .storyboard import character, frame_chars, shot, speaker

# ============ 常量 ============

_STYLE_DEFAULT = ("The target video is in a realistic vertical fitness-short style "
                  "with bright overhead gym lighting and an energetic mood.")

# 固定健身房屋调 (dialog/voiceover/ambient 三类, 实验验证用)
_SOUNDSCAPES = {
    "dialog": ("Bright indoor gym room tone with a low ventilation hum and faint echoes "
               "of weights, continuing throughout the video."),
    "voiceover": ("Bright indoor gym room tone with a low ventilation hum, "
                  "continuing throughout the video."),
    "ambient": ("Bright indoor gym room tone with rhythmic footsteps on the mat and the "
                "rustle of athletic wear, continuing throughout the video."),
}

# 说话子句 (声线措辞按说话者 voice_id 性别自适应, 2026-08-20 ep26 镜2 教训:
# 男教练说话时 prompt 写 "female voice" 而音频是男声 → H3 把嘴型安到女主脸上)
_SPEECH_VOICE_DESC = {
    "female": "a clear, natural youthful female voice",
    "male": "a clear, natural adult male voice",
}
_SPEECH_CLAUSE = ("<Subject 1> (S1) says in {voice_desc}, "
                  "<d>[中文]{line}</d> Her lips move in sync with the Mandarin line and "
                  "close naturally after speaking, keeping an upright natural posture, "
                  "head facing the camera with a natural expression.")
_SPEECH_CLAUSE_MALE = ("<Subject 1> (S1) says in {voice_desc}, "
                       "<d>[中文]{line}</d> His lips move in sync with the Mandarin line and "
                       "close naturally after speaking, keeping an upright natural posture, "
                       "head facing the camera with a natural expression.")
_SILENT_CLAUSE = ("<Subject 1> stays completely silent with mouth closed, does not speak "
                  "and has no speech, only natural subtle breathing and idle motion, "
                  "lips remaining closed.")
_SILENT_CLAUSE_AMB = ("<Subject 1> stays completely silent with mouth closed, does not "
                      "speak and has no speech, natural breathing only.")
# 发力镜沉默子句 (2026-08-18 ep22 镜3 教训): motion 写喘气/发力时, "闭嘴+轻呼吸+idle"
# 会压过喘气描述 (用户看不到喘气)。"不说话"≠"不喘气": 保沉默, 放呼吸/嘴型交还 motion。
_SILENT_CLAUSE_EXERTION = ("<Subject 1> stays completely silent and does not speak or "
                           "say any words, breathing exactly as described in the motion, "
                           "mouth open as described, no speech sounds, only breath sounds.")
# 发力/喘气关键词 (检测 motion 是否发力镜; 小写匹配)
_EXERTION_WORDS = (
    "pant", "gasp", "out of breath", "breathless", "winded", "catching her breath",
    "catches his breath", "catches her breath", "exhausted", "heaving", "sweat",
    "trembling", "gasping", "huffing", "puffing", "winded", "heavy lifting", "wiped out",
)


def _is_exertion_motion(motion: str) -> bool:
    """motion 含发力/喘气词 → True (沉默子句切发力版, 不压呼吸/嘴型)."""
    low = (motion or "").lower()
    return any(w in low for w in _EXERTION_WORDS)


def _voice_gender(script: dict, sid: int | str) -> str:
    """说话者声线性别: voice_id 前缀 male-/female- 判定, 未知默认 female (历史行为).

    2026-08-20 ep26 镜2 教训: 双人镜教练说话, prompt 硬编码 female voice 与男声 TTS
    冲突 → H3 让女主对口型。声线措辞必须与 TTS 实际声线一致。
    """
    vid = ((character(script, sid) or {}).get("voice_id") or "").lower()
    return "male" if vid.startswith("male") else "female"


def _speech_clause(script: dict, sid: int | str) -> str:
    """说话子句按说话者性别选措辞 (voice_desc + His/Her 代词)."""
    gender = _voice_gender(script, sid)
    clause = _SPEECH_CLAUSE_MALE if gender == "male" else _SPEECH_CLAUSE
    return clause.format(voice_desc=_SPEECH_VOICE_DESC[gender],
                         line=_clean_line(shot(script, sid).get("line", "")))


def _camera_clause(script: dict, sid: int | str) -> str:
    """从 motion 提取运镜描述 (含 camera/pan/tilt/push 等词的部分), 写进 detailed_description."""
    motion = shot(script, sid).get("motion", "")
    for marker in ("camera", "push", "pull", "pan", "tilt", "track", "dolly", "arc", "zoom"):
        if marker in motion:
            seg = motion[motion.index(marker):].split(",")[0].strip().rstrip(".")
            return f" {seg}."
    return ""


# ============ helpers ============

def _clean_line(line: str) -> str:
    """台词清洗: strip + 引号转义 + 空白折叠 (与 legacy `_dialog_speech_cn` 同规)."""
    return " ".join((line or "").strip().replace('"', "'").split())


def _capitalize(s: str) -> str:
    """句子首字母大写 (用于 `[Shot 1]` 后的场景描述)."""
    return s[:1].upper() + s[1:] if s else s


def _ensure_period(s: str) -> str:
    """结尾补句号 (motion/scene 插入 detailed_description 前)."""
    s = (s or "").strip()
    return s if s.endswith((".", "!", "?")) else s + "."


def _strip_tags(fp: str) -> str:
    """fingerprint 去 danbooru 头标签/角色名, 保留外观正文原样.

    例: "1girl, solo, Linxia (林夏), a slim-fit 26-30 year-old Chinese woman, ..."
        → "a slim-fit 26-30 year-old Chinese woman, ..."
    规则: 跳过已知 danbooru 标签 (1girl/1boy/solo/...), 再最多跳过 1 个角色名段,
    其余正文原样保留. 不依赖"正文必须以冠词/动词开头", 防把 "misty-blue top" 等剥掉.
    """
    parts = [p.strip() for p in (fp or "").split(",") if p.strip()]
    _LEADING = {"1girl", "1boy", "solo", "male", "female", "boy", "girl", "man", "woman"}
    _NAME_LEAD = ("a ", "an ", "the ", "wearing ", "dressed ", "wears ", "has ", "in ")
    i = 0
    # 1) 跳过已知标签
    while i < len(parts) and parts[i].lower().strip() in _LEADING:
        i += 1
    # 2) 最多跳过 1 个角色名段 (不以冠词/外观动词开头且较短)
    if i < len(parts):
        p = parts[i]
        low = p.lower()
        if len(p) < 24 and not any(low.startswith(m) for m in _NAME_LEAD):
            i += 1
    body = ", ".join(parts[i:]).strip()
    return body if body else (fp or "").strip()


def _subject_desc(fp: str, pic: str) -> str:
    """`<Subject N>` 定义的主体描述: "the person in <Picture N>, {外观正文}"."""
    body = _strip_tags(fp)
    return f"the person in {pic}, {body}"


def _style_sentence(script: dict | None = None) -> str:
    """风格句 (script 可覆盖, 默认健身短片风格)."""
    return (script or {}).get("style_sentence") or _STYLE_DEFAULT


def _soundscape_for(script: dict, sid: int | str, kind: str) -> str:
    """soundscape 优先级: shot.soundscape → script.soundscape → 默认."""
    sh = shot(script, sid)
    return (sh.get("soundscape") or (script or {}).get("soundscape")
            or _SOUNDSCAPES[kind])


def _retention_subject(n: int) -> str:
    return (f"<Subject {n}> (appears in [Shot 1]): fully_preserved - the appearance, "
            "clothing, hairstyle, and outfit are retained as defined in subject_definitions.")


def _retention_picture(pic: str) -> str:
    return (f"{pic} ([Shot 1] composition anchor): fully_preserved - "
            "the framing and composition of the shot are retained.")


def _retention_audio_ref() -> str:
    return ("<Audio 1>: reference - the Mandarin voice and delivery of the line follow "
            "<Audio 1> without copying the original signal.")


# ============ 单人 ============

def build_dialog_prompt(script: dict, sid: int | str) -> str:
    """单人对白 (有 TTS 音频, Ref2VA 单参考图 = 关键帧)."""
    sh = shot(script, sid)
    fp = (character(script, sid) or {}).get("fingerprint", "")
    scene = _capitalize(sh.get("shot", "").rstrip("."))
    motion = _ensure_period(sh.get("motion", ""))
    line = _clean_line(sh.get("line", ""))
    subj = _subject_desc(fp, "<Picture 1>")
    sound = _soundscape_for(script, sid, "dialog")
    gender = _voice_gender(script, sid)
    return f"""subject_definitions:
<Subject 1> is {subj}.
<Audio 1> is the speech reference for <Subject 1> (S1), containing a spoken Mandarin {gender} vocal layer.

summary:
[reference generation + audio reference] The target video shows <Subject 1> in the described gym scene, performing the described motion, and speaking a line in Mandarin while the appearance and clothing stay unchanged. <Picture 1> provides the identity, pose, and composition; <Audio 1> guides the voice and delivery of <Subject 1>'s spoken line.

retention_analysis:
{_retention_subject(1)}
{_retention_picture('<Picture 1>')}
{_retention_audio_ref()}

detailed_description:
{_style_sentence(script)}
[Shot 1] {scene}. {motion} {_camera_clause(script, sid)}{_speech_clause(script, sid)}

overall_soundscape:
{sound}

non_diegetic_music:
N/A"""


def _silent_clause(script: dict, sid: int | str, ambient: bool = False) -> str:
    """沉默子句按镜选: 发力镜 (喘气/力竭) 用发力版, 其余用严格版.

    ambient=True → 严格版用 _SILENT_CLAUSE_AMB (历史行为).
    """
    if _is_exertion_motion(shot(script, sid).get("motion", "")):
        return _SILENT_CLAUSE_EXERTION
    return _SILENT_CLAUSE_AMB if ambient else _SILENT_CLAUSE


def build_dialog_prompt_identity(script: dict, sid: int | str) -> str:
    """单人对白双参考混合 (R2V A/B, 2026-08-19).

    ref_images = [kf 关键帧, 角色定妆照]: <Picture 1> 锁构图/姿势 (kf),
    <Picture 2> 锁身份 (定妆照), 两者同为 <Subject 1> — 脸以 <Picture 2> 为准.
    其余 (音频链/台词/六段式) 与 build_dialog_prompt 一致.
    """
    sh = shot(script, sid)
    fp = (character(script, sid) or {}).get("fingerprint", "")
    scene = _capitalize(sh.get("shot", "").rstrip("."))
    motion = _ensure_period(sh.get("motion", ""))
    line = _clean_line(sh.get("line", ""))
    subj = _subject_desc(fp, "<Picture 1>")
    sound = _soundscape_for(script, sid, "dialog")
    gender = _voice_gender(script, sid)
    return f"""subject_definitions:
<Subject 1> is {subj}.
<Picture 1> and <Picture 2> show the same person: <Picture 1> and <Picture 2> both depict <Subject 1>.
<Audio 1> is the speech reference for <Subject 1> (S1), containing a spoken Mandarin {gender} vocal layer.

summary:
[reference generation + audio reference] The target video shows <Subject 1> in the described gym scene, performing the described motion, and speaking a line in Mandarin while the appearance and clothing stay unchanged. <Picture 1> provides the framing and composition of the shot; <Picture 2> provides the identity of <Subject 1>; <Audio 1> guides the voice and delivery of <Subject 1>'s spoken line.

retention_analysis:
{_retention_subject(1)}
<Picture 1> ([Shot 1] composition anchor): fully_preserved - the framing, pose, and composition of the shot are retained.
<Picture 2> ([Shot 1] identity reference): fully_preserved - the facial identity of <Subject 1> is retained, the face follows <Picture 2>.
{_retention_audio_ref()}

detailed_description:
{_style_sentence(script)}
[Shot 1] {scene}. {motion} {_camera_clause(script, sid)}{_speech_clause(script, sid)}

overall_soundscape:
{sound}

non_diegetic_music:
N/A"""


def build_voiceover_prompt(script: dict, sid: int | str) -> str:
    """单人旁白 (无音频; ⛔ 不写旁白字样, TTS 旁白由组装叠加)."""
    sh = shot(script, sid)
    fp = (character(script, sid) or {}).get("fingerprint", "")
    scene = _capitalize(sh.get("shot", "").rstrip("."))
    motion = _ensure_period(sh.get("motion", ""))
    subj = _subject_desc(fp, "<Picture 1>")
    sound = _soundscape_for(script, sid, "voiceover")
    return f"""subject_definitions:
<Subject 1> is {subj}.

summary:
[reference generation] The target video shows <Subject 1> in the described gym scene, performing the described motion, remaining completely silent. <Picture 1> provides the identity, pose, and composition.

retention_analysis:
{_retention_subject(1)}
{_retention_picture('<Picture 1>')}

detailed_description:
{_style_sentence(script)}
[Shot 1] {scene}. {motion} {_camera_clause(script, sid)}{_silent_clause(script, sid)}

overall_soundscape:
{sound}

non_diegetic_music:
N/A"""


def build_ambient_prompt(script: dict, sid: int | str) -> str:
    """单人环境音 (无音频; 沉默 + overall_soundscape)."""
    sh = shot(script, sid)
    fp = (character(script, sid) or {}).get("fingerprint", "")
    scene = _capitalize(sh.get("shot", "").rstrip("."))
    motion = _ensure_period(sh.get("motion", ""))
    subj = _subject_desc(fp, "<Picture 1>")
    sound = _soundscape_for(script, sid, "ambient")
    return f"""subject_definitions:
<Subject 1> is {subj}.

summary:
[reference generation] The target video shows <Subject 1> in the described gym scene, performing the described motion, remaining completely silent. <Picture 1> provides the identity, pose, and composition.

retention_analysis:
{_retention_subject(1)}
{_retention_picture('<Picture 1>')}

detailed_description:
{_style_sentence(script)}
[Shot 1] {scene}. {motion} {_camera_clause(script, sid)}{_silent_clause(script, sid, ambient=True)}

overall_soundscape:
{sound}

non_diegetic_music:
N/A"""


# ============ 双人 ============

def build_buffer_prompt(script: dict, sid: int | str) -> str:
    """环境缓冲空镜 (织体第④律 2026-08-20): frame_chars=[] 无人 — 无 Subject/ref,
    纯场景 + overall_soundscape. kf 是 txt2img 无 PuLID, i2v 无 ref 图.
    """
    sh = shot(script, sid)
    scene = _capitalize(sh.get("shot", "").rstrip("."))
    motion = _ensure_period(sh.get("motion", ""))
    sound = _soundscape_for(script, sid, "ambient")
    return f"""subject_definitions:
N/A (no characters in this shot)

summary:
[reference generation] The target video shows only the described environment scene with no people, {motion} The scene remains completely empty of any person or face throughout.

retention_analysis:
- <Picture 1> provides the composition and lighting of the empty scene.

detailed_description:
{_style_sentence(script)}
[Shot 1] {scene}, with no people visible anywhere in the frame. {motion} No characters, no faces, no people appear at any point.

overall_soundscape:
{sound}

non_diegetic_music:
N/A"""

def _dual_roles(script: dict, sid: int | str):
    """返回 (listener_key, speaker_key, listener_fp, speaker_fp). 顺序对齐 ref_images=[倾听者, 说话者]."""
    spk = speaker(script, sid)
    fc = frame_chars(script, sid) or list(script["characters"].keys())
    other = next((c for c in fc if c != spk), spk)
    return (other, spk,
            (script["characters"].get(other) or {}).get("fingerprint", ""),
            (script["characters"].get(spk) or {}).get("fingerprint", ""))


def build_dual_dialog_prompt(script: dict, sid: int | str) -> str:
    """双人对话 (双 ref + TTS 音频; <Subject 1>=倾听者/<Picture 1>, <Subject 2>=说话者/<Picture 2>)."""
    sh = shot(script, sid)
    _, spk_k, fp_other, fp_spk = _dual_roles(script, sid)
    scene = _capitalize(sh.get("shot", "").rstrip("."))
    motion = _ensure_period(sh.get("motion", ""))
    line = _clean_line(sh.get("line", ""))
    subj1 = _subject_desc(fp_other, "<Picture 1>")
    subj2 = _subject_desc(fp_spk, "<Picture 2>")
    sound = _soundscape_for(script, sid, "dialog")
    gender = _voice_gender(script, sid)
    voice_desc = _SPEECH_VOICE_DESC[gender]
    spk_pron = "His" if gender == "male" else "Her"
    return f"""subject_definitions:
<Subject 1> is {subj1}.
<Subject 2> is {subj2}.
<Audio 1> is the voice-timbre reference for <Subject 2> (S1), containing a spoken Mandarin {gender} vocal layer.

summary:
[reference generation + audio reference] The target video shows <Subject 2> and <Subject 1> in the described gym scene, <Subject 2> speaking a line in Mandarin while <Subject 1> listens silently. <Picture 1> provides the listener's identity; <Picture 2> provides the speaker's identity; <Audio 1> guides the voice and delivery of <Subject 2>'s line.

retention_analysis:
{_retention_subject(1)}
{_retention_subject(2)}
<Picture 1> ([Shot 1] identity reference): fully_preserved - the listener's appearance is retained.
<Picture 2> ([Shot 1] identity reference): fully_preserved - the speaker's appearance is retained.
{_retention_audio_ref()}

detailed_description:
{_style_sentence(script)}
[Shot 1] {scene}. <Subject 2>, the person in <Picture 2> whose appearance is retained, and <Subject 1>, the person in <Picture 1> whose appearance is retained, are both fully visible with no obstruction, bodies upright and natural. {motion} {_camera_clause(script, sid)}<Subject 2> (S1) looks at <Subject 1> and says in {voice_desc}, <d>[中文]{line}</d> {spk_pron} lips move in sync with the Mandarin line and close naturally after speaking, speaking the line exactly once, then immediately becoming fully silent with no extra sound. <Subject 1> remains completely silent with mouth closed throughout, no speech and no lip movement, listening to <Subject 2> with a natural expression and subtle nodding, minimal extra movement, body posture stable.

overall_soundscape:
{sound}

non_diegetic_music:
N/A"""


def build_dual_scene_prompt(script: dict, sid: int | str, kind: str = "ambient") -> str:
    """双人无对白 (voiceover/ambient 双人同框互动, 无音频; 两人都沉默)."""
    sh = shot(script, sid)
    _, _, fp_other, fp_spk = _dual_roles(script, sid)
    scene = _capitalize(sh.get("shot", "").rstrip("."))
    motion = _ensure_period(sh.get("motion", ""))
    subj1 = _subject_desc(fp_other, "<Picture 1>")
    subj2 = _subject_desc(fp_spk, "<Picture 2>")
    sound_kind = "voiceover" if kind == "voiceover" else "ambient"
    sound = _soundscape_for(script, sid, sound_kind)
    return f"""subject_definitions:
<Subject 1> is {subj1}.
<Subject 2> is {subj2}.

summary:
[reference generation] The target video shows <Subject 1> and <Subject 2> in the described gym scene, performing the described motion together, both remaining completely silent. <Picture 1> and <Picture 2> provide their identities and composition.

retention_analysis:
{_retention_subject(1)}
{_retention_subject(2)}
<Picture 1> ([Shot 1] identity reference): fully_preserved - the listener's appearance is retained.
<Picture 2> ([Shot 1] identity reference): fully_preserved - the speaker's appearance is retained.

detailed_description:
{_style_sentence(script)}
[Shot 1] {scene}. <Subject 1> and <Subject 2> are both fully visible with no obstruction, bodies upright and natural. {motion} {_camera_clause(script, sid)}Both stay completely silent with mouths closed, no speech and no lip movement, natural breathing only.

overall_soundscape:
{sound}

non_diegetic_music:
N/A"""
