"""lib/h3prompt 官方六段式 builder 单测 (2026-08-15 整合).

不 mock 生成, 直接调 builder + 手工 script dict.
"""
import pytest

from lib import h3prompt

_CHARS = {
    "LINXIA": {"fingerprint": "1girl, solo, Linxia (林夏), a slim-fit 26-30 year-old "
                              "Chinese woman, black hair in a high ponytail, misty-blue top",
               "voice_id": "female-shaonv"},
    "client_k": {"fingerprint": "1girl, solo, client_k, red dress",
             "voice_id": "female-wenrou"},
    "LICOACH": {"fingerprint": "1boy, solo, Licoach, a muscular 28-35 year-old Chinese man",
                "voice_id": "male-qn-qingse"},
}


def _single_script():
    return {
        "title": "t", "duration_s": 5,
        "characters": {"LINXIA": _CHARS["LINXIA"]},
        "shots": {
            "1": {"shot": "medium shot, gym mat", "line": "今天开始自律",
                  "motion": "motion1 stays misty-blue", "speaker": "LINXIA"},
            "2": {"shot": "medium shot, gym mirror", "line": "这是画外旁白",
                  "motion": "motion2 stays", "speaker": "LINXIA", "line_type": "voiceover"},
            "3": {"shot": "full body, jumping jacks", "line": "",
                  "motion": "motion3 stays", "speaker": "LINXIA"},
        },
    }


def _dual_script():
    return {
        "title": "t", "duration_s": 5,
        "characters": _CHARS,
        "shots": {
            "1": {"shot": "two people facing each other", "line": "你练得怎么样？",
                  "motion": "motion1 stays", "speaker": "LINXIA",
                  "frame_chars": ["LINXIA", "client_k"]},
            "2": {"shot": "two people in gym", "line": "",
                  "motion": "motion2 stays", "speaker": "LINXIA",
                  "frame_chars": ["LINXIA", "client_k"], "line_type": "ambient"},
        },
    }


# ---- dialog ----

def test_dialog_six_sections_in_order():
    p = h3prompt.build_dialog_prompt(_single_script(), 1)
    order = [p.index(s) for s in (
        "subject_definitions:", "summary:", "retention_analysis:",
        "detailed_description:", "overall_soundscape:", "non_diegetic_music:")]
    assert order == sorted(order)
    assert "non_diegetic_music:\nN/A" in p


def test_dialog_subject_and_audio_and_dialogue():
    p = h3prompt.build_dialog_prompt(_single_script(), 1)
    assert "<Subject 1> is the person in <Picture 1>, a slim-fit 26-30 year-old" in p
    assert "<Audio 1> is the speech reference for <Subject 1> (S1)" in p
    assert "<d>[中文]今天开始自律</d>" in p
    assert "motion1 stays misty-blue" in p      # motion 原样
    assert "1girl" not in p.split("subject_definitions:")[1]  # danbooru 标签剥掉


def test_dialog_no_legacy_chinese_instruction():
    p = h3prompt.build_dialog_prompt(_single_script(), 1)
    assert "用中文清晰自然地念出台词" not in p
    assert "stays completely silent" not in p   # dialog 不写沉默


# ---- voiceover ----

def test_voiceover_no_narration_wording():
    p = h3prompt.build_voiceover_prompt(_single_script(), 2)
    low = p.lower()
    for w in ("voiceover", "narration", "off-screen"):
        assert w not in low, f"voiceover 不该出现 {w}"
    assert "stays completely silent" in p
    assert "overall_soundscape:" in p
    assert "<Audio 1>" not in p


# ---- voiceover 发力镜 (2026-08-18 ep22 镜3 教训) ----
# 沉默子句把"不说话"错写成"闭嘴+轻呼吸+idle", 压过 motion 的喘气描述
# (用户: 没看到女主喘气的样子)。发力镜应保"不说话"、放"呼吸/嘴型"。

def test_voiceover_exertion_motion_keeps_heavy_breathing():
    s = _single_script()
    s["shots"]["2"]["motion"] = ("Linxia lowers the dumbbell and pants heavily, "
                                 "mouth wide open gasping for air, out of breath")
    p = h3prompt.build_voiceover_prompt(s, 2)
    assert "mouth closed" not in p          # 不再压嘴型
    assert "idle motion" not in p           # 不再压运动幅度
    assert "stays completely silent" in p   # 沉默本意 (不说话) 保留
    assert "does not speak" in p
    assert "breathing exactly as described in the motion" in p  # 呼吸交还 motion
    assert "mouth open as described" in p   # 嘴型交还 motion


def test_voiceover_calm_motion_keeps_strict_silence():
    p = h3prompt.build_voiceover_prompt(_single_script(), 2)  # motion: "motion2 stays"
    assert "mouth closed" in p              # 平静镜维持原严格沉默
    assert "idle motion" in p


def test_ambient_exertion_motion_keeps_heavy_breathing():
    s = _single_script()
    s["shots"]["3"]["motion"] = ("Linxia catches her breath, panting and gasping, "
                                 "exhausted and sweaty after the set")
    p = h3prompt.build_ambient_prompt(s, 3)
    assert "mouth closed" not in p
    assert "does not speak" in p


# ---- ambient ----

def test_ambient_silence_and_soundscape():
    p = h3prompt.build_ambient_prompt(_single_script(), 3)
    assert "stays completely silent" in p
    assert "overall_soundscape:" in p
    assert "<d>" not in p
    assert "<Audio 1>" not in p
    assert "rhythmic footsteps" in p           # ambient soundscape


# ---- dual ----

def test_dual_dialog_two_subjects_speaker_audio():
    p = h3prompt.build_dual_dialog_prompt(_dual_script(), 1)
    assert "<Subject 1> is the person in <Picture 1>, red dress." in p          # 倾听者 client_k
    assert "<Subject 2> is the person in <Picture 2>, a slim-fit 26-30 year-old" in p  # 说话者 LINXIA
    assert "<Audio 1> is the voice-timbre reference for <Subject 2> (S1)" in p
    assert "<Subject 2> (S1)" in p
    assert "<d>[中文]你练得怎么样？</d>" in p
    assert "mouth closed throughout" in p      # 倾听者闭嘴
    assert "no speech and no lip movement" in p


def test_dual_scene_both_silent():
    p = h3prompt.build_dual_scene_prompt(_dual_script(), 2, kind="ambient")
    assert "Both stay completely silent" in p
    assert "<Audio 1>" not in p
    assert "rhythmic footsteps" in p
    assert "<d>" not in p


# ---- 声线性别自适应 (2026-08-20 ep26 镜2 教训) ----
# 男教练说话时 prompt 硬编码 "female voice"/"female vocal layer" 与男声 TTS 冲突
# → H3 把嘴型安到女主脸上 (视觉说话人错位)。声线措辞必须跟着 voice_id 走。

def _male_single_script():
    return {
        "title": "t", "duration_s": 5,
        "characters": {"LICOACH": _CHARS["LICOACH"]},
        "shots": {
            "1": {"shot": "medium shot, gym floor", "line": "第一次都这样",
                  "motion": "coach gestures calmly", "speaker": "LICOACH"},
        },
    }


def _male_dual_script():
    """双人镜, 教练说话 / 林夏倾听 (ep26 镜2 同构)."""
    return {
        "title": "t", "duration_s": 5,
        "characters": {"LINXIA": _CHARS["LINXIA"], "LICOACH": _CHARS["LICOACH"]},
        "shots": {
            "1": {"shot": "two people facing each other", "line": "你已经比很多人先迈出第一步了",
                  "motion": "coach pats the rack", "speaker": "LICOACH",
                  "frame_chars": ["LINXIA", "LICOACH"]},
        },
    }


def test_dialog_male_speaker_male_wording():
    p = h3prompt.build_dialog_prompt(_male_single_script(), 1)
    assert "adult male voice" in p              # 声线措辞 = male
    assert "female voice" not in p
    assert "male vocal layer" in p              # Audio 1 描述同步
    assert "female vocal layer" not in p
    assert "His lips move" in p                 # 代词跟随


def test_dialog_female_speaker_wording_unchanged():
    p = h3prompt.build_dialog_prompt(_single_script(), 1)   # LINXIA female-shaonv
    assert "youthful female voice" in p
    assert "female vocal layer" in p
    assert "Her lips move" in p
    assert "adult male voice" not in p       # 精确断言 (female 含 male 子串, 不能反着查)


def test_dual_dialog_male_speaker_male_wording():
    """ep26 镜2 病灶: 教练说话 → Subject 2 声线必须是 male, 倾听者 (女主) 闭嘴."""
    p = h3prompt.build_dual_dialog_prompt(_male_dual_script(), 1)
    assert "<Subject 2> is the person in <Picture 2>, a muscular 28-35" in p  # 教练=说话者
    assert "adult male voice" in p
    assert "male vocal layer" in p
    assert "female vocal layer" not in p       # 病灶断言: 不再硬编码女声
    assert "His lips move" in p
    assert "mouth closed throughout" in p      # 倾听者仍闭嘴


def test_dual_dialog_female_speaker_wording_unchanged():
    p = h3prompt.build_dual_dialog_prompt(_dual_script(), 1)  # LINXIA 说话
    assert "youthful female voice" in p
    assert "female vocal layer" in p
    assert "Her lips move" in p


def test_dual_dialog_kind_voiceover_soundscape():
    p = h3prompt.build_dual_scene_prompt(_dual_script(), 2, kind="voiceover")
    assert "low ventilation hum" in p          # voiceover soundscape (无 footsteps)


# ---- identity 双参考 (R2V A/B, 2026-08-19) ----

def test_dialog_identity_two_pictures_one_subject():
    """双参考混合: Picture 1=kf 构图锚, Picture 2=定妆照身份锚, 同为 Subject 1."""
    p = h3prompt.build_dialog_prompt_identity(_single_script(), 1)
    # 六段式结构不变
    order = [p.index(s) for s in (
        "subject_definitions:", "summary:", "retention_analysis:",
        "detailed_description:", "overall_soundscape:", "non_diegetic_music:")]
    assert order == sorted(order)
    # Subject 1 由两张 Picture 共同定义 (构图 + 身份)
    assert "<Subject 1> is the person in <Picture 1>" in p
    assert "<Picture 1> and <Picture 2> show the same person" in p
    assert "<Subject 2>" not in p                    # 单人, 不出现第二主体
    # retention: 两张图分工明确
    assert "composition anchor" in p                 # P1 构图
    assert "identity reference" in p                 # P2 身份
    # 台词/音频链路保持
    assert "<Audio 1> is the speech reference for <Subject 1> (S1)" in p
    assert "<d>[中文]今天开始自律</d>" in p


def test_dialog_identity_face_from_picture2():
    """明确脸以 Picture 2 (定妆照) 为准 — 这是 A/B 实验的核心变量."""
    p = h3prompt.build_dialog_prompt_identity(_single_script(), 1)
    assert "the face follows <Picture 2>" in p
    # summary 说明两图分工
    assert "<Picture 1> provides the framing and composition" in p
    assert "<Picture 2> provides the identity" in p


# ---- soundscape 覆盖 ----

def test_soundscape_override_script_and_shot():
    s = _single_script()
    s["soundscape"] = "script-level room tone."
    assert "script-level room tone" in h3prompt.build_dialog_prompt(s, 1)
    s["shots"]["1"]["soundscape"] = "shot-level hum."
    assert "shot-level hum" in h3prompt.build_dialog_prompt(s, 1)


# ---- _strip_tags ----

def test_strip_tags_strips_leading_tags_and_name():
    assert h3prompt._strip_tags(
        "1girl, solo, Linxia (林夏), a slim-fit Chinese woman, black hair") \
        == "a slim-fit Chinese woman, black hair"
    # 无名字直接外观 (男教练)
    assert h3prompt._strip_tags(
        "1boy, solo, a fit male coach, short black hair, strong jawline") \
        == "a fit male coach, short black hair, strong jawline"
    # 短正文不剥过头
    assert h3prompt._strip_tags("1girl, solo, client_k, red dress") == "red dress"
    # 空 → 回退整段
    assert h3prompt._strip_tags("1girl, solo") == "1girl, solo"


def test_style_sentence_default_and_override():
    assert "fitness-short style" in h3prompt._style_sentence()
    assert "custom style" in h3prompt._style_sentence({"style_sentence": "custom style."})


# ---- 缓冲空镜 (织体第④律 2026-08-20) ----

def _buffer_script():
    """镜3 = 缓冲空镜: frame_chars=[] + speaker='' (无角色, 无人)."""
    return {
        "title": "t", "duration_s": 5,
        "characters": {"LINXIA": _CHARS["LINXIA"]},
        "shots": {
            "3": {"shot": "close-up of gym door card scanner",
                  "line": "", "motion": "screen blinks red",
                  "speaker": "", "frame_chars": [], "line_type": "ambient"},
        },
    }


def test_buffer_prompt_no_character_crash():
    """缓冲空镜: 无人镜调 build_buffer_prompt 不崩 (character() 返回 None)."""
    prompt = h3prompt.build_buffer_prompt(_buffer_script(), 3)
    assert "no characters" in prompt or "no people" in prompt


def test_buffer_prompt_has_soundscape_no_subject():
    """缓冲空镜: 六段式里无 <Subject>, 保留 overall_soundscape."""
    prompt = h3prompt.build_buffer_prompt(_buffer_script(), 3)
    assert "<Subject" not in prompt
    assert "overall_soundscape:" in prompt


def test_character_none_contract_all_builders_tolerate():
    """character() → None 契约: 所有单人 builder 对缓冲空镜不崩 (容错 or {})."""
    script = _buffer_script()
    # 直接喂 speaker='' 的缓冲镜给各 builder — 必须返回字符串, 不得抛 NoneType
    for builder in (h3prompt.build_dialog_prompt,
                    h3prompt.build_dialog_prompt_identity,
                    h3prompt.build_voiceover_prompt,
                    h3prompt.build_ambient_prompt):
        out = builder(script, 3)
        assert isinstance(out, str) and out.strip()


# ---- 运镜 clause (2026-08-21 ep30 镜3 教训: 剧本 motion 含运镜词) ----

def test_camera_clause_extracts_from_motion():
    script = _single_script()
    script["shots"]["1"]["motion"] = "Linxia squats, subtle motion, camera slowly pushes in, the bra stays misty-blue"
    assert "camera slowly pushes in" in h3prompt._camera_clause(script, 1)
    assert h3prompt.build_dialog_prompt(script, 1)  # 不崩
