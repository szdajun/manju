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


def test_dual_dialog_kind_voiceover_soundscape():
    p = h3prompt.build_dual_scene_prompt(_dual_script(), 2, kind="voiceover")
    assert "low ventilation hum" in p          # voiceover soundscape (无 footsteps)


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
