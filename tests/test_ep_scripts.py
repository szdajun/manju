"""tests/test_ep_scripts.py — 通用化阶段脚本从剧本读数据 (manju Phase 3, 2026-08-06).

不调真实 ComfyUI/Seedance/Minimax/ffmpeg — mock 重依赖.
验证 4 个阶段脚本加 --ep 后, 从 load_script 读剧本数据驱动 (而非硬编码 ep03).
"""
import os
import sys
from unittest.mock import patch

import pytest

from lib.scriptgen import STYLES, merge_script


def _ep04_script():
    """构造 ep04 规范化剧本 dict (供阶段脚本 mock 测试)."""
    chars = {"LINXIA": {"fingerprint": "1girl, solo, Linxia, misty-blue top",
                        "voice_id": "female-shaonv"}}
    shots = {}
    for i in range(1, 4):
        shots[str(i)] = {"shot": f"medium shot {i}, Linxia in misty-blue top",
                         "line": f"台词{i}",
                         "motion": f"motion {i} stays misty-blue throughout",
                         "speaker": "LINXIA"}
    return merge_script(
        {"title": "测试集", "duration_s": 5, "characters": chars, "shots": shots},
        ep="04", style=STYLES["real"]["style"], neg=STYLES["real"]["neg"],
        char_overrides={"LINXIA": {"face_ref": "ref_face.png", "body_ref": "ref_body.png"}})


def _write_script(tmp_path, script):
    """写 output/ep04/script.json 到 tmp_path (load_script 自动探测)."""
    import json
    ep_dir = os.path.join(str(tmp_path), "output", "ep04")
    os.makedirs(ep_dir, exist_ok=True)
    with open(os.path.join(ep_dir, "script.json"), "w", encoding="utf-8") as f:
        json.dump(script, f, ensure_ascii=False)
    return str(tmp_path)


# ---- kf: build_positive 从剧本取字段 ----

def test_kf_build_positive_uses_script(tmp_path):
    """build_positive 用剧本: fingerprint + style + motion (非硬编码 ep03)."""
    from scripts import gen_ep03_real_batch as kf
    script = _ep04_script()
    pos = kf.build_positive(1, script)
    assert "misty-blue top" in pos          # 剧本 fingerprint
    assert "photorealistic" in pos           # 剧本 style
    assert "stays misty-blue" in pos         # 剧本 motion 不变式
    assert "medium shot 1" in pos            # 剧本 shot 场景


# ---- i2v: 从剧本读 motion ----

def test_i2v_uses_script_motion(tmp_path):
    """i2v 脚本: motion 从剧本 shots[N].motion, 输出 ep04_shotNN.mp4."""
    from scripts import gen_ep03_seedance as i2v
    script = _ep04_script()
    root = _write_script(tmp_path, script)
    # 造关键帧
    os.makedirs(os.path.join(root, "kf"), exist_ok=True)
    open(os.path.join(root, "kf", "ep04_shot01_123.png"), "w").close()

    calls = {}
    with patch("scripts.gen_ep03_seedance.load_script", return_value=script) as _ls, \
         patch("scripts.gen_ep03_seedance.generate") as _gen:
        os.makedirs(os.path.join(root, "out"), exist_ok=True)
        rc = i2v.main.__wrapped__ if hasattr(i2v.main, "__wrapped__") else i2v.main
        # 直接构造 args 调用 main 会 parse sys.argv, 用 monkeypatch argv
        _run_main(i2v, ["--ep", "04", "--shots", "1",
                        "--kf-dir", os.path.join(root, "kf"),
                        "--out", os.path.join(root, "out")])
        calls["gen"] = _gen
    # generate (provider 统一入口) 收到剧本 motion + FACE_POS
    assert calls["gen"].called
    args = calls["gen"].call_args[0]
    assert "stays misty-blue" in args[1]     # 剧本 motion
    assert "face stable" in args[1]          # FACE_POS 追加
    assert args[2].endswith("ep04_shot01.mp4")


# ---- i2v: line_type 三模式路由 (端到端整合 2026-08-06) ----

def _line_type_script():
    """3 镜: dialog / voiceover / ambient (缺省推断 + 显式标注)."""
    from lib.scriptgen import merge_script
    chars = {"LINXIA": {"fingerprint": "1girl, solo, Linxia, misty-blue top",
                        "voice_id": "female-shaonv"}}
    shots = {
        "1": {"shot": "medium shot 1", "line": "你好,我要说话",
              "motion": "motion1 stays", "speaker": "LINXIA"},                       # 缺省 → dialog
        "2": {"shot": "medium shot 2", "line": "这是画外旁白",
              "motion": "motion2 stays", "speaker": "LINXIA", "line_type": "voiceover"},
        "3": {"shot": "medium shot 3", "line": "",
              "motion": "motion3 stays", "speaker": "LINXIA"},                       # 无 line → ambient
    }
    return merge_script(
        {"title": "三模式", "duration_s": 5, "characters": chars, "shots": shots},
        ep="04", style="st", neg="ng")


def test_i2v_h3_line_type_routing(tmp_path):
    """官方六段式 (默认): dialog → 六段式+audio; voiceover → 沉默+⛔无旁白字样; ambient → 沉默+soundscape."""
    from scripts import gen_ep03_seedance as i2v
    script = _line_type_script()
    root = str(tmp_path)
    kf_dir = os.path.join(root, "kf"); os.makedirs(kf_dir, exist_ok=True)
    out_dir = os.path.join(root, "out"); os.makedirs(out_dir, exist_ok=True)
    for s in (1, 2, 3):
        open(os.path.join(kf_dir, f"ep04_shot{s:02d}_1.png"), "w").close()

    gen_calls = []
    with patch("scripts.gen_ep03_seedance.load_script", return_value=script), \
         patch("scripts.gen_ep03_seedance.generate",
               side_effect=lambda src, prompt, out, **kw: gen_calls.append(
                   {"src": src, "prompt": prompt, "out": out, "kw": kw})), \
         patch("scripts.gen_ep03_seedance.synthesize",
               side_effect=lambda text, out, **kw: open(out, "w").write("x")), \
         patch("scripts.gen_ep03_seedance._pad_audio_to_duration"), \
         patch("lib.video_gen.PROVIDER", "h3_local"):
        _run_main(i2v, ["--ep", "04", "--shots", "1,2,3",
                        "--kf-dir", kf_dir, "--out", out_dir])

    assert len(gen_calls) == 3
    # shot1 dialog: audio + 六段式 + <d>[中文]台词</d> + 无旧中文指令
    assert gen_calls[0]["kw"].get("audio")
    p0 = gen_calls[0]["prompt"]
    assert "subject_definitions:" in p0
    assert "<d>[中文]你好,我要说话</d>" in p0      # 台词保留中文
    assert "overall_soundscape:" in p0
    assert "misty-blue top" in p0                    # 服装 (subject 定义流入)
    assert "用中文清晰自然地念出台词" not in p0      # 官方格式无旧指令
    # shot2 voiceover: 无 audio + 沉默 + ⛔ 无旁白字样
    assert "audio" not in gen_calls[1]["kw"]
    p1 = gen_calls[1]["prompt"]
    assert "stays completely silent" in p1
    assert not any(w in p1.lower() for w in ("voiceover", "narration", "off-screen"))
    # shot3 ambient: 无 audio + 沉默 + soundscape
    assert "audio" not in gen_calls[2]["kw"]
    p2 = gen_calls[2]["prompt"]
    assert "stays completely silent" in p2
    assert "overall_soundscape:" in p2


def test_i2v_h3_line_type_routing_legacy(tmp_path):
    """--prompt-style legacy 保持旧中文指令格式 (回退)."""
    from scripts import gen_ep03_seedance as i2v
    script = _line_type_script()
    root = str(tmp_path)
    kf_dir = os.path.join(root, "kf"); os.makedirs(kf_dir, exist_ok=True)
    out_dir = os.path.join(root, "out"); os.makedirs(out_dir, exist_ok=True)
    for s in (1, 2, 3):
        open(os.path.join(kf_dir, f"ep04_shot{s:02d}_1.png"), "w").close()

    gen_calls = []
    with patch("scripts.gen_ep03_seedance.load_script", return_value=script), \
         patch("scripts.gen_ep03_seedance.generate",
               side_effect=lambda src, prompt, out, **kw: gen_calls.append(
                   {"src": src, "prompt": prompt, "out": out, "kw": kw})), \
         patch("scripts.gen_ep03_seedance.synthesize",
               side_effect=lambda text, out, **kw: open(out, "w").write("x")), \
         patch("scripts.gen_ep03_seedance._pad_audio_to_duration"), \
         patch("lib.video_gen.PROVIDER", "h3_local"):
        _run_main(i2v, ["--ep", "04", "--shots", "1,2,3",
                        "--kf-dir", kf_dir, "--out", out_dir,
                        "--prompt-style", "legacy"])

    assert len(gen_calls) == 3
    # shot1 dialog: 旧中文说话指令 + 台词内嵌 + 服装前置
    assert gen_calls[0]["kw"].get("audio")
    assert "用中文清晰自然地念出台词" in gen_calls[0]["prompt"]
    assert "你好,我要说话" in gen_calls[0]["prompt"]
    assert "misty-blue top" in gen_calls[0]["prompt"]
    # shot2 voiceover: 无 audio + 闭嘴 prompt
    assert "audio" not in gen_calls[1]["kw"]
    assert "stays completely silent" in gen_calls[1]["prompt"]
    # shot3 ambient: 无 audio + 无说话/闭嘴后缀
    assert "audio" not in gen_calls[2]["kw"]
    assert "用中文清晰自然地念出台词" not in gen_calls[2]["prompt"]
    assert "stays completely silent" not in gen_calls[2]["prompt"]


def _dual_script(tmp_path, chars, shots):
    """双角色剧本 (真实 ref 文件), 返回 merge_script 后的 script."""
    from lib.scriptgen import merge_script
    linxia_ref = os.path.join(str(tmp_path), "linxia.png"); open(linxia_ref, "w").close()
    client_k_ref = os.path.join(str(tmp_path), "client_k.png"); open(client_k_ref, "w").close()
    return merge_script(
        {"title": "双人", "duration_s": 5, "characters": chars, "shots": shots},
        ep="05", style="st", neg="ng",
        char_overrides={"LINXIA": {"face_ref": linxia_ref},
                        "client_k": {"face_ref": client_k_ref}})


def _dual_script_fixture(tmp_path):
    chars = {
        "LINXIA": {"fingerprint": "1girl, solo, Linxia, misty-blue top",
                   "voice_id": "female-shaonv"},
        "client_k": {"fingerprint": "1girl, solo, client_k, red dress",
                 "voice_id": "female-wenrou"},
    }
    shots = {"1": {"shot": "medium shot both", "line": "你练得怎么样？",
                   "motion": "motion1 stays", "speaker": "LINXIA"},
             "2": {"shot": "medium shot both", "line": "还不错，继续！",
                   "motion": "motion2 stays", "speaker": "client_k"}}
    return _dual_script(tmp_path, chars, shots)


def _run_dual(i2v, script, root, extra_argv):
    kf_dir = os.path.join(root, "kf"); os.makedirs(kf_dir, exist_ok=True)
    out_dir = os.path.join(root, "out"); os.makedirs(out_dir, exist_ok=True)
    for s in (1, 2):
        open(os.path.join(kf_dir, f"ep05_shot{s:02d}_1.png"), "w").close()
    gen_calls = []
    with patch("scripts.gen_ep03_seedance.load_script", return_value=script), \
         patch("scripts.gen_ep03_seedance.generate",
               side_effect=lambda src, prompt, out, **kw: gen_calls.append(
                   {"kw": kw, "prompt": prompt})), \
         patch("scripts.gen_ep03_seedance.synthesize",
               side_effect=lambda text, out, **kw: open(out, "w").write("x")), \
         patch("scripts.gen_ep03_seedance._pad_audio_to_duration"), \
         patch("lib.video_gen.PROVIDER", "h3_local"):
        _run_main(i2v, ["--ep", "05", "--shots", "1,2",
                        "--kf-dir", kf_dir, "--out", out_dir] + extra_argv)
    return gen_calls


def test_i2v_h3_dual_dialog_routing(tmp_path):
    """官方六段式 (默认): dual 双角色 ref + 说话人音频 + 六段式 subject definitions."""
    from scripts import gen_ep03_seedance as i2v
    script = _dual_script_fixture(tmp_path)
    gen_calls = _run_dual(i2v, script, str(tmp_path), [])

    assert len(gen_calls) == 2
    # shot1: LINXIA 说话 → dual, 两 ref, audio; 六段式 Subject 1=倾听者 client_k, Subject 2=说话者 LINXIA
    kw1 = gen_calls[0]["kw"]
    assert kw1.get("mode") == "dual"
    assert len(kw1.get("ref_images", [])) == 2
    assert kw1.get("audio")
    p0 = gen_calls[0]["prompt"]
    assert "subject_definitions:" in p0
    assert "<Subject 1> is the person in <Picture 1>, red dress." in p0          # 倾听者 client_k
    assert "<Subject 2> is the person in <Picture 2>, misty-blue top." in p0     # 说话者 LINXIA
    assert "<Audio 1> is the voice-timbre reference for <Subject 2> (S1)" in p0
    assert "<d>[中文]你练得怎么样？</d>" in p0
    assert "mouth closed throughout" in p0          # 倾听者闭嘴沉默
    assert "用中文清晰自然地念出台词" not in p0     # 官方格式无旧指令
    # ref 顺序: [倾听者, 说话者]
    refs0 = kw1["ref_images"]
    assert "client_k.png" in refs0[0] and "linxia.png" in refs0[1]
    # shot2: client_k 说话 → Subject 1=倾听者 LINXIA, Subject 2=说话者 client_k
    kw2 = gen_calls[1]["kw"]
    assert kw2.get("mode") == "dual"
    p1 = gen_calls[1]["prompt"]
    assert "<Subject 2> is the person in <Picture 2>, red dress." in p1          # 说话者 client_k
    assert "<d>[中文]还不错，继续！</d>" in p1
    refs1 = kw2["ref_images"]
    assert "linxia.png" in refs1[0] and "client_k.png" in refs1[1]


def test_i2v_h3_dual_dialog_routing_legacy(tmp_path):
    """--prompt-style legacy 双人对话保持旧中文格式."""
    from scripts import gen_ep03_seedance as i2v
    script = _dual_script_fixture(tmp_path)
    gen_calls = _run_dual(i2v, script, str(tmp_path), ["--prompt-style", "legacy"])

    assert len(gen_calls) == 2
    kw1 = gen_calls[0]["kw"]
    assert kw1.get("mode") == "dual"
    assert len(kw1.get("ref_images", [])) == 2
    assert kw1.get("audio")
    p0 = gen_calls[0]["prompt"]
    assert "对应【倾听者：client_k" in p0    # Picture 1 = 倾听者 (ref0)
    assert "对应【说话人：LINXIA" in p0
    assert "双人无遮挡并列" in p0
    assert "绝对只说 1 次" in p0
    assert "嘴巴始终紧闭" in p0
    refs0 = kw1["ref_images"]
    assert "client_k.png" in refs0[0] and "linxia.png" in refs0[1]


# ---- voice: 从剧本读 voice_id ----

def test_voice_uses_script_voice_id(tmp_path):
    """voice 脚本: voice_id 从剧本 characters[speaker], 台词从 shots[N].line."""
    from scripts import voice_ep03 as voice
    script = _ep04_script()
    root = _write_script(tmp_path, script)
    i2v_dir = os.path.join(root, "i2v")
    os.makedirs(i2v_dir, exist_ok=True)
    open(os.path.join(i2v_dir, "ep04_shot01.mp4"), "w").close()

    calls = {}
    with patch("scripts.voice_ep03.load_script", return_value=script), \
         patch("scripts.voice_ep03.dur", return_value=5.0), \
         patch("scripts.voice_ep03.synthesize") as _syn:
        _run_main(voice, ["--ep", "04", "--shots", "1",
                          "--i2v-dir", i2v_dir,
                          "--out", os.path.join(root, "voice")])
        calls["syn"] = _syn
    assert calls["syn"].called
    call = calls["syn"].call_args
    assert "台词1" in call.args[0]           # 剧本 line
    assert call.kwargs.get("voice") == "female-shaonv"  # 剧本 voice_id (kwargs)


# ---- make: 从剧本读 line ----

def test_make_uses_script_line(tmp_path):
    """make 脚本: 字幕文案从剧本 shots[N].line, 输出 ep04_real.mp4."""
    from scripts import make_ep03_real as make
    script = _ep04_script()
    root = _write_script(tmp_path, script)
    i2v_dir = os.path.join(root, "i2v")
    os.makedirs(i2v_dir, exist_ok=True)
    open(os.path.join(i2v_dir, "ep04_shot01.mp4"), "w").close()
    voice_dir = os.path.join(root, "voice")
    os.makedirs(voice_dir, exist_ok=True)
    open(os.path.join(voice_dir, "voice_shot01.mp3"), "w").close()

    class _CP:
        returncode = 0
        stdout = b"5.0"

    calls = {}
    with patch("scripts.make_ep03_real.load_script", return_value=script), \
         patch("scripts.make_ep03_real.latest", return_value=os.path.join(i2v_dir, "ep04_shot01.mp4")), \
         patch("subprocess.run", return_value=_CP()) as _run, \
         patch("subprocess.check_output", return_value=b"5.0"), \
         patch("make_ep01.render_subtitle") as _sub, \
         patch("make_ep01.mux_seg_audio"), \
         patch("make_ep01.concat_xfade_av"), \
         patch("make_ep01.probe_dur", return_value=5.0):
        _run_main(make, ["--ep", "04", "--shots", "1",
                         "--i2v-dir", i2v_dir,
                         "--voice-dir", voice_dir,
                         "--out", os.path.join(root, "ep04_real.mp4")])
        calls["sub"] = _sub
        calls["run"] = _run
    assert calls["sub"].called
    text_arg = calls["sub"].call_args[0][0]
    assert "台词1" in text_arg               # 剧本 line
    assert calls["run"].called               # ffmpeg 被 mock 掉 (避免真跑)


# ---- 尾帧衔接 (2026-08-15 P0) ----

def test_i2v_h3_tail_link(tmp_path):
    """--tail-link: 单人无对白镜 (ambient shot3) 用上镜输出尾帧作首帧; 无上一镜回退 keyframe."""
    from scripts import gen_ep03_seedance as i2v
    script = _line_type_script()
    root = str(tmp_path)
    kf_dir = os.path.join(root, "kf"); os.makedirs(kf_dir, exist_ok=True)
    out_dir = os.path.join(root, "out"); os.makedirs(out_dir, exist_ok=True)
    for s in (1, 2, 3):
        open(os.path.join(kf_dir, f"ep04_shot{s:02d}_1.png"), "w").close()
    # 造 shot02 输出 (tail-link 从中取尾帧)
    open(os.path.join(out_dir, "ep04_shot02.mp4"), "w").close()

    gen_calls = []
    with patch("scripts.gen_ep03_seedance.load_script", return_value=script), \
         patch("scripts.gen_ep03_seedance.generate",
               side_effect=lambda src, prompt, out, **kw: gen_calls.append(
                   {"src": src, "prompt": prompt, "out": out, "kw": kw})), \
         patch("scripts.gen_ep03_seedance._extract_last_frame") as _elf, \
         patch("scripts.gen_ep03_seedance.synthesize",
               side_effect=lambda text, out, **kw: open(out, "w").write("x")), \
         patch("scripts.gen_ep03_seedance._pad_audio_to_duration"), \
         patch("lib.video_gen.PROVIDER", "h3_local"):
        # 只跑 shot3 (ambient): 上一镜 shot02 有 mp4 → 尾帧衔接
        _run_main(i2v, ["--ep", "04", "--shots", "3",
                        "--kf-dir", kf_dir, "--out", out_dir, "--tail-link"])

    assert len(gen_calls) == 1
    shot3 = gen_calls[0]
    assert "_tail_02.png" in shot3["src"]          # 用了尾帧 png
    assert _elf.called                              # 调用了尾帧提取
    assert "ep04_shot02.mp4" in _elf.call_args[0][0]  # 提取的是 shot02.mp4 尾帧


def test_i2v_h3_no_tail_link_uses_keyframe(tmp_path):
    """默认 (无 --tail-link): 即使有上镜 mp4 也用 keyframe."""
    from scripts import gen_ep03_seedance as i2v
    script = _line_type_script()
    root = str(tmp_path)
    kf_dir = os.path.join(root, "kf"); os.makedirs(kf_dir, exist_ok=True)
    out_dir = os.path.join(root, "out"); os.makedirs(out_dir, exist_ok=True)
    for s in (1, 2, 3):
        open(os.path.join(kf_dir, f"ep04_shot{s:02d}_1.png"), "w").close()
    open(os.path.join(out_dir, "ep04_shot02.mp4"), "w").close()

    gen_calls = []
    with patch("scripts.gen_ep03_seedance.load_script", return_value=script), \
         patch("scripts.gen_ep03_seedance.generate",
               side_effect=lambda src, prompt, out, **kw: gen_calls.append(
                   {"src": src, "prompt": prompt, "out": out, "kw": kw})), \
         patch("scripts.gen_ep03_seedance._extract_last_frame"), \
         patch("scripts.gen_ep03_seedance.synthesize",
               side_effect=lambda text, out, **kw: open(out, "w").write("x")), \
         patch("scripts.gen_ep03_seedance._pad_audio_to_duration"), \
         patch("lib.video_gen.PROVIDER", "h3_local"):
        _run_main(i2v, ["--ep", "04", "--shots", "3",
                        "--kf-dir", kf_dir, "--out", out_dir])

    assert len(gen_calls) == 1
    shot3 = gen_calls[0]
    assert "ep04_shot03_1.png" in shot3["src"]   # 仍 keyframe


def _run_main(module, argv):
    """调模块 main() 但替换 sys.argv (绕 argparse)."""
    import importlib
    old = sys.argv
    sys.argv = ["x"] + argv
    try:
        module.main()
    finally:
        sys.argv = old
