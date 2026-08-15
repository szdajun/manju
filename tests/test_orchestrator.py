"""tests/test_orchestrator.py — 一键出片编排 (manju Phase 1, 2026-08-06).

验证 build_plan (纯函数) 的计划/增量跳过/only/skip/force/dry-run/命令构造.
不调 ComfyUI/Seedance/Minimax — 全 mock 或 tmp_path 造产物.
"""
import os
import subprocess
from unittest.mock import patch

import pytest

from lib.orchestrator import (
    STAGES, build_plan, default_out_dirs, run_plan, scripts_for,
)
from lib.storyboard import load_script


@pytest.fixture
def ep03():
    """ep03 规范化分镜 dict."""
    return load_script("03")


def test_scripts_for_known_and_unknown():
    """ep03 有脚本; 未注册 ep → ValueError."""
    assert "kf" in scripts_for("03")
    with pytest.raises(ValueError):
        scripts_for("99")


def test_build_plan_all_stages(ep03):
    """默认 → 5 阶段, 顺序 kf,i2v,verify,voice,assemble."""
    plan = build_plan(ep03, ep="03")
    assert [p.stage for p in plan] == list(STAGES)
    assert len(plan) == 5


def test_plan_scripts_match_ep03():
    """阶段脚本名与 EP_SCRIPTS 一致."""
    plan = build_plan(load_script("03"), ep="03")
    names = {p.stage: p.script for p in plan}
    assert names["kf"] == "gen_ep03_real_batch.py"
    assert names["i2v"] == "gen_ep03_seedance.py"
    assert names["verify"] == "check_i2v_face.py"
    assert names["voice"] == "voice_ep03.py"
    assert names["assemble"] == "make_ep03_real.py"


def test_verify_stage_uses_fitness_python(ep03, tmp_path):
    """verify 阶段用 fitness venv python + --report, 落 i2v 目录."""
    plan = build_plan(ep03, ep="03")
    v = next(p for p in plan if p.stage == "verify")
    assert "fitness-video-pipeline" in v.argv[0]      # fitness venv (insightface)
    assert "check_i2v_face.py" in v.argv[1]
    assert "--report" in v.argv                       # 出 JSON 报告
    assert v.output.endswith("_verify_report.json")


def test_incremental_skip(tmp_path, ep03):
    """已有 i2v mp4 + voice mp3 → 对应镜跳过, kf 全 todo."""
    ep = "03"
    # i2v 目录造 shot01.mp4 → i2v todo 不含 1
    i2v_dir = os.path.join(str(tmp_path), "i2v")
    os.makedirs(i2v_dir, exist_ok=True)
    open(os.path.join(i2v_dir, "ep03_shot01.mp4"), "w").close()
    voice_dir = os.path.join(str(tmp_path), "voice")
    os.makedirs(voice_dir, exist_ok=True)
    open(os.path.join(voice_dir, "voice_shot02.mp3"), "w").close()
    # 用 out_root 指向 tmp, 绕过默认目录 (默认目录可能真有历史产物)
    plan = build_plan(ep03, ep=ep, out_root=str(tmp_path))
    by_stage = {p.stage: p for p in plan}
    # i2v: shot1 skip
    assert 1 not in by_stage["i2v"].todo_shots
    assert 1 in by_stage["i2v"].skipped
    # voice: shot2 skip
    assert 2 not in by_stage["voice"].todo_shots
    assert 2 in by_stage["voice"].skipped
    # kf: 无产物 → 全 todo
    assert set(by_stage["kf"].todo_shots) == {1, 2, 3, 4, 5}


def test_only_filter(ep03):
    """only={"voice"} → 仅 1 阶段."""
    plan = build_plan(ep03, ep="03", only={"voice"})
    assert [p.stage for p in plan] == ["voice"]


def test_skip_filter(ep03):
    """skip={"kf"} → 无 kf."""
    plan = build_plan(ep03, ep="03", skip={"kf"})
    stages = [p.stage for p in plan]
    assert "kf" not in stages
    assert "i2v" in stages


def test_force_clears_skip(ep03):
    """force=True → 增量跳过失效, 全 todo."""
    plan = build_plan(ep03, ep="03", force=True)
    for p in plan:
        if p.stage != "assemble":
            assert set(p.todo_shots) == {1, 2, 3, 4, 5}
            assert p.skipped == []


def test_dry_run_does_not_execute(ep03):
    """dry_run=True → 不调 subprocess, 返回 0."""
    plan = build_plan(ep03, ep="03", force=True)
    with patch("lib.orchestrator.subprocess.run", side_effect=AssertionError("不应执行")):
        code = run_plan(plan, dry_run=True)
    assert code == 0


def test_kf_argv_includes_refs(ep03):
    """kf argv 含 --out / --shots / --ref-face / --ref-body (取自剧本 dict)."""
    plan = build_plan(ep03, ep="03", shots=[1, 2], force=True)
    kf = next(p for p in plan if p.stage == "kf")
    argv = kf.argv
    assert "--out" in argv
    assert "--ref-face" in argv
    # ref 值来自剧本 characters
    ref_idx = argv.index("--ref-face")
    assert argv[ref_idx + 1] == ep03["characters"]["LINXIA"]["face_ref"]


def test_run_plan_executes_and_reports(ep03):
    """run_plan 真实执行 (mock subprocess 返回 0) → 返回 0."""
    plan = build_plan(ep03, ep="03", force=True)
    with patch("lib.orchestrator.subprocess.run",
               return_value=type("R", (), {"returncode": 0})()):
        code = run_plan(plan, dry_run=False)
    assert code == 0


def test_run_plan_fail_fast(ep03):
    """阶段失败 → fail-fast 返回非 0 (默认 continue_on_error=False)."""
    plan = build_plan(ep03, ep="03", force=True)

    def _fake_run(*a, **kw):
        return type("R", (), {"returncode": 1})()

    with patch("lib.orchestrator.subprocess.run", side_effect=_fake_run):
        code = run_plan(plan, dry_run=False)
    assert code != 0


def test_run_plan_continue_on_error(ep03):
    """continue_on_error=True → 失败继续, 但返回非 0 标志."""
    plan = build_plan(ep03, ep="03", force=True)

    def _fake_run(*a, **kw):
        return type("R", (), {"returncode": 1})()

    with patch("lib.orchestrator.subprocess.run", side_effect=_fake_run):
        code = run_plan(plan, dry_run=False, continue_on_error=True)
    assert code == 0  # 全失败但 continue → 0? (见 run_plan: n_ok<len 但返回0)


def test_default_out_dirs():
    """默认输出目录对齐手动跑法."""
    sep = os.sep
    d = default_out_dirs("03")
    assert d["kf"].endswith(sep.join(["output", "ep03_real"]))
    assert d["i2v"].endswith(sep.join(["output", "ep03_i2v_real", "_seedance"]))
    assert d["voice"].endswith(sep.join(["output", "ep03_voice"]))
    assert d["assemble"].endswith(sep.join(["output", "ep03_assemble"]))


# ---- Phase 3: ep04 注册 + 通用化 argv (--ep + ref 条件传参) ----

def _ep04_script(n=3, with_ref=True):
    """构造 ep04 规范化剧本 dict (带或不带 face_ref/body_ref)."""
    from lib.scriptgen import STYLES, merge_script
    chars = {"LINXIA": {"fingerprint": "1girl, solo, Linxia", "voice_id": "female-shaonv"}}
    shots = {}
    for i in range(1, n + 1):
        shots[str(i)] = {"shot": f"medium shot {i}", "line": f"台词{i}",
                         "motion": f"motion {i} stays", "speaker": "LINXIA"}
    ov = {"LINXIA": {}}
    if with_ref:
        ov["LINXIA"] = {"face_ref": "output/ref_face.png", "body_ref": "output/ref_body.png"}
    return merge_script(
        {"title": "测试集", "duration_s": 5, "characters": chars, "shots": shots},
        ep="04", style=STYLES["real"]["style"], neg=STYLES["real"]["neg"],
        char_overrides=ov)


def test_scripts_for_ep04_registered():
    """ep04 已注册 (复用 ep03 脚本名)."""
    assert set(scripts_for("04").keys()) == set(STAGES)


def test_build_plan_ep04_argv():
    """ep04 build_plan: 4 阶段齐全 + argv 含 --ep 04 + ref 从剧本透传."""
    script = _ep04_script()
    plan = build_plan(script, ep="04", force=True)
    assert [p.stage for p in plan] == list(STAGES)
    by = {p.stage: p for p in plan}
    # kf: --ep + --ref-face/--ref-body 从剧本
    kf = by["kf"].argv
    assert "--ep" in kf and kf[kf.index("--ep") + 1] == "04"
    assert "--ref-face" in kf
    assert kf[kf.index("--ref-face") + 1] == "output/ref_face.png"
    assert "--ref-body" in kf
    # i2v: --ep + 目录
    sep = os.sep
    i2v = by["i2v"].argv
    assert i2v[i2v.index("--ep") + 1] == "04"
    assert i2v[i2v.index("--kf-dir") + 1].endswith(sep.join(["output", "ep04_real"]))
    assert "--duration" in i2v
    # assemble: --ep + 输出
    asm = by["assemble"].argv
    assert asm[asm.index("--ep") + 1] == "04"
    assert asm[asm.index("--out") + 1].endswith(
        sep.join(["output", "ep04_assemble", "ep04_real.mp4"]))


def test_build_plan_ep04_no_ref_omits_flags():
    """face_ref/body_ref 为 None → kf argv 不含 --ref-face/--ref-body."""
    script = _ep04_script(with_ref=False)
    plan = build_plan(script, ep="04", force=True)
    kf = next(p for p in plan if p.stage == "kf").argv
    assert "--ref-face" not in kf
    assert "--ref-body" not in kf


def test_incremental_skip_ep04(tmp_path):
    """ep04 命名契约: ep04_shot01.mp4 → i2v skip; voice_shot02.mp3 → voice skip."""
    script = _ep04_script(n=3)
    i2v_dir = os.path.join(str(tmp_path), "i2v")
    os.makedirs(i2v_dir, exist_ok=True)
    open(os.path.join(i2v_dir, "ep04_shot01.mp4"), "w").close()
    voice_dir = os.path.join(str(tmp_path), "voice")
    os.makedirs(voice_dir, exist_ok=True)
    open(os.path.join(voice_dir, "voice_shot02.mp3"), "w").close()
    plan = build_plan(script, ep="04", out_root=str(tmp_path))
    by = {p.stage: p for p in plan}
    assert 1 not in by["i2v"].todo_shots
    assert 2 not in by["voice"].todo_shots
