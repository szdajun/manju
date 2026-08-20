"""tests/test_scriptgen.py — 剧本 LLM 引擎 (manju Phase 2, 2026-08-06).

全 mock, 不调真实 API. 验证:
  parse_llm_json 鲁棒性 / build_prompt 结构 / merge_script 补全 /
  gen_script 主入口 (注入 fake ask) / orchestrator 衔接 (load_script 自动探测).
"""
import contextlib
import io
import json
import os
import sys

import pytest

from lib.scriptgen import (
    STYLES, VOICE_DEFAULT, VOICE_OPTIONS,
    build_prompt, gen_script, merge_script, parse_llm_json,
)
from lib.storyboard import MANJU, load_json_script, load_script, validate


# ---- parse_llm_json 鲁棒性 ----

def test_parse_plain():
    assert parse_llm_json('{"a": 1}') == {"a": 1}


def test_parse_fenced():
    assert parse_llm_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_with_prefix_suffix():
    text = '好的，以下是JSON：\n{"a": 1}\n希望对你有帮助'
    assert parse_llm_json(text) == {"a": 1}


def test_parse_trailing_comma():
    assert parse_llm_json('{"a": 1,}') == {"a": 1}


def test_parse_garbage_raises():
    with pytest.raises(ValueError):
        parse_llm_json("not json at all")


def test_parse_nested():
    assert parse_llm_json('{"shots": {"1": {"x": "y"}}}')["shots"]["1"]["x"] == "y"


# ---- build_prompt 结构 ----

def _spec_with_fingerprint():
    return {
        "LINXIA": {
            "name": "林夏",
            "brief": "健身女孩",
            "fingerprint": "1girl, solo, Linxia (林夏), a slim-fit woman, "
                           "wearing a misty-blue top with a white arc logo",
        }
    }


def _spec_brief_only():
    return {"XIAOYU": {"name": "小玉", "brief": "活泼健身女孩"}}  # 无 fingerprint


def test_prompt_contains_few_shot():
    p = build_prompt("t", "title", 5, STYLES["real"]["style"], STYLES["real"]["neg"])
    assert "misty-blue" in p            # 主角指纹 (duet 与 niuma 共用)
    assert "stays misty-blue throughout" in p   # 服装不变式


# ---- 对话织体四律 + 双人 few-shot (2026-08-20, 同日 +缓冲镜律) ----

def test_prompt_contains_texture_rules():
    """织体四律进 prompt: 轮转/接话/反应镜/缓冲镜 + 六镜织体模板."""
    p = build_prompt("t", "title", 5, STYLES["real"]["style"], STYLES["real"]["neg"])
    assert "轮转律" in p
    assert "接话律" in p
    assert "反应镜律" in p
    assert "缓冲镜律" in p                     # 第④律 (2026-08-20 节奏修)
    assert "牛马，明天见" in p                  # sign-off 规则保留


def test_prompt_dual_few_shot_full_episode():
    """few-shot 默认 ep27_duet 全 6 镜 — 织体是跨镜结构, 只看单镜学不到."""
    p = build_prompt("t", "title", 5, STYLES["real"]["style"], STYLES["real"]["neg"])
    few = p.split("【格式示例")[1]
    for sid in ("\"1\"", "\"2\"", "\"3\"", "\"4\"", "\"5\"", "\"6\""):
        assert sid in few                    # 6 镜齐
    assert "LICOACH" in few                  # 双人示例
    assert "\"line_type\": \"ambient\"" in few   # 反应镜/缓冲镜示范 (无台词)
    assert "frame_chars" in few                  # 空镜 []/双人同框示范 (json 多行展开)


def test_prompt_buffer_shot_rule():
    """缓冲镜律: 无人纯环境空镜 frame_chars=[] 与反应镜区分."""
    p = build_prompt("t", "title", 5, STYLES["real"]["style"], STYLES["real"]["neg"])
    assert "frame_chars 留空数组 []" in p      # 显式无人标注
    assert "纯场景画面给观众换气" in p          # 呼吸位语义
    assert "六镜织体参考模板" in p              # 6 镜模板替换 5 镜版


def test_prompt_dual_dialog_unban_rule():
    """双人同框 dialog 解禁但限量: ≤1/集, 默认正反打."""
    p = build_prompt("t", "title", 5, STYLES["real"]["style"], STYLES["real"]["neg"])
    assert "每集 ≤1 个" in p                  # 限量规则
    assert "正反打" in p                      # 单人默认仍标榜
    assert "绝不能让角色在双人画面里说台词" not in p   # 旧全禁令已删


def test_prompt_contains_style_block():
    p = build_prompt("t", "title", 5, STYLES["real"]["style"], STYLES["real"]["neg"])
    assert "photorealistic" in p
    assert "不要输出" in p or "markdown" in p


def test_prompt_known_fingerprint_reused():
    p = build_prompt("t", "title", 5, STYLES["real"]["style"],
                     STYLES["real"]["neg"], _spec_with_fingerprint())
    assert "misty-blue top with a white arc logo" in p  # 指纹逐字嵌入
    assert "逐字复用" in p or "不得改服装" in p


def test_prompt_brief_only_expand():
    p = build_prompt("t", "title", 5, STYLES["real"]["style"],
                     STYLES["real"]["neg"], _spec_brief_only())
    assert "小玉" in p
    assert "展开成完整英文指纹" in p


def test_prompt_no_markdown_instruction():
    p = build_prompt("t", "title", 5, STYLES["real"]["style"], STYLES["real"]["neg"])
    assert "不要输出 markdown" in p
    assert "第一个字符必须是 {" in p


# ---- merge_script 补全 ----

def _llm_subset(n=5):
    chars = {"LINXIA": {"fingerprint": "1girl, solo, Linxia", "voice_id": "female-shaonv"}}
    shots = {}
    for i in range(1, n + 1):
        shots[str(i)] = {"shot": f"medium shot {i}", "line": f"台词{i}",
                         "motion": f"motion {i} stays", "speaker": "LINXIA"}
    return {"title": "测试集", "duration_s": 5, "characters": chars, "shots": shots}


def test_merge_adds_required_fields():
    s = merge_script(_llm_subset(), ep="04", style="st", neg="ng", eye_fix="ef")
    assert s["ep"] == "04"
    assert s["style"] == "st"
    assert s["neg"] == "ng"
    assert s["eye_fix"] == "ef"
    assert "LINXIA" in s["characters"]
    ch = s["characters"]["LINXIA"]
    assert ch["face_ref"] is None
    assert ch["body_ref"] is None
    assert ch["voice_id"] == "female-shaonv"
    assert ch["lora"] is None
    validate(s)


def test_merge_defaults():
    s = merge_script(_llm_subset(), ep="04", style="st", neg="ng")
    assert s["title"] == "测试集"
    assert s["duration_s"] == 5


def test_merge_line_type_passthrough():
    """LLM 标注 line_type 透传; 缺省 None (由 storyboard.line_type 推断)."""
    subset = _llm_subset()
    subset["shots"]["1"]["line_type"] = "voiceover"
    subset["shots"]["2"]["line_type"] = "AMBIENT"     # 大小写归一
    s = merge_script(subset, ep="04", style="st", neg="ng")
    assert s["shots"]["1"]["line_type"] == "voiceover"
    assert s["shots"]["2"]["line_type"] == "ambient"
    assert s["shots"]["3"]["line_type"] is None       # 未标注 → None
    validate(s)


def test_merge_char_overrides():
    ov = {"LINXIA": {"face_ref": "f.png", "body_ref": "b.png", "voice_id": "female-wenrou"}}
    s = merge_script(_llm_subset(), ep="04", style="st", neg="ng", char_overrides=ov)
    ch = s["characters"]["LINXIA"]
    assert ch["face_ref"] == "f.png"
    assert ch["body_ref"] == "b.png"
    assert ch["voice_id"] == "female-wenrou"


def test_merge_key_canonical():
    subset = _llm_subset()
    subset["characters"] = {"linxia": {"fingerprint": "fp", "voice_id": "x"}}
    subset["shots"]["1"]["speaker"] = "linxia"
    s = merge_script(subset, ep="04", style="st", neg="ng")
    assert "LINXIA" in s["characters"]
    assert s["shots"]["1"]["speaker"] == "LINXIA"


def test_merge_voice_id_whitelist_fallback():
    """merge 对白名单外的 voice_id 告警并回退 VOICE_DEFAULT (防 2054 直达 TTS)."""
    subset = _llm_subset(1)
    subset["characters"]["LINXIA"]["voice_id"] = "female-shaonun"  # r3 真跑拼错值
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        s = merge_script(subset, ep="04", style="st", neg="ng")
    assert s["characters"]["LINXIA"]["voice_id"] == VOICE_DEFAULT
    out = buf.getvalue()
    assert "female-shaonun" in out          # 告警含原值
    assert VOICE_DEFAULT in out             # 告警含回退值


def test_merge_voice_id_valid_passthrough():
    """白名单内 voice_id 原样透传, 无告警; 空值缺省回退 VOICE_DEFAULT."""
    subset = _llm_subset(1)
    subset["characters"]["LINXIA"]["voice_id"] = "female-wenrou"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        s = merge_script(subset, ep="04", style="st", neg="ng")
    assert s["characters"]["LINXIA"]["voice_id"] == "female-wenrou"
    assert "voice" not in buf.getvalue()

    subset["characters"]["LINXIA"]["voice_id"] = ""   # 空值 → 默认, 无告警
    with contextlib.redirect_stdout(buf):
        s = merge_script(subset, ep="04", style="st", neg="ng")
    assert s["characters"]["LINXIA"]["voice_id"] == VOICE_DEFAULT
    assert "female-shaonun" not in buf.getvalue()


def test_prompt_shot5_sign_off():
    """镜5 指令含固定 sign-off (spec 拍板), 且与 few-shot 省略场合的说明共存."""
    p = build_prompt(subject="测试", title="测试", n_shots=5,
                     style="s", neg="n", characters_spec=None)
    assert "牛马，明天见。" in p
    assert "sign-off" in p or "收尾词" in p
    assert "不得改写" in p


def test_merge_n_shots_mismatch_raises():
    with pytest.raises(ValueError):
        merge_script(_llm_subset(4), ep="04", style="st", neg="ng", expected_n_shots=5)


def test_merge_missing_shots_raises():
    subset = _llm_subset()
    subset["shots"] = {}
    with pytest.raises(ValueError):
        merge_script(subset, ep="04", style="st", neg="ng")


# ---- gen_script 主入口 (mock ask) ----

def _never_ask(*a, **kw):
    raise AssertionError("dry-run 不应调 API")


def test_dry_run_no_api(tmp_path):
    # 用临时 out_dir 避免撞 output/ep04 已有产物 (Phase 2 真调豆包生成过)
    r = gen_script("subject", "title", ep="04", dry_run=True, ask=_never_ask,
                   out_dir=str(tmp_path))
    assert r["dry_run"] is True
    assert "输出格式" in r["prompt"] or "JSON" in r["prompt"]
    assert not os.path.exists(r["path"])


def test_gen_full_mocked(tmp_path):
    subset = _llm_subset(5)

    def _fake_ask(prompt, **kw):
        return "```json\n" + json.dumps(subset, ensure_ascii=False) + "\n```"

    r = gen_script("subject", "标题", n_shots=5, ep="04",
                   out_dir=str(tmp_path), ask=_fake_ask)
    assert r["ep"] == "04"
    assert len(r["shots"]) == 5
    # 落盘 + 可读回
    p = os.path.join(str(tmp_path), "script.json")
    assert os.path.exists(p)
    back = load_json_script(p)
    assert back == r


def test_gen_raises_on_nonjson(tmp_path):
    def _fake_ask(prompt, **kw):
        return "这不是 JSON"

    with pytest.raises(ValueError):
        gen_script("subject", "标题", ep="04", out_dir=str(tmp_path), ask=_fake_ask)


# ---- orchestrator 衔接 (零改动证明) ----

def test_load_script_autodetect_llm_json(monkeypatch, tmp_path):
    """gen_script 产物落 output/ep04/script.json → load_script 零改动自动探测."""
    import lib.storyboard as sb
    subset = _llm_subset(3)
    s = merge_script(subset, ep="04", style=STYLES["real"]["style"],
                     neg=STYLES["real"]["neg"])
    ep_dir = os.path.join(str(tmp_path), "output", "ep04")
    os.makedirs(ep_dir, exist_ok=True)
    with open(os.path.join(ep_dir, "script.json"), "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False)

    monkeypatch.setattr(sb, "MANJU", str(tmp_path))
    loaded = load_script("04")
    assert loaded["ep"] == "04"
    assert len(loaded["shots"]) == 3


def test_run_ep_dry_run_consumes_llm_json(monkeypatch, tmp_path, capsys):
    """run_ep CLI 端到端吃 LLM JSON 剧本 (monkeypatch 注册 ep04)."""
    import lib.storyboard as sb
    import lib.orchestrator as orch
    import scripts.run_ep as run_ep

    # 造 ep04 JSON 剧本
    subset = _llm_subset(3)
    s = merge_script(subset, ep="04", style=STYLES["real"]["style"],
                     neg=STYLES["real"]["neg"])
    ep_dir = os.path.join(str(tmp_path), "output", "ep04")
    os.makedirs(ep_dir, exist_ok=True)
    with open(os.path.join(ep_dir, "script.json"), "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False)
    monkeypatch.setattr(sb, "MANJU", str(tmp_path))

    # 注册 ep04 (模拟 Phase 3 新集脚本)
    monkeypatch.setitem(orch.EP_SCRIPTS, "04", orch.EP_SCRIPTS["03"])

    monkeypatch.setattr(sys, "argv", ["run_ep.py", "--ep", "04", "--dry-run"])
    with pytest.raises(SystemExit) as exc:
        run_ep.main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "ep04" in out or "04" in out


# ---- 牛马梗版 few-shot 素材测试 ----

def test_niuma_fewshot_loads():
    """牛马梗 few-shot 素材: 文件存在, 5 镜齐, 每镜有四字段, 台词含牛马梗关键词."""
    import json as _json
    path = os.path.join(MANJU, "scripts", "ep03_niuma.json")
    with open(path, encoding="utf-8") as f:
        d = _json.load(f)
    shots = d["shots"]
    assert len(shots) == 5
    niuma_words = ("KPI", "加班", "老板", "工资", "上班", "打工", "周报", "同事", "牛马", "领导")
    all_lines = " ".join(s["line"] for s in shots.values())
    assert any(w in all_lines for w in niuma_words), "台词必须含打工牛马元素"
    for s in shots.values():
        assert s.get("shot") and s.get("line") and s.get("motion") and s.get("speaker")
        # 服装不变式三句 (与 ep03 指纹一致)
        assert "misty-blue top stays misty-blue" in s["motion"]
    # 镜1 冲突前置: 非宣言式开场
    first = shots["1"]["line"]
    banned = ("我要做", "我要成为", "从今天开始")
    assert not any(b in first for b in banned), f"镜1 禁宣言式开场: {first}"


def test_prompt_narrative_structure():
    """prompt 含完播率叙事结构节 + 牛马梗引导 + 新 few-shot 指纹."""
    p = build_prompt(subject="测试", title="测试", n_shots=5,
                     style="s", neg="n", characters_spec=None)
    # 叙事结构节
    assert "冲突前置" in p
    assert "追更钩" in p or "下集悬念" in p
    assert "牛马" in p
    # 台词趣味规范
    assert "截图" in p or "发朋友" in p
    # few-shot 已换血 (niuma 台词进 prompt, 老版宣言台词已出)
    assert "KPI" in p
    assert "真正自律的人" not in p
