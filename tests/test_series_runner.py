"""test_series_runner.py — parse_eps / run_series 多集批量编排核心单测 + CLI 冒烟."""
import json
import os
import tempfile

import pytest

from lib.series_runner import failed_eps, latest_state, parse_eps, run_series


class TestParseEps:
    def test_single(self):
        assert parse_eps("20") == ["20"]

    def test_range(self):
        assert parse_eps("20-25") == ["20", "21", "22", "23", "24", "25"]

    def test_list(self):
        assert parse_eps("22,20,21") == ["20", "21", "22"]  # 排序去重

    def test_mixed(self):
        assert parse_eps("20-22,25") == ["20", "21", "22", "25"]

    def test_pad(self):
        assert parse_eps("3-4") == ["03", "04"]  # 与 EP_SCRIPTS 键规范一致

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="20-x"):
            parse_eps("20-x")
        with pytest.raises(ValueError):
            parse_eps("")
        with pytest.raises(ValueError, match="倒置"):
            parse_eps("25-20")  # 区间倒置 (b < a)
        with pytest.raises(ValueError):
            parse_eps("   ")  # 纯空白


def _fake_script(ep="20"):
    """最小合法剧本 (build_plan 消费形状: characters/shots/duration_s)."""
    return {"title": "t", "duration_s": 5,
            "characters": {"LINXIA": {"fingerprint": "1girl, solo, x",
                                      "face_ref": "f.png", "body_ref": "b.png"}},
            "shots": {"1": {"shot": "s", "line": "l", "motion": "m", "speaker": "LINXIA"}}}


class TestRunSeries:
    def test_all_ok(self, tmp_path, monkeypatch):
        """两集全成功: 顺序执行, 汇总 ok, 状态文件落盘."""
        monkeypatch.setattr("lib.series_runner.MANJU", str(tmp_path))
        calls = []
        results = run_series(
            ["20", "21"],
            run_plan_fn=lambda plan, **kw: (calls.append(len(plan)), 0)[1],
            load_fn=lambda ep: _fake_script(ep),
            state_dir=str(tmp_path / "series_runs"),
        )
        assert calls  # 真跑了
        assert results["n_ok"] == 2 and results["n_fail"] == 0
        assert all(r["status"] == "ok" for r in results["results"])
        state_files = list((tmp_path / "series_runs").glob("*.json"))
        assert len(state_files) == 1
        saved = json.loads(state_files[0].read_text(encoding="utf-8"))
        assert saved["n_ok"] == 2

    def test_unregistered_skipped(self, tmp_path):
        """未注册 EP_SCRIPTS 的集 → skipped 不算失败, 不调 load."""
        loaded = []
        results = run_series(
            ["99"],
            run_plan_fn=lambda plan, **kw: 0,
            load_fn=lambda ep: (loaded.append(ep), _fake_script(ep))[1],
            state_dir=str(tmp_path / "series_runs"),
        )
        assert results["results"][0]["status"] == "skipped"
        assert results["n_fail"] == 0
        assert not loaded

    def test_fail_isolated(self, tmp_path):
        """单集失败 → 记录继续下一集; 汇总 n_fail=1; exit 语义由返回值承载."""
        ran = []

        def fake_run(plan, **kw):
            n_done = len(ran)
            ran.append(1)
            return 1 if n_done == 0 else 0  # 第一集失败, 第二集成功

        results = run_series(
            ["20", "21"],
            run_plan_fn=fake_run,
            load_fn=lambda ep: _fake_script(ep),
            state_dir=str(tmp_path / "series_runs"),
        )
        assert len(ran) == 2  # 第二集没有被第一集失败挡住
        assert results["n_fail"] == 1 and results["n_ok"] == 1

    def test_load_error_is_fail_not_crash(self, tmp_path):
        """剧本缺失/损坏 (load 抛异常) → 该集 fail, 整批不炸."""
        def bad_load(ep):
            raise ValueError(f"ep{ep} 无剧本")

        results = run_series(
            ["20"], run_plan_fn=lambda plan, **kw: 0, load_fn=bad_load,
            state_dir=str(tmp_path / "series_runs"),
        )
        assert results["results"][0]["status"] == "fail"
        assert "无剧本" in results["results"][0]["stage"]


class TestRetryOnly:
    def test_latest_state_and_failed_eps(self, tmp_path):
        d = tmp_path / "series_runs"
        d.mkdir()
        (d / "20260819-100000.json").write_text(
            json.dumps({"results": [{"ep": "20", "status": "fail"}, {"ep": "21", "status": "ok"}]}),
            encoding="utf-8")
        (d / "20260819-110000.json").write_text(
            json.dumps({"results": [{"ep": "20", "status": "ok"}, {"ep": "22", "status": "fail"}]}),
            encoding="utf-8")
        st = latest_state(str(d))
        assert st is not None and failed_eps(st) == ["22"]  # 只看最新一次

    def test_latest_state_empty_dir(self, tmp_path):
        assert latest_state(str(tmp_path)) is None


class TestCli:
    def test_main_dry_run_registered_ep(self, capsys, monkeypatch, tmp_path):
        """--dry-run 单集已注册: 打印计划, exit 0, 不落状态文件.

        密封: 不依赖不入 git 的 output/ep22/script.json — patch 模块级
        load_script (CLI 无法注入 load_fn, 缺省绑定在 lib.series_runner)
        返回 fake script; _preflight 同样走被 patch 的探测 → 不误报.
        """
        import scripts.run_series as rs
        monkeypatch.setattr("lib.series_runner.load_script",
                            lambda ep: _fake_script(ep))
        monkeypatch.setattr(rs, "_preflight", lambda eps: None)  # 真实 output 状态不可知
        state_dir = str(tmp_path / "rs_smoke")
        rc = rs.main(["--eps", "22", "--dry-run", "--state-dir", state_dir])
        assert rc == 0
        out = capsys.readouterr().out
        assert "ep22" in out and "RESULT" in out
        assert not os.path.exists(state_dir) or not os.listdir(state_dir)  # dry-run 不落盘

    def test_main_unregistered_exit_zero_with_skip(self, capsys):
        """未注册集: 汇总 skipped, exit 0 (skipped 不算失败)."""
        import scripts.run_series as rs
        rc = rs.main(["--eps", "99", "--dry-run"])
        assert rc == 0
        assert "skipped" in capsys.readouterr().out


_VALID_SCRIPT = {  # validate 通过的最小剧本 (preflight 真调 load_script, 不是纯存在性)
    "ep": "20", "title": "t", "style": "s", "neg": "n",
    "characters": {"LINXIA": {"fingerprint": "1girl, solo, x", "voice_id": "v"}},
    "shots": {"1": {"shot": "s", "line": "l", "motion": "m", "speaker": "LINXIA"}},
}


class TestPreflight:
    """_preflight 三候选探测 (M1 修复): 与 load_script 回退链一致, 不再误报.

    load_script 的 JSON 候选路径走 lib.storyboard.MANJU (非 rs.MANJU),
    故 monkeypatch 目标是 lib.storyboard.MANJU.
    """

    def test_all_candidates_missing_warns(self, tmp_path, monkeypatch, capsys):
        """①三候选全缺 (output JSON / scripts JSON / lib 模块) → 提示含该 ep."""
        import scripts.run_series as rs
        monkeypatch.setattr("lib.storyboard.MANJU", str(tmp_path))
        # ep21 已注册但: tmp 下无 output/ep21/script.json、无 scripts/ep21.json,
        # 且 lib/ep21_shots.py 本 repo 不存在 → 三候选全缺
        rs._preflight(["21"])
        out = capsys.readouterr().out
        assert "21" in out and "预检" in out

    def test_output_json_candidate_no_warn(self, tmp_path, monkeypatch, capsys):
        """②候选1在 (output/ep{N}/script.json) → 不提示."""
        import scripts.run_series as rs
        monkeypatch.setattr("lib.storyboard.MANJU", str(tmp_path))
        d = tmp_path / "output" / "ep20"
        d.mkdir(parents=True)
        (d / "script.json").write_text(json.dumps(_VALID_SCRIPT), encoding="utf-8")
        rs._preflight(["20"])
        assert "预检" not in capsys.readouterr().out

    def test_scripts_json_candidate_no_warn(self, tmp_path, monkeypatch, capsys):
        """③候选2在 (scripts/ep{N}.json, ep03 现状) → 不提示 (M1 误报场景)."""
        import scripts.run_series as rs
        monkeypatch.setattr("lib.storyboard.MANJU", str(tmp_path))
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "ep03.json").write_text(
            json.dumps(dict(_VALID_SCRIPT, ep="03")), encoding="utf-8")
        rs._preflight(["03"])
        assert "预检" not in capsys.readouterr().out

    def test_lib_module_candidate_no_warn(self, tmp_path, monkeypatch, capsys):
        """④候选3在 (lib.ep{ep}_shots 模块, ep04 现状) → 不提示 (M1 误报场景)."""
        import scripts.run_series as rs
        monkeypatch.setattr("lib.storyboard.MANJU", str(tmp_path))  # tmp 下无任何 JSON
        rs._preflight(["04"])  # 但 lib/ep04_shots.py 真实存在 (入 git)
        assert "预检" not in capsys.readouterr().out
