"""lib/series_runner.py — 多集批量编排核心 (循环复用 orchestrator, 失败隔离).

Task 1: parse_eps  规格解析 ('20-22,25' → ["20","21","22","25"])
Task 2: run_series 逐集 build_plan+run_plan; 单集 fail/skip 隔离不炸整批;
                  汇总 + 状态落盘 (output/series_runs/{时间戳}.json);
                  latest_state/failed_eps 供 --retry-only 重读失败集.
"""
from __future__ import annotations

import datetime
import glob
import json
import os
import re
import time

from lib.orchestrator import EP_SCRIPTS, build_plan
from lib.storyboard import load_script

MANJU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_TOKEN = re.compile(r"^(\d+)(?:-(\d+))?$")


def parse_eps(spec: str) -> list[str]:
    """'20-25' / '20,21' / '20-22,25' → 补零排序去重的 ep 列表. 非法 token ValueError."""
    if not spec.strip():
        raise ValueError("eps 不能为空")
    out: set[int] = set()
    for tok in spec.split(","):
        tok = tok.strip()
        m = _TOKEN.match(tok)
        if not m:
            raise ValueError(f"eps token 非法: {tok!r} (期望 N 或 N-M, 逗号分隔)")
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        if b < a:
            raise ValueError(f"eps 区间倒置: {tok!r}")
        out.update(range(a, b + 1))
    return [str(e).zfill(2) for e in sorted(out)]


# build_plan 认的关键字; 其余 stage_opts (如 continue_on_error) 归 run_plan
_BP_KEYS = ("shots", "out_root", "force", "only", "skip", "scene_ref", "weight_scene")


def _bp_kw(stage_opts: dict) -> dict:
    """从 stage_opts (CLI 透传) 挑 build_plan 认的键, 别让 run_plan 吃到陌生 kwarg."""
    return {k: v for k, v in stage_opts.items() if k in _BP_KEYS}


def assemble_path_guess(ep: str, out_root: str | None) -> str:
    """成片路径 (相对 MANJU; 与 orchestrator.assemble_path 同规则).

    跨盘 (如 MANJU 在 F: 而 out_root 在 D:) 时 relpath 抛 ValueError → 回退原路径.
    """
    from lib.orchestrator import assemble_path
    p = assemble_path(ep, out_root)
    try:
        return os.path.relpath(p, MANJU)
    except ValueError:
        return p


def run_series(eps: list[str], *, run_plan_fn=None, load_fn=None,
               dry_run: bool = False, verbose: bool = False,
               state_dir: str | None = None, **stage_opts) -> dict:
    """逐集执行 build_plan+run_plan; 单集失败/skip 隔离; 返回汇总 dict.

    run_plan_fn/load_fn: 可注入 (测试 mock 点), 缺省 orchestrator.run_plan /
      storyboard.load_script.
    state_dir: 状态目录 (缺省 {MANJU}/output/series_runs); dry_run 不落盘.
    stage_opts: 透传 build_plan (shots/only/skip/force/scene_ref/weight_scene/
      out_root) 与 run_plan (continue_on_error).

    返回 {"results": [{ep, status, stage, final, seconds}...],
          "n_ok": int, "n_fail": int};  status ∈ "ok"|"fail"|"skipped".
    """
    if run_plan_fn is None:
        from lib.orchestrator import run_plan as run_plan_fn  # 延迟 import, 允许测试替身
    if load_fn is None:
        load_fn = load_script
    if state_dir is None:
        state_dir = os.path.join(MANJU, "output", "series_runs")

    results = []
    for ep in eps:
        row = {"ep": ep, "status": "ok", "stage": None, "final": None, "seconds": 0.0}
        t0 = time.time()
        try:
            if ep not in EP_SCRIPTS:
                row["status"] = "skipped"
                row["stage"] = "未注册 EP_SCRIPTS"
            else:
                script = load_fn(ep)
                plan = build_plan(script, ep=ep, **_bp_kw(stage_opts))
                rc = run_plan_fn(plan, dry_run=dry_run, verbose=verbose,
                                 continue_on_error=stage_opts.get("continue_on_error", False))
                if rc != 0:
                    row["status"] = "fail"
                    row["stage"] = "run_plan 退出码 " + str(rc)
                else:
                    row["final"] = assemble_path_guess(ep, stage_opts.get("out_root"))
        except Exception as e:  # 单集任何异常 (剧本缺/损坏等) → fail 不炸整批
            row["status"] = "fail"
            row["stage"] = str(e)[:200]
        row["seconds"] = round(time.time() - t0, 1)
        results.append(row)
        print(f"[series] ep{ep}: {row['status']}"
              + (f" ({row['stage']})" if row["stage"] else "")
              + f"  {row['seconds']}s", flush=True)

    summary = {"results": results,
               "n_ok": sum(r["status"] == "ok" for r in results),
               "n_fail": sum(r["status"] == "fail" for r in results)}
    if not dry_run:
        os.makedirs(state_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        with open(os.path.join(state_dir, f"{stamp}.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def latest_state(state_dir: str) -> dict | None:
    """series_runs 目录里时间戳最新的状态文件内容; 空目录 None."""
    files = sorted(glob.glob(os.path.join(state_dir, "*.json")))
    if not files:
        return None
    with open(files[-1], encoding="utf-8") as f:
        return json.load(f)


def failed_eps(state: dict) -> list[str]:
    """状态文件里 status=fail 的 ep 列表 (--retry-only 用)."""
    return [r["ep"] for r in state.get("results", []) if r.get("status") == "fail"]
