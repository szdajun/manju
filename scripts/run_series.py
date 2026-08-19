#!/usr/bin/env python3
"""run_series.py — 多集批量出片 (循环复用 run_ep 编排, 失败隔离 + 汇总).

用法:
  uv run --no-sync python scripts/run_series.py --eps 20-22
  uv run --no-sync python scripts/run_series.py --retry-only          # 重跑上次失败集
  uv run --no-sync python scripts/run_series.py --eps 20-25 --only kf,i2v
  uv run --no-sync python scripts/run_series.py --eps 20 --dry-run    # 预览计划

状态文件: output/series_runs/<时间戳>.json (每集 status/stage/final/seconds)。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.orchestrator import EP_SCRIPTS  # noqa: E402
from lib.series_runner import (  # noqa: E402
    MANJU, failed_eps, latest_state, parse_eps, run_series)

DEFAULT_STATE_DIR = os.path.join(MANJU, "output", "series_runs")


def _utf8_stdout() -> None:
    """Windows GBK 控制台/管道打不出 ✅❌ → 强制 utf-8 (失败则 replace 不炸)."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def _preflight(eps: list[str]) -> None:
    """打印缺剧本的已注册集 (提示, 不禁跑)."""
    missing = [ep for ep in eps
               if ep in EP_SCRIPTS
               and not os.path.exists(os.path.join(MANJU, "output", f"ep{ep}", "script.json"))]
    if missing:
        print(f"[预检] ⚠ 以下集缺 output/ep{{N}}/script.json (会失败, 建议先 novel_to_series): "
              + ",".join(missing), flush=True)


def main(argv=None) -> int:
    _utf8_stdout()
    ap = argparse.ArgumentParser(description="多集批量出片: 逐集 kf→i2v→verify→voice→assemble, 失败隔离")
    ap.add_argument("--eps", default=None, help="集数: '20-25' / '20,21' / '20-22,25' (与 --retry-only 二选一)")
    ap.add_argument("--retry-only", action="store_true",
                    help="只重跑上次运行 (最新状态文件) 里失败的集")
    ap.add_argument("--only", default=None, help="只跑这些阶段 (逗号分隔, 同 run_ep)")
    ap.add_argument("--skip", default=None, help="跳过这些阶段")
    ap.add_argument("--shots", default=None, help="只跑这些镜")
    ap.add_argument("--force", action="store_true", help="忽略增量跳过, 全部重跑")
    ap.add_argument("--scene-ref", default=None, help="场景参考图 (kf 第3 IPAdapter)")
    ap.add_argument("--weight-scene", type=float, default=None)
    ap.add_argument("--continue-on-error", action="store_true",
                    help="单集内阶段失败也继续后续阶段 (同 run_ep)")
    ap.add_argument("--state-dir", default=DEFAULT_STATE_DIR, help="状态文件目录")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划, 不执行不落盘")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    if not args.eps and not args.retry_only:
        ap.error("需要 --eps 或 --retry-only")
    if args.retry_only:
        state = latest_state(args.state_dir)
        if not state:
            print("[retry-only] 无历史状态文件, 无可重跑", file=sys.stderr)
            return 0
        eps = failed_eps(state)
        if not eps:
            print("[retry-only] 上次运行无失败集", file=sys.stderr)
            return 0
        print(f"[retry-only] 重跑上次失败集: {','.join(eps)}", flush=True)
    else:
        eps = parse_eps(args.eps)

    _preflight(eps)
    summary = run_series(
        eps,
        dry_run=args.dry_run, verbose=args.verbose,
        state_dir=args.state_dir,
        shots=args.shots,
        only={s.strip() for s in args.only.split(",")} if args.only else (),
        skip={s.strip() for s in args.skip.split(",")} if args.skip else (),
        force=args.force, scene_ref=args.scene_ref,
        weight_scene=args.weight_scene,
        continue_on_error=args.continue_on_error)

    print("\n" + "=" * 60)
    print("批量汇总")
    print("=" * 60)
    for r in summary["results"]:
        mark = {"ok": "✅", "fail": "❌", "skipped": "⏭️"}[r["status"]]
        extra = f" → {r['final']}" if r["final"] else (f" ({r['stage']})" if r["stage"] else "")
        print(f"  {mark} ep{r['ep']}{extra}  {r['seconds']}s")
    print(f"RESULT: ok={summary['n_ok']} fail={summary['n_fail']} skipped="
          f"{len(summary['results']) - summary['n_ok'] - summary['n_fail']}")
    return 1 if summary["n_fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
