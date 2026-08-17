#!/usr/bin/env python3
"""scripts/run_ep.py — 一键出片 orchestrator CLI (manju Phase 1, 2026-08-06).

把"关键帧→i2v→配音→组装"4 步手动串, 变成一条命令:
    python scripts/run_ep.py --ep 03                  # 一键全链
    python scripts/run_ep.py --ep 03 --dry-run        # 打印 4 条命令不跑
    python scripts/run_ep.py --ep 03 --only assemble  # 只组装
    python scripts/run_ep.py --ep 03 --skip voice     # 跳过配音
    python scripts/run_ep.py --ep 03 --shots 1,2,3 --force  # 重抽前 3 镜
    python scripts/run_ep.py --ep 03 --script scripts/ep03.json  # 直接吃 JSON 剧本

数据从 lib.storyboard.load_script 读 (自动探测 JSON / Python 模块).
阶段用 subprocess 调现有 CLI 脚本 (零侵入), 增量跳过已有产物.

产物默认落 output/ep{ep}_real / _i2v_real / _voice / _assemble (对齐手动跑法).
"""
import argparse
import os
import sys

MANJU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MANJU)

from lib.orchestrator import (  # noqa: E402
    STAGES, build_plan, run_plan)
from lib.storyboard import load_script  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="一键出片: 剧本 → 关键帧 → i2v → 配音 → 组装")
    ap.add_argument("--ep", default=os.environ.get("MANJU_EP"),
                    help="集数 (如 03; 或 MANJU_EP env)")
    ap.add_argument("--shots", default=None,
                    help="只跑指定镜 (如 1,2,3; 默认剧本全部)")
    ap.add_argument("--out-root", default=None,
                    help="全新输出根目录 (kf/i2v/voice/assemble 子目录; 跳过检测失效)")
    ap.add_argument("--only", default=None,
                    help=f"只跑这些阶段, 逗号分隔 ({','.join(STAGES)}; 支持别名)")
    ap.add_argument("--skip", default=None,
                    help=f"跳过这些阶段, 逗号分隔 ({','.join(STAGES)})")
    ap.add_argument("--script", default=None,
                    help="剧本来源: JSON 文件路径 (--script xx.json) 或模块名 (--script ep05_shots); "
                         "默认自动探测 JSON → lib.ep{ep}_shots")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印待执行命令, 不真正运行")
    ap.add_argument("--force", action="store_true",
                    help="忽略增量跳过, 全部重跑")
    ap.add_argument("--scene-ref", default=None,
                    help="场景参考图 (kf 第 3 IPAdapter 锁背景, 如 assets/scenes/健身房.png)")
    ap.add_argument("--weight-scene", type=float, default=None,
                    help="场景 IPAdapter weight (默认 0.35, 2026-08-17 标定)")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="单阶段失败继续后续阶段 (默认 fail-fast)")
    ap.add_argument("--verbose", action="store_true",
                    help="打印每条命令 argv")
    args = ap.parse_args()

    if not args.ep:
        ap.error("--ep 必填 (或设 MANJU_EP env)")

    only = set(args.only.split(",")) if args.only else ()
    skip = set(args.skip.split(",")) if args.skip else ()

    # 1. 读分镜数据 (自动探测 JSON / Python 模块)
    print(f"[storyboard] 读取 ep{args.ep} 分镜 ...")
    script = load_script(args.ep, source=args.script)
    print(f"  {script.get('title', '')}  {len(script['shots'])} 镜  "
          f"角色: {list(script['characters'].keys())}")

    # 2. 建计划 (纯函数, 增量跳过)
    plan = build_plan(
        script, ep=args.ep,
        shots=args.shots, out_root=args.out_root,
        force=args.force, only=only, skip=skip,
        scene_ref=args.scene_ref, weight_scene=args.weight_scene,
    )

    # 3. 执行
    code = run_plan(plan, dry_run=args.dry_run,
                    verbose=args.verbose,
                    continue_on_error=args.continue_on_error)
    sys.exit(code)


if __name__ == "__main__":
    main()
