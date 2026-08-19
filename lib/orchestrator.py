"""lib/orchestrator.py — 一键出片编排 (manju Phase 1, 2026-08-06).

用户拍板: 真人短剧管线, Phase 1 先做 orchestrator — 把"关键帧→i2v→配音→组装"
4 步手动串, 变成 `python scripts/run_ep.py --ep 03` 一条命令.

架构:
  - 数据从 lib.storyboard.load_script 读 (规范化 dict, 与来源解耦)
  - 阶段用 subprocess 调现有 CLI 脚本 (零侵入, 退出码隔离失败)
  - build_plan 纯函数 (输入分镜 dict + 选项 → 输出待执行计划, 不碰系统)
  - run_plan 逐阶段 subprocess.run + 增量跳过 + 汇总

权威: 本文件 STAGES / EP_SCRIPTS / default_out_dirs 是编排真相源.
新增集 = EP_SCRIPTS 加一行; 改阶段顺序 = 改 STAGES.
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
from dataclasses import dataclass, field

MANJU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(MANJU, "scripts")

# 阶段顺序 (固定, 不可调换)
# 2026-08-15 P0: verify = 质量门 (i2v 后跑 check_i2v_face 跨镜人脸一致性, fitness venv)
STAGES = ("kf", "i2v", "verify", "voice", "assemble")
FIT_PY = "../fitness-video-pipeline/.venv/Scripts/python.exe"  # insightface 在 fitness venv
STAGE_ALIASES = {
    "kf": "kf", "keyframes": "kf",
    "i2v": "i2v", "video": "i2v",
    "voice": "voice", "tts": "voice",
    "assemble": "assemble", "make": "assemble",
}

# ep → 各阶段脚本名 (Phase 1 只有 ep03 有完整脚本)
EP_SCRIPTS = {
    "03": {
        "kf": "gen_ep03_real_batch.py",
        "i2v": "gen_ep03_seedance.py",
        "verify": "check_i2v_face.py",   # P0-2 质量门 (fitness venv 跑)
        "voice": "voice_ep03.py",
        "assemble": "make_ep03_real.py",
    },
    "04": {   # Phase 3: 通用化脚本复用, --ep 切剧本
        "kf": "gen_ep03_real_batch.py",
        "i2v": "gen_ep03_seedance.py",
        "verify": "check_i2v_face.py",
        "voice": "voice_ep03.py",
        "assemble": "make_ep03_real.py",
    },
    "20": {   # 2026-08-15 P1/P2: 小说→剧本→定妆图 (LINXIA+李教练), 全链路验证
        "kf": "gen_ep03_real_batch.py",
        "i2v": "gen_ep03_seedance.py",
        "verify": "check_i2v_face.py",
        "voice": "voice_ep03.py",
        "assemble": "make_ep03_real.py",
    },
    "21": {   # 2026-08-15 P1-1: 分集规划 第1集 (系列角色 LINXIA+李教练, 跨集锁定)
        "kf": "gen_ep03_real_batch.py",
        "i2v": "gen_ep03_seedance.py",
        "verify": "check_i2v_face.py",
        "voice": "voice_ep03.py",
        "assemble": "make_ep03_real.py",
    },
    "22": {   # 2026-08-15 P1-1: 分集规划 第2集
        "kf": "gen_ep03_real_batch.py",
        "i2v": "gen_ep03_seedance.py",
        "verify": "check_i2v_face.py",
        "voice": "voice_ep03.py",
        "assemble": "make_ep03_real.py",
    },
    "23": {   # 2026-08-19 多集真跑 (novel_to_series 产, 继承 ep22 资产+声线)
        "kf": "gen_ep03_real_batch.py",
        "i2v": "gen_ep03_seedance.py",
        "verify": "check_i2v_face.py",
        "voice": "voice_ep03.py",
        "assemble": "make_ep03_real.py",
    },
    "24": {   # 2026-08-19 多集真跑
        "kf": "gen_ep03_real_batch.py",
        "i2v": "gen_ep03_seedance.py",
        "verify": "check_i2v_face.py",
        "voice": "voice_ep03.py",
        "assemble": "make_ep03_real.py",
    },
    "25": {   # 2026-08-19 多集真跑
        "kf": "gen_ep03_real_batch.py",
        "i2v": "gen_ep03_seedance.py",
        "verify": "check_i2v_face.py",
        "voice": "voice_ep03.py",
        "assemble": "make_ep03_real.py",
    },
}


def scripts_for(ep: str) -> dict:
    """取该 ep 的 4 阶段脚本映射. 未注册 → ValueError."""
    if ep not in EP_SCRIPTS:
        raise ValueError(
            f"ep{ep} 无生成脚本, 先注册 EP_SCRIPTS (lib/orchestrator.py). "
            f"已注册: {list(EP_SCRIPTS)}")
    return EP_SCRIPTS[ep]


def default_out_dirs(ep: str) -> dict[str, str]:
    """对齐现有手动跑法的目录 (让增量跳过能找到历史产物)."""
    return {
        "kf": os.path.join(MANJU, "output", f"ep{ep}_real"),
        "i2v": os.path.join(MANJU, "output", f"ep{ep}_i2v_real", "_seedance"),
        "verify": os.path.join(MANJU, "output", f"ep{ep}_i2v_real", "_seedance"),  # 报告落 i2v 目录
        "voice": os.path.join(MANJU, "output", f"ep{ep}_voice"),
        "assemble": os.path.join(MANJU, "output", f"ep{ep}_assemble"),
    }


def assemble_path(ep: str, out_root: str | None = None) -> str:
    """最终成片路径. out_root=None → output/ep{ep}_assemble/ep{ep}_real.mp4."""
    if out_root:
        return os.path.join(out_root, "assemble", f"ep{ep}_real.mp4")
    return os.path.join(default_out_dirs(ep)["assemble"], f"ep{ep}_real.mp4")


# ---- 产物存在性 (增量跳过) ----

def _products_exist(patterns: list[str]) -> bool:
    return any(glob.glob(p) for p in patterns)


def kf_done(ep: str, shot: int, out_dir: str) -> bool:
    """关键帧: ep{ep}_shot{s:02d}_*.png 任一存在 → done (生成器每镜多 seed 文件)."""
    return _products_exist([os.path.join(out_dir, f"ep{ep}_shot{shot:02d}_*.png")])


def i2v_done(ep: str, shot: int, out_dir: str) -> bool:
    """i2v: ep{ep}_shot{s:02d}.mp4 存在 → done."""
    return os.path.exists(os.path.join(out_dir, f"ep{ep}_shot{shot:02d}.mp4"))


def voice_done(shot: int, out_dir: str) -> bool:
    """配音: voice_shot{s:02d}.mp3 存在 → done."""
    return os.path.exists(os.path.join(out_dir, f"voice_shot{shot:02d}.mp3"))


def verify_done(ep: str, i2v_dir: str, report_path: str) -> bool:
    """质量门: i2v 输出 ≥1 且报告已生成 → done."""
    if not _products_exist([os.path.join(i2v_dir, f"ep{ep}_shot*.mp4")]):
        return False
    return os.path.exists(report_path)


# ---- plan ----

@dataclass
class StagePlan:
    stage: str                       # "kf"|"i2v"|"voice"|"assemble"
    script: str                      # scripts/ 下的文件名
    argv: list[str]                  # 完整子进程 argv
    todo_shots: list[int] = field(default_factory=list)   # 本阶段需跑的镜
    skipped: list[int] = field(default_factory=list)      # 增量跳过的镜
    output: str = ""                 # 主产物路径 (汇总打印)


def _parse_shots(shots: str | list[int] | None, script: dict) -> list[int]:
    """归一化镜头列表. None → 剧本全部 shot id."""
    if shots is None:
        return sorted(int(s) for s in script["shots"].keys())
    if isinstance(shots, str):
        return [int(x) for x in shots.split(",") if x.strip()]
    return sorted(int(s) for s in shots)


def _resolve_stages(only, skip):
    """解析 only/skip (支持别名) → (选中的阶段 set)."""
    want = set(STAGES)
    if only:
        want = {STAGE_ALIASES.get(s, s) for s in only}
    if skip:
        want -= {STAGE_ALIASES.get(s, s) for s in skip}
    unknown = (want | {STAGE_ALIASES.get(s, s) for s in skip if s in skip}) - set(STAGES)
    if unknown:
        raise ValueError(f"未知阶段: {unknown} (可选 {STAGES})")
    return want


def build_plan(script: dict, *, ep: str,
               shots: str | list[int] | None = None,
               out_root: str | None = None,
               force: bool = False,
               only: set[str] = (),
               skip: set[str] = (),
               scene_ref: str | None = None,
               weight_scene: float | None = None) -> list[StagePlan]:
    """纯函数: 输入分镜 dict + 选项 → 输出待执行计划. 不碰系统/文件变更.

    only/skip 空集默认全阶段. force=True → 跳过检测失效 (全部重跑).
    scene_ref: 场景参考图 (kf 第 3 IPAdapter 锁背景); weight_scene 缺省走 kf 脚本默认 (0.35).
    返回按 STAGES 顺序的 StagePlan 列表.
    """
    ep_norm = str(ep).zfill(2) if str(ep).isdigit() else str(ep)
    ep_scripts = scripts_for(ep_norm)
    dirs = default_out_dirs(ep_norm)
    if out_root:
        dirs = {k: os.path.join(out_root, v) for k, v in {
            "kf": "kf", "i2v": "i2v", "verify": "i2v",
            "voice": "voice", "assemble": "assemble", }.items()}

    shot_ids = _parse_shots(shots, script)
    want = _resolve_stages(only, skip)
    # 主角 (第一个 SPEAKER 或第一个角色)
    main_char = script["characters"].get("LINXIA") or next(
        iter(script["characters"].values()))
    face_ref = (main_char or {}).get("face_ref") or ""
    body_ref = (main_char or {}).get("body_ref") or ""
    duration_s = script.get("duration_s", 5)

    plan = []
    for stage in STAGES:
        if stage not in want:
            continue
        script_name = ep_scripts[stage]
        todo = []
        skipped = []
        out = dirs[stage]

        if stage == "kf":
            for s in shot_ids:
                if force or not kf_done(ep_norm, s, out):
                    todo.append(s)
                else:
                    skipped.append(s)
            shots_arg = ",".join(str(s) for s in (todo or shot_ids))
            argv = [sys.executable, os.path.join(SCRIPTS, script_name),
                    "--ep", ep_norm, "--shots", shots_arg, "--out", out]
            if face_ref:
                argv += ["--ref-face", face_ref]
            if body_ref:
                argv += ["--ref-body", body_ref]
            if scene_ref:
                argv += ["--scene-ref", scene_ref]
                if weight_scene is not None:
                    argv += ["--weight-scene", str(weight_scene)]
        elif stage == "i2v":
            for s in shot_ids:
                if force or not i2v_done(ep_norm, s, out):
                    todo.append(s)
                else:
                    skipped.append(s)
            shots_arg = ",".join(str(s) for s in (todo or shot_ids))
            argv = [sys.executable, os.path.join(SCRIPTS, script_name),
                    "--ep", ep_norm, "--shots", shots_arg,
                    "--kf-dir", dirs["kf"], "--out", out,
                    "--duration", str(duration_s)]
        elif stage == "verify":
            # P0-2 质量门: i2v 后跑 check_i2v_face (fitness venv), 出报告到 i2v 目录.
            # 非阻塞: 报告漂移但不中断 (组装仍继续); --strict 时 exit 非 0 由 run_plan 判.
            report = os.path.join(dirs["i2v"], "_verify_report.json")
            if force or not verify_done(ep_norm, dirs["i2v"], report):
                todo = shot_ids
            else:
                skipped = shot_ids
            argv = [FIT_PY, os.path.join(SCRIPTS, script_name),
                    "--ep", ep_norm, "--i2v-dir", dirs["i2v"],
                    "--report", report]
            out = report
        elif stage == "voice":
            for s in shot_ids:
                if force or not voice_done(s, out):
                    todo.append(s)
                else:
                    skipped.append(s)
            shots_arg = ",".join(str(s) for s in (todo or shot_ids))
            argv = [sys.executable, os.path.join(SCRIPTS, script_name),
                    "--ep", ep_norm, "--shots", shots_arg,
                    "--out", out, "--i2v-dir", dirs["i2v"]]
            if todo:
                argv += ["--skip-existing"]
        else:  # assemble
            out = dirs["assemble"]
            final = os.path.join(out, f"ep{ep_norm}_real.mp4")
            if out_root:
                final = os.path.join(out_root, "assemble", f"ep{ep_norm}_real.mp4")
            if force or not os.path.exists(final):
                todo = shot_ids
            else:
                skipped = shot_ids
            argv = [sys.executable, os.path.join(SCRIPTS, script_name),
                    "--ep", ep_norm,
                    "--shots", ",".join(str(s) for s in shot_ids),
                    "--i2v-dir", dirs["i2v"],
                    "--voice-dir", dirs["voice"],
                    "--out", final]
            out = final

        # todo 空 = 该阶段全部增量跳过 → 不再生成子进程 (避免"全 done"仍重跑:
        # kf 脚本无自身 skip 检测, 会在同目录用新 seed 覆盖式重生成关键帧)
        if not todo and stage != "assemble":
            continue

        plan.append(StagePlan(
            stage=stage, script=script_name, argv=argv,
            todo_shots=todo, skipped=skipped, output=out))
    return plan


def run_plan(plan: list[StagePlan], *, dry_run: bool = False,
             verbose: bool = False, continue_on_error: bool = False) -> int:
    """逐阶段 subprocess.run(plan.argv, cwd=MANJU). 返回退出码 (0 全 ok).

    dry_run=True → 只打印 argv, 绝不执行.
    """
    print("=" * 60)
    print("一键出片计划")
    print("=" * 60)
    n_ok = 0
    for p in plan:
        done = len([s for s in p.todo_shots]) if p.todo_shots else 0
        label = f"[{p.stage:<8}]"
        todo_s = ",".join(str(s) for s in p.todo_shots) if p.todo_shots else "(全 done/skip)"
        skip_s = ",".join(str(s) for s in p.skipped) or "-"
        print(f"{label} todo={todo_s:<12} skip={skip_s:<12} 脚本={p.script}")
        if verbose or dry_run:
            print(f"        cmd: {' '.join(p.argv)}")
        if dry_run:
            continue
        print(f"        >> {p.script} ...", flush=True)
        r = subprocess.run(p.argv, cwd=MANJU)
        if r.returncode != 0:
            print(f"        [FAIL] {p.stage} 退出码 {r.returncode}")
            if not continue_on_error:
                print(f"RESULT: 失败于 {p.stage} (--continue-on-error 可跳过)")
                return r.returncode
        else:
            n_ok += 1
            print(f"        [OK] {p.output}")
    print("-" * 60)
    print(f"RESULT: {n_ok}/{len(plan)} 阶段 OK"
          + (" (dry-run, 未执行)" if dry_run else ""))
    return 0
