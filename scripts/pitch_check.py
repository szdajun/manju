# -*- coding: utf-8 -*-
"""scripts/pitch_check.py — 成片音轨基频检测 (manju-accept 验收三件套之三).

用途: 判定成片/片段里说话人声线性别 (女 200-280Hz / 男 70-180Hz, 180 分界)。
出处: 2026-08-20 ep26 镜2 说话人错位事故 — H3 重合成语音性别错, 靠基频检测实锤。

用法:
  python scripts/pitch_check.py output/epXX_assemble/epXX_real.mp4
  python scripts/pitch_check.py video.mp4 --window 5.5-9.8        # 只测时间窗 (单镜)
  python scripts/pitch_check.py video.mp4 --window 0-5,4.8-10,19.5-24  # 多窗逐镜

 ffmpeg 路径: 默认 ~/ffmpeg/ffmpeg.exe, MANJU_FFMPEG 环境变量可覆盖。
原理: 16kHz 单声道重采样 → 自相关 → 70-400Hz 人声区找峰 → 中值平滑。
依赖: manju venv (numpy) + ffmpeg。
"""
import os
import subprocess
import sys

import numpy as np

FFMPEG = os.environ.get("MANJU_FFMPEG", "~/ffmpeg/ffmpeg.exe")
SR = 16000
F_MIN, F_MAX = 70, 400      # 人声基频搜索区 (Hz)
FEMALE_BELOW = 180          # >180Hz 判女声 (成年男声 70-180, 女声 200-280)


def _decode(path: str, ss: float | None, to: float | None) -> np.ndarray:
    cmd = [FFMPEG, "-v", "error"]
    if ss is not None:
        cmd += ["-ss", f"{ss}"]
    if to is not None:
        cmd += ["-to", f"{to}"]
    cmd += ["-i", path, "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError(f"ffmpeg 解码失败: {r.stderr.decode(errors='replace')[:200]}")
    return np.frombuffer(r.stdout, dtype=np.int16).astype(float)


def pitch_hz(w: np.ndarray) -> float:
    """自相关基频 (70-400Hz 峰). 过短音频 → ValueError."""
    if len(w) < SR // 2:
        raise ValueError(f"音频过短 ({len(w)/SR:.2f}s), 至少 0.5s")
    w = w / (np.abs(w).max() + 1e-9)
    ac = np.correlate(w, w, "full")[len(w):]
    ac /= ac[0] + 1e-12
    lo, hi = int(SR / F_MAX), int(SR / F_MIN)
    lag = lo + int(np.argmax(ac[lo:hi]))
    return SR / lag


def judge(f0: float) -> str:
    return "女" if f0 > FEMALE_BELOW else "男"


def _parse_windows(spec: str) -> list[tuple[float, float] | None]:
    """'5.5-9.8,0-5' → [(5.5,9.8),(0,5)]; '*' → [None] (全片)."""
    if spec in ("*", "all"):
        return [None]
    out = []
    for part in spec.split(","):
        a, b = part.split("-")
        out.append((float(a), float(b)))
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    path = args[0]
    win_spec = "all"
    if "--window" in args:
        win_spec = args[args.index("--window") + 1]
    if not os.path.exists(path):
        print(f"[ERR] 文件不存在: {path}")
        return 1
    if not os.path.exists(FFMPEG):
        print(f"[ERR] ffmpeg 不存在: {FFMPEG} (MANJU_FFMPEG 环境变量可覆盖)")
        return 1

    bad = 0
    for win in _parse_windows(win_spec):
        try:
            ss, to = (None, None) if win is None else win
            w = _decode(path, ss, to)
            f0 = pitch_hz(w)
            label = judge(f0)
            rng = f"({ss}-{to}s)" if win else "(全片)"
            print(f"{path} {rng}: F0={f0:.0f}Hz → {label}声")
        except (ValueError, RuntimeError) as e:
            print(f"{path} 窗 {win}: [ERR] {e}")
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main() or 0)
