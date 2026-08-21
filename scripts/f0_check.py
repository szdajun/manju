"""帧基中位法判声线性别 (2026-08-21 新增).

为什么: `pitch_check.py` 全片自相关对带谐波/背景噪的男声会锁到谐波峰,
误读成女声 (ep29/30/31 验证 — 教练男声 179/165/154Hz 被报成 217-239Hz 假女声).
帧基中位法 (每 20ms 窗自相关) 更稳, 是 manju-accept 声线检查的推荐法.

用法:
    uv run --no-sync python scripts/f0_check.py <mp4> --window <起>-<止>[,<起>-<止>...]
    # 每窗一行: "窗口起止: F0=xxxHz 女声/男声/边界"
判定: 女 >200Hz, 男 <180Hz, 之间为边界 (需听感确认).
"""
import argparse
import subprocess
import sys
import tempfile
import os

import numpy as np


def _extract_audio(mp4: str, start: float, dur: float, out_wav: str):
    ff = os.environ.get("MANJU_FFMPEG", "~/ffmpeg/ffmpeg.exe")
    subprocess.run(
        [ff, "-y", "-loglevel", "error", "-ss", str(start), "-t", str(dur),
         "-i", mp4, "-vn", "-ac", "1", "-ar", "44100", out_wav],
        check=True)


def frame_median_f0(wav_path: str) -> float:
    import wave
    with wave.open(wav_path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        data = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    hop = int(0.02 * sr)
    win = int(0.04 * sr)
    f0s = []
    for st in range(0, len(data) - win, hop):
        seg = data[st:st + win] - np.mean(data[st:st + win])
        ac = np.correlate(seg, seg, "full")[win - 1:]
        lo, hi = int(sr / 300), int(sr / 70)
        if hi - lo < 10 or len(ac) <= hi:
            continue
        pk = np.argmax(ac[lo:hi]) + lo
        f0 = sr / pk
        if 60 < f0 < 320 and ac[pk] > np.max(ac) * 0.3:
            f0s.append(f0)
    return float(np.median(f0s)) if f0s else 0.0


def classify(f0: float) -> str:
    if f0 > 200:
        return "女声"
    if f0 < 180:
        return "男声"
    return "边界(需听感确认)"


def main():
    ap = argparse.ArgumentParser(description="帧基中位法判声线性别")
    ap.add_argument("mp4", help="视频/音频文件")
    ap.add_argument("--window", default=None,
                    help="时间窗, 逗号分隔 '起-止' (如 0.5-4.7,5.8-9.8)")
    args = ap.parse_args()
    windows = []
    if args.window:
        for seg in args.window.split(","):
            if "-" in seg:
                a, b = seg.split("-")
                windows.append((float(a), float(b) - float(a)))
    if not windows:
        windows = [(0.0, 5.0)]

    tmp = os.path.join(tempfile.gettempdir(), "_f0_check_tmp.wav")
    for i, (start, dur) in enumerate(windows):
        _extract_audio(args.mp4, start, dur, tmp)
        f0 = frame_median_f0(tmp)
        print(f"窗{i + 1} {start:.1f}-{start + dur:.1f}s: F0={f0:.0f}Hz → {classify(f0)}")
    try:
        os.remove(tmp)
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
