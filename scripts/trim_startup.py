# -*- coding: utf-8 -*-
"""裁掉 Wan-Animate 输出开头 N 帧 (治"开头特写/大小跳").

根因 (实测 + 社区 research 2026-08-11): Wan-Animate 首帧从 reference_image 初始化
(连尺度/构图都跟参考). client_a四视图参考照人物占满画面 (h_frac~0.96) vs 驱动半身 (~0.50)
→ 输出 f0-f1 是近景特写 (实测 0.96), f2 起 pose 接管才回驱动尺度 (0.50, 稳定匹配).
即"开头扑满画面再退回半身"= 参考照(0.96)→pose(0.50) 的尺度交接.

社区常态: Wan2.2 开头几帧已知不稳 (r/comfyui, Comfy-Org #9091), 标准修法 = 裁开头几帧.
上游 "bad first latent" 修复 (b9fd93c) 不在本仓库 (已最新 origin/main), 拿不到 → 裁帧是
当前最可靠修法. f2+ 实测稳定, 默认裁 2 帧.

跑: /f/wkspace/fitness-video-pipeline/.venv/Scripts/python.exe scripts/trim_startup.py --input <mp4>
"""
import argparse
import os
import subprocess
import sys

FFMPEG = r"~/ffmpeg/ffmpeg.exe"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Wan-Animate 输出 mp4")
    ap.add_argument("--frames", type=int, default=2, help="裁掉开头帧数 (默认 2, 实测 f0-f1 特写)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not os.path.exists(args.input):
        print(f"[ERR] 输入不存在: {args.input}", file=sys.stderr); sys.exit(2)
    stem, ext = os.path.splitext(args.input)
    out = args.out or f"{stem}_t{args.frames}{ext}"
    # output-seek select 帧精确 (per memory ffmpeg-fast-seek-frame-extraction-artifact), 重编码 (短片无所谓)
    vf = f"select='gte(n\\,{args.frames})',setpts=PTS-STARTPTS"
    cmd = [FFMPEG, "-y", "-i", args.input, "-vf", vf, "-an",
           "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", out]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"[done] 裁掉开头 {args.frames} 帧 -> {os.path.basename(out)}")


if __name__ == "__main__":
    main()
