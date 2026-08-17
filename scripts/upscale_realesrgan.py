# -*- coding: utf-8 -*-
"""Real-ESRGAN 生成后超分 (Wan-Animate 输出 448×672 → 896×1344).

为什么生成后超分而非生成阶段放大: 钉死根因 — 生成画布 ≠ 驱动源尺寸 (drive_dance 448×672)
就鬼影/重影 (见 memory wan-animate-framing-pose-dominates). 所以生成必须锁 448×672,
质量提升走生成后超分. 相关 memory: wan-animate-priorities-not-faceswap.

模型选择 (2026-08-11 vision 实测钉死, 别换):
- **realesr-animevideov3 -s 2** = 默认. 虽名 anime, 实为视频专用模型 (帧间时序一致, 无闪烁),
  对真人脸反而比 x4plus 自然 (x4plus 过锐→塑料感/waxy, face-crop 对比实测). 2x→896×1344 命中目标.
- realesrgan-x4plus = 备选 (更锐但塑料脸 + 逐帧无时序→视频闪烁), 仅当要 4x 单图时用.

流程 (视频帧超分标准三步, 见 tools/realesrgan/README_windows.md):
  1. ffmpeg 抽帧 (qscale 1 无损 jpg, output-seek 安全)
  2. realesrgan-ncnn-vulkan 逐帧超分
  3. ffmpeg 按源帧率重编码 libx264 crf16 yuv420p

跑:
  /f/wkspace/fitness-video-pipeline/.venv/Scripts/python.exe scripts/upscale_realesrgan.py \
      --input output/wan_animate/client_a_448x672_matchsrc_00001__t2.mp4
  # (先裁首2帧再超分; 或对已裁版直接 --input)
  --scale 2 --model realesr-animevideov3  # 默认
  --crf 16  # 越低越高质量; 默认 16 (源是生成产物, 给足码率)
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

_DIM_RE = re.compile(r"(\d{2,5})x(\d{2,5})")

FFMPEG = r"~/ffmpeg/ffmpeg.exe"
REALESRGAN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tools", "realesrgan", "realesrgan-ncnn-vulkan.exe",
)


def probe_fps(path):
    """取源帧率 (超分后按此重编码, 别让 ffmpeg 猜成 25)."""
    # errors=replace: ffmpeg stderr 回显中文路径, GBK 默认解码会崩 (UnicodeDecodeError → None)
    out = subprocess.run(
        [FFMPEG, "-i", path], capture_output=True, text=True, errors="replace"
    ).stderr
    for line in out.splitlines():
        if "Video:" in line and "fps" in line:
            # "..., 16 fps, 16 tbr, ..."
            parts = line.split(",")
            for p in parts:
                p = p.strip()
                if p.endswith("fps") and "tbr" not in p:
                    try:
                        return float(p.split()[0])
                    except (ValueError, IndexError):
                        pass
    return 16.0  # Wan-Animate 默认 16fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Wan-Animate 输出 mp4 (建议先裁首2帧)")
    ap.add_argument("--scale", type=int, default=2, choices=(2, 3, 4), help="放大倍数 (默认 2 → 896×1344)")
    ap.add_argument("--model", default="realesr-animevideov3",
                    help="realesr-animevideov3 (默认, 视频/真人脸自然) | realesrgan-x4plus (4x 备选)")
    ap.add_argument("--crf", type=int, default=16, help="输出质量 (默认 16, 源是生成产物给足码率)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    for p, tag in ((args.input, "输入"), (REALESRGAN, "realesrgan-ncnn-vulkan.exe"), (FFMPEG, "ffmpeg")):
        if not os.path.exists(p):
            print(f"[ERR] {tag}不存在: {p}", file=sys.stderr); sys.exit(2)

    fps = probe_fps(args.input)
    stem, ext = os.path.splitext(args.input)
    # 输出名带分辨率: 从 ffmpeg stderr 的 "Video: ... WxH" 取源尺寸
    # (token 后常带逗号, 故用 regex 而非 split isdigit; errors=replace 防中文路径 GBK 崩)
    probe = subprocess.run([FFMPEG, "-i", args.input], capture_output=True, text=True, errors="replace").stderr
    src_w = src_h = 0
    for line in probe.splitlines():
        if "Video:" in line:
            m = _DIM_RE.search(line)
            if m:
                src_w, src_h = int(m.group(1)), int(m.group(2))
            break
    out_w, out_h = src_w * args.scale, src_h * args.scale
    default_out = f"{stem}_{out_w}x{out_h}_{args.model.split('-')[-1]}{ext}"
    out = args.out or default_out

    tmp = tempfile.mkdtemp(prefix="realesrgan_")
    tmp_frames = os.path.join(tmp, "in")
    out_frames = os.path.join(tmp, "out")
    os.makedirs(tmp_frames); os.makedirs(out_frames)
    try:
        # 1. 抽帧 (qscale 1 无损; -vsync 0 保留全部帧)
        print(f"[1/3] 抽帧 @ {fps}fps -> {tmp_frames}")
        subprocess.run([FFMPEG, "-y", "-i", args.input, "-qscale:v", "1", "-qmin", "1",
                        "-qmax", "1", "-vsync", "0",
                        os.path.join(tmp_frames, "f%08d.jpg")], check=True, capture_output=True)
        # 2. 超分
        print(f"[2/3] Real-ESRGAN {args.model} -s {args.scale}")
        subprocess.run([REALESRGAN, "-i", tmp_frames, "-o", out_frames,
                        "-n", args.model, "-s", str(args.scale), "-f", "jpg"],
                       check=True, cwd=os.path.dirname(REALESRGAN))
        # 3. 重编码 (按源帧率)
        print(f"[3/3] 重编码 {out_w}x{out_h} crf{args.crf}")
        subprocess.run([FFMPEG, "-y", "-framerate", str(fps),
                        "-i", os.path.join(out_frames, "f%08d.jpg"),
                        "-c:v", "libx264", "-crf", str(args.crf),
                        "-pix_fmt", "yuv420p", "-r", str(fps), out],
                       check=True, capture_output=True)
        print(f"[done] -> {os.path.basename(out)}  ({out_w}x{out_h}, {fps}fps)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
