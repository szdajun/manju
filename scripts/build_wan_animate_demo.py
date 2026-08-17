# -*- coding: utf-8 -*-
"""Wan-Animate 动作克隆 demo 一键后处理链 (raw 生成 → 最终交付).

把 gen_wan_animate.py 的原始输出 (开头特写 + 参考照渗出影子 + 768 偏软) 经三步
变成最终交付片:

  raw 768 生成
    → trim_startup      裁开头 N 帧 (f0-f1 满屏特写 + 满参考照渗出影子; 见 memory
                         wan-animate-startup-scale-trim, 默认 N=9 治渗出)
    → upscale_realesrgn Real-ESRGAN animevideov3 -s2 (身/背景 2× 锐; 视频时序一致)
    → face_protect       贴回原始好脸 (Real-ESRGAN 把小脸超成僵尸, 见 memory
                         realesrgan-upscale-pipeline)
    → 最终交付 client_a_{2W}x{2H}_matchsrc_t{N}_faceprot.mp4

各步幂等 (输出已存在则跳过), 命令即文档。validated 配置钉死在默认值里 (2026-08-11
client_a demo 实案), 别随手改:
  --frames 9   满 h_frac≈0.96 四视图参考照渗出影子到 f8, 裁到 f9 缩略图才无影
               (半身参考照渗出轻, 传 --frames 2)
  --scale 2    animevideov3 (真人脸自然 + 时序一致); -s 2 → 1536×2304
  face_protect 贴回原始脸 (脸是生成端解决的 58px, 后处理只动身/背景, 见
               memory wan-animate-768-fullbody-face-fix)

跑 (face_protect 要 insightface → 用 fitness venv):
  /f/wkspace/fitness-video-pipeline/.venv/Scripts/python.exe \\
      scripts/build_wan_animate_demo.py \\
      --input output/wan_animate/wan_client_a_768x1152_matchsrc_00001_.mp4 \\
      --coach client_a
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FFPROBE = r"~/ffmpeg/ffprobe.exe"
PYFIT = r"../fitness-video-pipeline/.venv/Scripts/python.exe"


def probe(path):
    """返回 (W, H, fps)。"""
    import json
    out = subprocess.check_output(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate", "-of", "json", path]
    ).decode(errors="replace")
    s = json.loads(out)["streams"][0]
    fr = s.get("r_frame_rate", "16/1")
    try:
        num, den = fr.split("/")
        fps = float(num) / float(den) if float(den) else 16.0
    except Exception:
        fps = 16.0
    return int(s["width"]), int(s["height"]), fps


def run(label, cmd):
    print(f"\n[{label}] {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description="Wan-Animate demo 一键后处理链")
    ap.add_argument("--input", required=True, help="gen_wan_animate.py 的原始输出 mp4")
    ap.add_argument("--coach", default="client_a", help="成品文件名前缀 (ASCII)")
    ap.add_argument("--frames", type=int, default=9,
                    help="裁开头帧数 (满 h_frac≈0.96 参考照渗出→9; 半身参考照→2)")
    ap.add_argument("--scale", type=int, default=2, choices=(2, 3, 4), help="超分倍数 (默认 2)")
    ap.add_argument("--model", default="realesr-animevideov3",
                    help="Real-ESRGAN 模型 (默认 animevideov3, 真人脸自然+时序一致)")
    ap.add_argument("--crf", type=int, default=16, help="输出质量 (默认 16)")
    ap.add_argument("--python", default=PYFIT, help="跑子脚本的 python (face_protect 要 insightface)")
    ap.add_argument("--force", action="store_true", help="忽略已有产物, 全链重跑")
    args = ap.parse_args()

    inp = os.path.abspath(args.input)
    if not os.path.exists(inp):
        print(f"[ERR] 输入不存在: {inp}", file=sys.stderr); sys.exit(2)
    indir = os.path.dirname(inp)
    stem, ext = os.path.splitext(os.path.basename(inp))
    py = args.python
    suffix = args.model.split("-")[-1]

    W, H, fps = probe(inp)
    oW, oH = W * args.scale, H * args.scale

    # 1. trim (默认输出名 = {stem}_t{N}.mp4, 与手跑一致)
    trimmed = os.path.join(indir, f"{stem}_t{args.frames}{ext}")
    if not args.force and os.path.exists(trimmed):
        print(f"[1/3 trim] 已存在, 跳过 -> {os.path.basename(trimmed)}")
    else:
        run("1/3 trim", [py, os.path.join(HERE, "trim_startup.py"),
                         "--input", inp, "--frames", str(args.frames)])

    # 2. upscale (默认输出名 = {stem}_t{N}_{oW}x{oH}_{suffix}.mp4)
    upscaled = os.path.join(indir, f"{stem}_t{args.frames}_{oW}x{oH}_{suffix}{ext}")
    if not args.force and os.path.exists(upscaled):
        print(f"[2/3 upscale] 已存在, 跳过 -> {os.path.basename(upscaled)}")
    else:
        run("2/3 upscale", [py, os.path.join(HERE, "upscale_realesrgan.py"),
                            "--input", trimmed, "--scale", str(args.scale),
                            "--model", args.model, "--crf", str(args.crf)])

    # 3. face_protect (显式 --out = 干净成品名)
    final = os.path.join(indir, f"{args.coach}_{oW}x{oH}_matchsrc_t{args.frames}_faceprot{ext}")
    if not args.force and os.path.exists(final):
        print(f"[3/3 face_protect] 已存在, 跳过 -> {os.path.basename(final)}")
    else:
        run("3/3 face_protect", [py, os.path.join(HERE, "face_protect_upscale.py"),
                                  "--orig", trimmed, "--upscaled", upscaled,
                                  "--out", final, "--crf", str(args.crf)])

    print(f"\n[done] 最终交付 -> {final}")
    print(f"       ({oW}x{oH}, {fps:.0f}fps, 链: trim{args.frames} → {suffix} -s{args.scale} → face_protect)")


if __name__ == "__main__":
    main()
