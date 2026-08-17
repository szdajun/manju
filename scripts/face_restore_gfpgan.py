# -*- coding: utf-8 -*-
"""GFPGAN 脸修复 — 全身构图小脸补真实细节, 治僵尸脸 (用户拍板"保持全身 + 脸修复").

根因 (实测钉死): 全身构图脸仅 34px (生成448) → 超分/animevideov3 猜成僵尸,
贴回原始脸也才 34px (没细节) → 都救不了. GFPGAN 是**脸修复**模型 (非换脸,
保持身份), 专为低质/小脸补真实人脸细节. 用户原话"保持全身+CodeFormer修脸";
GFPGANv1.4 是直接可下的等价脸修复器 (CodeFormer 模型在 gdrive 不便).

fitness venv 有 gfpgan, 但 torchvision>=0.17 删了 functional_tensor → 需 shim.
逐帧 GFPGANer.enhance(paste_back=True): 检测脸→对齐→修复成清晰脸→羽化贴回原位
(身/背景纹丝不动, 只换脸). GPU 快, CPU ~7min/35帧.

跑 (fitness venv):
  /f/wkspace/fitness-video-pipeline/.venv/Scripts/python.exe scripts/face_restore_gfpgan.py \
      --input output/wan_animate/client_a_896x1344_matchsrc_t2_2x.mp4 \
      --out   output/wan_animate/client_a_896x1344_matchsrc_t2_2x_gfpgan.mp4
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import types

# --- shim: torchvision>=0.17 删了 functional_tensor, basicsr/gfpgan 仍 import 它 ---
import torchvision.transforms.functional as _F

_mod = types.ModuleType("torchvision.transforms.functional_tensor")
for _a in dir(_F):
    try:
        setattr(_mod, _a, getattr(_F, _a))
    except Exception:
        pass
sys.modules["torchvision.transforms.functional_tensor"] = _mod

import cv2  # noqa: E402
import torch  # noqa: E402

FFMPEG = r"~/ffmpeg/ffmpeg.exe"
MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tools", "gfpgan", "GFPGANv1.4.pth",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="超分全身 mp4 (脸区待修)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--upscale", type=int, default=2, help="脸修复中间分辨率倍数 (默认2)")
    args = ap.parse_args()

    if not os.path.exists(MODEL):
        print(f"[ERR] 模型不在: {MODEL}", file=sys.stderr); sys.exit(2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[init] device={device}  model=GFPGANv1.4")
    from gfpgan import GFPGANer
    restorer = GFPGANer(
        model_path=MODEL, upscale=args.upscale,
        arch="clean", channel_multiplier=2, device=device,
    )

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print("[ERR] 打不开输入", file=sys.stderr); sys.exit(2)
    fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    tmp = tempfile.mkdtemp(prefix="gfpgan_")
    outdir = os.path.join(tmp, "out"); os.makedirs(outdir)
    n = restored = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            try:
                _cf, _rf, restored_img = restorer.enhance(frame, paste_back=True)
                if restored_img is not None:
                    out = restored_img
                    restored += 1
                else:
                    out = frame
            except Exception as e:
                print(f"  f{n} 跳过 (无脸/err): {type(e).__name__}: {e}")
                out = frame
            cv2.imwrite(os.path.join(outdir, f"f{n:08d}.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 96])
            n += 1
            if n % 5 == 0:
                print(f"  ...{n} 帧 ({restored} 修脸)")
        cap.release()
        subprocess.run([FFMPEG, "-y", "-framerate", str(fps),
                        "-i", os.path.join(outdir, "f%08d.jpg"),
                        "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
                        "-r", str(fps), args.out], check=True, capture_output=True)
        print(f"[done] {n}帧, {restored}帧修脸 ({restored / max(n, 1) * 100:.0f}%) -> {os.path.basename(args.out)}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
