#!/usr/bin/env python3
"""facelock_h3_ep.py — H3 i2v 输出逐镜换脸 (2026-08-15, P0 完善).

H3 每镜独立重渲脸 → 跨镜身份不一致 (verify 标漂移). 用 inswapper 把每镜输出脸
换成该镜说话人的角色 ref 脸 (face_ref), 锁死身份.

复用 facelock_wan_animate.py 的 LOCKED recipe (inswapper + color_match + skin_and_lip_fix).
跑 fitness venv (insightface + inswapper_128). 不需要 ComfyUI.

用法:
  ../fitness-video-pipeline/.venv/Scripts/python.exe scripts/facelock_h3_ep.py --ep 20
  ../fitness-video-pipeline/.venv/Scripts/python.exe scripts/facelock_h3_ep.py --ep 20 --i2v-dir ... --out-dir ...
产物: {i2v_dir}/facelock/{ep}_shot{s:02d}_facelock.mp4
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from facelock_wan_animate import facelock, get_app, get_swapper, imread_unicode, primary_face  # noqa: E402

MANJU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description="H3 i2v 输出逐镜 inswapper 换脸 (锁角色身份)")
    ap.add_argument("--ep", required=True, help="集数, 读 output/ep{ep}/script.json")
    ap.add_argument("--i2v-dir", default=None, help="i2v 输出目录 (默认 output/ep{ep}_i2v_real/_seedance)")
    ap.add_argument("--out-dir", default=None,
                    help="换脸输出目录 (默认 {i2v_dir}/facelock; 文件名=标准 ep{ep}_shot{s:02d}.mp4, "
                         "组装可直接 --i2v-dir 指向它)")
    ap.add_argument("--shots", default=None, help="指定镜 (逗号分隔; 默认剧本全部)")
    ap.add_argument("--color-match", type=float, default=0.8)
    args = ap.parse_args()

    script_path = os.path.join(MANJU, "output", f"ep{args.ep}", "script.json")
    script = json.load(open(script_path, encoding="utf-8"))
    i2v_dir = args.i2v_dir or os.path.join(MANJU, "output", f"ep{args.ep}_i2v_real", "_seedance")
    out_dir = args.out_dir or os.path.join(i2v_dir, "facelock")
    os.makedirs(out_dir, exist_ok=True)
    shots = [int(x) for x in (args.shots or ",".join(script["shots"].keys())).split(",") if x.strip()]

    app = get_app()
    swapper = get_swapper()
    prefix = f"ep{args.ep}_shot"
    ok, fail = [], []
    for s in shots:
        spk = script["shots"][str(s)].get("speaker")
        ch = script["characters"].get(spk, {})
        face_ref = ch.get("face_ref")
        if not face_ref or not os.path.exists(face_ref):
            print(f"[skip] shot{s:02d} 说话人 {spk} 缺 face_ref: {face_ref!r}")
            fail.append(s)
            continue
        src_img = imread_unicode(face_ref)
        src_face = primary_face(app, src_img)
        if src_face is None:
            print(f"[skip] shot{s:02d} 源照无脸: {face_ref}")
            fail.append(s)
            continue
        mp4s = sorted(glob.glob(os.path.join(i2v_dir, f"{prefix}{s:02d}.mp4")))
        if not mp4s:
            print(f"[skip] shot{s:02d} 缺 i2v 输出")
            fail.append(s)
            continue
        # 标准命名 (与 i2v 输出同名) → 组装 --i2v-dir 可直接指向 facelock 目录
        out = os.path.join(out_dir, f"{prefix}{s:02d}.mp4")
        print(f"[shot{s:02d}] {spk} 换脸 {os.path.basename(face_ref)} ...", flush=True)
        facelock(app, swapper, src_face, mp4s[-1], out, args.color_match)
        ok.append(s)

    print(f"\n[done] ok={ok} fail={fail} -> {out_dir}  (组装: --i2v-dir {out_dir})")


if __name__ == "__main__":
    sys.exit(main() or 0)
