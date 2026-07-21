"""ep02 i2v v3 重跑: length=65(≈4s/镜) 为对白配音留时长 + 锁 v2 验过的 seed 保起势。

v2 镜长 41(2.56s) 装不下 37s 对白(shot2/6/8 哪怕 MiniMax 满速 2.0× 仍溢出 seg)。
对白剧用对白节奏: length=65(4.06s/镜) + 配音 MiniMax speed=1.35 → 每镜台词 ≤ seg。
锁 v2 _best 的 seed(size-match 出, 见 candidates/) → 同 keyframe+seed, 起势/风格延续 v2。
注意: LTX 同 seed 不同 length ≠ 原片段延长(采样轨迹变), 仅起势相近。

无 face 门(漫画 face_sim 无效, 已由用户肉眼验过 v2 seed)。
产物直接覆盖 ep02_i2v/manju_ep02_shot{NN}_best.mp4(v2 已备份 _v2_backup/)。

用法: .venv/Scripts/python.exe scripts/gen_ep02_i2v_v3.py            # 全 10 镜
      .venv/Scripts/python.exe scripts/gen_ep02_i2v_v3.py --shots 8  # 单镜
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

MANJU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MANJU)
from lib.comfyui_client import ComfyUIClient, build_ltx_i2v_workflow  # noqa: E402
from lib.ep02_shots import I2V_MOTION, NEG  # noqa: E402

KF = "../manju/output/ep02"
COMFY_OUT = "../ComfyUI/output"
OUT = "../manju/output/ep02_i2v"

LENGTH = 65          # 4.06s/镜, 为对白留时长
STEPS = 40
CFG = 2.5

# v2 _best 的 seed(size-match candidates/, 保起势)。shot6=手部特写无主角脸。
PINNED = {
    1: 4032586978, 2: 2089961790, 3: 1080337250, 4: 1177471316, 5: 4019952382,
    6: 2433695285, 7: 1784910358, 8: 2283813465, 9: 3155011197, 10: 1719447198,
}
# shot2 v2 是 576×896 hi-res(治糊), 保留; 其余 512×768。
RESHI = {2: (576, 896)}

FACE_STABLE = (", face stable, identical face throughout, consistent facial features, symmetrical face, "
               "face centered, minimal head movement, face clearly visible, no face morphing, no face distortion")
FACE_NEG = ", face morphing, face distortion, warped face, melting face, asymmetrical eyes, deformed face"


def latest(pat):
    fs = glob.glob(pat)
    return max(fs, key=os.path.getmtime) if fs else None


def run_shot(c, shot):
    src = latest(os.path.join(KF, f"ep02_shot{shot:02d}_*.png"))
    if not src:
        print(f"[skip] 关键帧缺: ep02_shot{shot:02d}_*.png", flush=True)
        return None
    w, h = RESHI.get(shot, (512, 768))
    seed = PINNED[shot]
    positive = I2V_MOTION[shot] + FACE_STABLE
    negative = NEG + FACE_NEG
    prefix = f"manju_ep02_shot{shot:02d}_v3"
    print(f"[i2v shot {shot:2d}] len={LENGTH}@16≈{LENGTH/16:.1f}s {w}x{h} seed={seed}", flush=True)
    img = c.upload_image(src)
    wf = build_ltx_i2v_workflow(
        img, positive, negative, width=w, height=h,
        length=LENGTH, steps=STEPS, cfg=CFG, seed=seed,
        frame_rate=16, filename_prefix=prefix)
    pid = c.queue_prompt(wf)
    c.wait_for_completion(pid, timeout=900)
    time.sleep(1)
    mp4 = latest(os.path.join(COMFY_OUT, f"{prefix}_*.mp4"))
    if not mp4:
        print(f"  [ERR] shot{shot} 无产出", flush=True)
        return None
    dst = os.path.join(OUT, f"manju_ep02_shot{shot:02d}_best.mp4")
    shutil.copy(mp4, dst)
    os.utime(dst, None)
    print(f"  [OK] -> {os.path.basename(dst)}", flush=True)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    args = ap.parse_args()
    shots = [int(x) for x in args.shots.split(",") if x.strip()]
    c = ComfyUIClient(args.server)
    ok = []
    for s in shots:
        try:
            if run_shot(c, s):
                ok.append(s)
        except Exception as e:
            print(f"[shot {s}] FAIL: {e}", flush=True)
    print(f"\n[done] {len(ok)}/{len(shots)} 镜: {ok}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
