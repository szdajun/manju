"""manju 图生视频入口(LTX-Video i2v, 本地 ComfyUI)。

拿 ep01 静帧 → LTX 出 ~2.5s 动画 → 存 mp4。漫剧每镜短动作够用。
静帧已 IPAdapter 锁脸, i2v 继承角色一致性。

用法:
    python scripts/gen_i2v.py --shot 4
    python scripts/gen_i2v.py --shot 7 --length 57 --seed 1
"""
import argparse
import glob
import os
import shutil
import sys
import time

MANJU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MANJU)
from lib.comfyui_client import ComfyUIClient, build_ltx_i2v_workflow  # noqa: E402

EP01_DIR = "../manju/output/ep01"
COMFY_OUT = "../ComfyUI/output"

# i2v 运动意图 prompt(每镜: 要发生的运动, 不是静态描述)
I2V_SHOTS = {
    4: ("the man holds and strains to lift a clearly visible small black dumbbell with both hands, "
        "gripping the dumbbell tightly, arms trembling slightly with effort, face turning red, "
        "mostly subtle small movement, the dumbbell stays sharp and clearly in focus, gym background static"),
    5: ("the man on a treadmill slowly loses balance and tilts backwards, "
        "his face and head remain stable and clearly visible with a clear sharp mouth, "
        "the stumbling and falling motion is mostly in his legs and lower body, "
        "comedic slow-motion fall, face stays sharp and in focus"),
    7: ("the man doing a plank, his whole body shaking and trembling violently, a fly buzzing "
        "and landing on his nose, small frantic trembling"),
    8: ("the man lying flat on the gym floor weakly raises a thumbs up, slight movement, "
        "the trainer stands still looking down at him"),
}

NEG = ("text, watermark, logo, distorted face, deformed hands, morphing identity, "
       "extra fingers, bad anatomy, blurry, low quality, violent scene change, "
       "mouth disappearing, blurry mouth, unstable mouth, melting face, face morphing")


def latest(pat):
    files = glob.glob(pat)
    return max(files, key=os.path.getmtime) if files else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", type=int, default=4, choices=list(I2V_SHOTS.keys()))
    ap.add_argument("--src", default=None, help="覆盖源静帧(默认 ep01/ep01_shotNN_00001_.png)")
    ap.add_argument("--out", default="../manju/output/ep01_i2v")
    ap.add_argument("--length", type=int, default=41, help="帧数(step8; 41@16fps≈2.5s)")
    ap.add_argument("--fps", type=float, default=16)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--width", type=int, default=512,
                    help="输出宽(默认 512 安全档; 768 仅小道具清晰度需要时显式传, 见 RAM 段错误警告)")
    ap.add_argument("--height", type=int, default=768, help="输出高(默认 768 竖屏)")
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    args = ap.parse_args()

    src = args.src or os.path.join(EP01_DIR, f"ep01_shot{args.shot:02d}_00001_.png")
    if not os.path.exists(src):
        print(f"[ERR] 源静帧不存在: {src}")
        return 1

    prefix = f"manju_i2v_shot{args.shot:02d}"
    positive = I2V_SHOTS[args.shot]
    print(f"[i2v shot {args.shot}] src={os.path.basename(src)}")
    print(f"  positive={positive}")
    print(f"  length={args.length} @ {args.fps}fps ~= {args.length/args.fps:.1f}s, "
          f"steps={args.steps} cfg={args.cfg}")

    c = ComfyUIClient(args.server)
    img_name = c.upload_image(src)
    print(f"[upload] src -> {img_name}")
    wf = build_ltx_i2v_workflow(
        img_name, positive, NEG,
        width=args.width, height=args.height,
        length=args.length, steps=args.steps, cfg=args.cfg,
        seed=args.seed, strength=args.strength,
        frame_rate=args.fps, filename_prefix=prefix)
    print("[queue] LTX i2v 生成中(首次加载 ltx ckpt 较慢)...")
    pid = c.queue_prompt(wf)
    entry = c.wait_for_completion(pid, timeout=900)
    print("[done] ComfyUI 生成完, 找输出 mp4...")

    time.sleep(1)
    mp4 = latest(os.path.join(COMFY_OUT, f"{prefix}_*.mp4"))
    if not mp4:
        anyf = latest(os.path.join(COMFY_OUT, f"{prefix}_*"))
        print(f"[WARN] ComfyUI/output 没找到 mp4; prefix 下最新文件: {anyf}")
        return 2
    os.makedirs(args.out, exist_ok=True)
    dst = os.path.join(args.out, os.path.basename(mp4))
    shutil.copy(mp4, dst)
    print(f"[saved] -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
