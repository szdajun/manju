"""逐镜 TTS 台词 → output/ep02_voice/voice_shotNN.mp3 + 时长核对。

ep02 双女对话(小雅+婷婷)+旁白。edge-tts 三个女声分角:
  小雅(主角,自信骄傲) = XiaoyiNeural(活泼); 婷婷(同事,好奇八卦) = XiaohanNeural(温柔); 旁白 = XiaomoNeural(冷静)。
每镜台词 vs 该镜 _best.mp4 时长, 超(段长-0.5s)标 [OVER](台词溢出该镜窗口)。

用法: python scripts/voice_ep02.py [--shots 3]       # pilot
      python scripts/voice_ep02.py                    # 全 10 镜
切 minimax: 设环境 MANJU_TTS=minimax(voice 名要换 minimax ID, 见 lib/tts.py)
"""
import argparse
import os
import subprocess
import sys

MANJU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MANJU)

# 强制 MiniMax(edge_tts 未装且中国不稳; minimax 是 .env.example 既定配音路径)。
# 必须在 import lib.tts 前设, 因 PROVIDER 在 import 时算定(_load_envfile 用 setdefault 不覆盖)。
os.environ.setdefault("MANJU_TTS", "minimax")

from lib.tts import synthesize, PROVIDER  # noqa: E402
from lib.ep02_shots import LINES  # noqa: E402

I2V = "../manju/output/ep02_i2v"
OUT = "../manju/output/ep02_voice"
FFPROBE = "~/ffmpeg/ffprobe.exe"
MARGIN = 0.5

# MiniMax speech-02-hd 女声分角(ep02 双女对话, 不能用 ep01 男声 male-qn-qingse)。
XIAOYA = "female-shaonv"     # 小雅 少女活泼(主角, 自信骄傲)
TINGTING = "female-tianmei"  # 婷婷 甜美好奇(同事, 八卦) — MiniMax id 无 -1 后缀
NARRATOR = "female-chengshu"   # 旁白 成熟稳重
SPEAKER = {1: NARRATOR, 2: TINGTING, 3: XIAOYA, 4: TINGTING, 5: XIAOYA,
           6: TINGTING, 7: XIAOYA, 8: NARRATOR, 9: TINGTING, 10: XIAOYA}


def dur(path):
    try:
        o = subprocess.check_output(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path]).decode().strip()
        return float(o)
    except Exception:
        return 0.0


def spk_name(voice):
    return "旁白" if voice == NARRATOR else ("小雅" if voice == XIAOYA else "婷婷")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--speed", type=float, default=1.35,
                    help="MiniMax speed(v2 默认 0.9 偏慢; 1.35 适配 length=65 的 4.06s seg)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    shots = [int(x) for x in args.shots.split(",") if x.strip()]
    print(f"[provider] {PROVIDER}  speed={args.speed}\n")
    print(f"{'shot':>4} {'spk':>4} {'voice_s':>8} {'seg_s':>7} {'flag':>6}  台词")
    print("-" * 72)
    results = []
    for s in shots:
        line = LINES[s]
        voice = SPEAKER[s]
        out = os.path.join(OUT, f"voice_shot{s:02d}.mp3")
        seg = dur(os.path.join(I2V, f"manju_ep02_shot{s:02d}_best.mp4"))
        try:
            synthesize(line, out, voice=voice, speed=args.speed)
        except Exception as e:
            print(f"{s:>4} {spk_name(voice):>4}  FAIL: {e}")
            results.append((s, "FAIL"))
            continue
        vd = dur(out)
        flag = "OK" if vd <= seg - MARGIN else ("FIT" if vd <= seg else "OVER")
        print(f"{s:>4} {spk_name(voice):>4} {vd:>8.2f} {seg:>7.2f} {flag:>6}  {line}")
        results.append((s, flag))
    overs = [r for r in results if r[1] == "OVER"]
    fails = [r for r in results if r[1] == "FAIL"]
    print("-" * 72)
    print(f"[done] {len(results)-len(fails)}/{len(results)} 镜 -> {OUT}/  (OVER={len(overs)} FAIL={len(fails)})")


if __name__ == "__main__":
    main()
