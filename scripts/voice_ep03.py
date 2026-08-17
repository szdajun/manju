"""ep03 单人 TTS — 林夏一人 5 镜。

沿用 ep02 模板 (voice_ep02.py), 单角色:
  - LINES 来自 lib/ep03_shots
  - SPEAKER 5 镜全 LINXIA (单一女声)
  - I2V 路径用 ep03 seedance (output/ep03_i2v_real/_seedance)
  - 台词短 (12-23 字), 默认 speed=1.0 (MiniMax 默认偏快, 不用 ep02 1.35)

用法: .venv/Scripts/python.exe scripts/voice_ep03.py
"""
import argparse
import os
import subprocess
import sys

MANJU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MANJU)

# 强制 MiniMax (沿用 ep02 拍板)
os.environ.setdefault("MANJU_TTS", "minimax")

from lib.tts import synthesize, PROVIDER  # noqa: E402  # TTS provider (MANJU_TTS)
from lib.video_gen import PROVIDER as VIDEO_PROVIDER  # noqa: E402  # i2v provider (MANJU_VIDEOGEN)
from lib.storyboard import (  # noqa: E402
    LINE_TYPE_DIALOG, character, clean_line, line_type, load_script, shot)

FFPROBE = "~/ffmpeg/ffprobe.exe"
MARGIN = 0.5


def dur(path):
    try:
        o = subprocess.check_output(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path]).decode().strip()
        return float(o)
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", default="03", help="集数 (读剧本数据, 默认 03)")
    ap.add_argument("--shots", default=None, help="指定镜 (如 1,2,3; 默认剧本全部)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="MiniMax speed (ep03 台词短, 1.0 自然; ep02 1.35 是给长台词)")
    # orchestrator 接入: 自定义输出/i2v 目录 + 增量跳过
    ap.add_argument("--out", default=None, help="配音输出目录 (默认 output/ep{ep}_voice)")
    ap.add_argument("--i2v-dir", default=None, help="i2v 片段目录 (取镜时长, 默认 ep{ep}_i2v_real/_seedance)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="已存在 voice_shotNN.mp3 的镜跳过 (增量)")
    args = ap.parse_args()

    script = load_script(args.ep)
    ep = script["ep"]
    out_dir = args.out or os.path.join(MANJU, "output", f"ep{ep}_voice")
    i2v_dir = args.i2v_dir or os.path.join(MANJU, "output", f"ep{ep}_i2v_real", "_seedance")
    os.makedirs(out_dir, exist_ok=True)
    shots = [int(x) for x in (args.shots or ",".join(script["shots"].keys())).split(",")
             if x.strip()]
    print(f"[provider] {PROVIDER}  speed={args.speed}\n")
    print(f"{'shot':>4} {'voice_s':>8} {'seg_s':>7} {'flag':>6}  台词")
    print("-" * 60)
    results = []
    for s in shots:
        line = clean_line(shot(script, s)["line"])  # 剥"（内心：）"壳, 防 TTS 念出
        voice_id = character(script, s)["voice_id"]
        out = os.path.join(out_dir, f"voice_shot{s:02d}.mp3")
        # 端到端整合 (2026-08-06): 只生成组装阶段需要的 TTS.
        #   - 无台词 (ambient) → 不需要 TTS
        #   - H3 对话镜 → 语音已内嵌 H3 视频 (i2v 阶段喂 audio 驱动口型), 不重复生成
        if not (line or "").strip():
            print(f"{s:>4}  SKIP (无台词 ambient)")
            results.append((s, "SKIP"))
            continue
        if VIDEO_PROVIDER == "h3_local" and line_type(script, s) == LINE_TYPE_DIALOG:
            # H3 对话镜: 语音已内嵌 H3. 清掉陈旧 voice 文件 (可能是之前 seedance
            # 跑法留下的) — 否则组装 has_voice=T × has_orig=T → 误走 mix 双叠音.
            if os.path.exists(out):
                os.remove(out)
                print(f"{s:>4}  DEL (H3 对话镜: 删陈旧 {os.path.basename(out)}, "
                      f"防组装双叠音)")
            else:
                print(f"{s:>4}  SKIP (H3 对话镜: 语音已内嵌 H3)")
            results.append((s, "SKIP"))
            continue
        # 增量跳过: 已存在产物跳过 (默认重建, --skip-existing 才跳)
        if args.skip_existing and os.path.exists(out):
            print(f"{s:>4}  SKIP (已存在 {out})")
            results.append((s, "SKIP"))
            continue
        # seg 时长从 i2v mp4 取 (5s 镜)
        seg = dur(os.path.join(i2v_dir, f"ep{ep}_shot{s:02d}.mp4"))
        try:
            synthesize(line, out, voice=voice_id, speed=args.speed)
        except Exception as e:
            print(f"{s:>4}  FAIL: {e}")
            results.append((s, "FAIL"))
            continue
        vd = dur(out)
        flag = "OK" if vd <= seg - MARGIN else ("FIT" if vd <= seg else "OVER")
        print(f"{s:>4} {vd:>8.2f} {seg:>7.2f} {flag:>6}  {line}")
        results.append((s, flag))
    overs = [r for r in results if r[1] == "OVER"]
    fails = [r for r in results if r[1] == "FAIL"]
    print("-" * 60)
    print(f"[done] {len(results)-len(fails)}/{len(results)} 镜 -> {out_dir}/  "
          f"(OVER={len(overs)} FAIL={len(fails)})")


if __name__ == "__main__":
    main()