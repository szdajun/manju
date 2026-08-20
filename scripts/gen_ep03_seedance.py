"""ep03 写实版 i2v — Seedance 1.0 pro (云端 doubao-seedance-1-0-pro-250528)。

沿用 ep02 写实路径 (gen_ep02_seedance.py), 适配:
  - KF_DIR: output/ep03_real (ep03 关键帧)
  - OUT: output/ep03_i2v_real/_seedance
  - 5 镜 (vs ep02 10 镜), 文件名 ep03_shotNN.mp4
  - I2V_MOTION 来自 lib/ep03_shots, 末尾必带"the misty-blue top stays misty-blue throughout"
    等 3 句不变式 → Seedance i2v 时也持续锁衣服
  - FACE_POS 沿用 ep02 (降头动脸漂 + 薄眼帘/素颜正向)

跑法:
  .venv/Scripts/python.exe scripts/gen_ep03_seedance.py                  # 全 5 镜
  .venv/Scripts/python.exe scripts/gen_ep03_seedance.py --shots 1,3,5    # 指定
"""
import argparse
import glob
import os
import sys

MANJU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MANJU)
# 强制 MiniMax TTS (对齐 voice_ep03.py 拍板) — ⚠ 必须在 import lib.tts 之前设,
# 否则 lib.tts 在 import 时读 PROVIDER 拿到默认 edge (不支持 speed kwarg)
os.environ.setdefault("MANJU_TTS", "minimax")
from lib.video_gen import generate, _extract_last_frame  # noqa: E402  # provider 由 MANJU_VIDEOGEN 切换 (seedance/h3_local/ltx_local)
from lib.storyboard import (  # noqa: E402
    LINE_TYPE_AMBIENT, LINE_TYPE_DIALOG, LINE_TYPE_VOICEOVER,
    character, clean_line, frame_chars, line_type, load_script, shot,
    speaker)
from lib import h3prompt  # noqa: E402  # 官方六段式 prompt builder (2026-08-15)
from lib.tts import synthesize  # noqa: E402  # 对话镜: 台词 TTS → 喂 H3 驱动口型

# FACE_POS 沿用 ep02 (降 Seedance 头动脸漂 + 薄眼帘/素颜正向, 因正向 > 负向)
FACE_POS = (
    ", face stable and centered, minimal head movement, face clearly visible throughout, "
    "bare fresh natural face, no makeup, nude natural lips, no eyeliner, no eyeshadow, "
    "thin smooth natural eyelids, wide open bright alert eyes, clean smooth under-eye area"
)
# 镜头台词类型后缀 (2026-08-06 端到端整合, H3 三音频模式)
# 对话镜: 角色开口, 唇型由音频驱动 (喂 audio 后 H3 自动对齐)
# ⚠ 2026-08-07 修复: H3 语音内容由 prompt 决定, 英文 prompt → 生成非中文.
#   改中文说话指令 + 内嵌台词文本 (锁中文 + 锁内容) + 前置服装描述 (加强跨镜一致).
def _dialog_speech_cn(line: str, speaker_name: str = "角色") -> str:
    """对话镜中文说话指令: 中文台词文本 + 中文同步约束 (H3 生成中文语音)."""
    line = (line or "").strip().replace('"', "'").replace("\n", " ")
    return (f"，{speaker_name}用中文清晰自然地念出台词：\"{line}\"，"
            "声音和嘴型与配音音频完全同步，嘴唇自然张合，语音清晰，"
            "身体保持端正姿态，头部面向镜头，表情自然")


# 旁白镜: H3 prompt 强制闭嘴 + off-screen voiceover → H3 只出环境音 (无台词)
VOICEOVER_SILENT = (
    ", the character stays completely silent with mouth closed and does not speak, "
    "off-screen voiceover narration only, natural subtle breathing and idle motion, "
    "lips remain closed"
)


def _outfit_prefix(script: dict, sid: int | str) -> str:
    """对话镜服装前置: 从角色指纹取服装段 (加强跨镜服装一致, 治 H3 每镜重渲染漂移)."""
    fp = (character(script, sid) or {}).get("fingerprint") or ""
    # 截取指纹里服装相关段: 从 "wearing" 或 上衣描述 起 (指纹规范含服装)
    outfit = fp
    for marker in ("wearing", "dressed in", "in a"):
        if marker in fp:
            outfit = fp[fp.index(marker):]
            break
    return outfit.strip().rstrip(".,")


_FFMPEG = "~/ffmpeg/ffmpeg.exe"


def _make_silent_audio(out_path: str, dur_s: float):
    """生成静音音频 (双人无对话镜喂静音, 激活 H3 双人场景; 实测无音频只渲 1 人)."""
    import subprocess
    subprocess.run(
        [_FFMPEG, "-y", "-v", "error", "-f", "lavfi",
         "-i", f"anullsrc=channel_layout=stereo:sample_rate=44100",
         "-t", f"{max(0.5, dur_s - 0.1):.2f}", "-c:a", "libmp3lame",
         "-b:a", "128k", out_path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path


def _pad_audio_to_duration(audio_path: str, target_s: float, out_path: str | None = None):
    """TTS 音频补尾静音到目标时长 (治 H3 音频<视频时长时自由发挥加杂音).

    2026-08-07 用户洞察: 台词 1.3s 但镜头 3s, 多出的 1.7s H3 没有音频参照 → 自由发挥
    (加杂音/乱动嘴). 补静音给 H3 全程音频参照. 实测 shot2 空档最小(1.04s)渲染最干净.
    """
    import subprocess
    import os
    dst = out_path or audio_path
    tmp = dst + ".pad.mp3"
    pad_s = max(0.3, target_s - 0.15)
    subprocess.run(
        [_FFMPEG, "-y", "-v", "error", "-i", audio_path,
         "-af", f"apad=pad_dur={pad_s}", "-c:a", "libmp3lame", "-b:a", "128k",
         "-t", f"{pad_s}", tmp],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.replace(tmp, dst)
    return dst


def _build_dual_scene_prompt(script: dict, sid: int | str) -> str:
    """双人同框无对话 (混合方案 2026-08-07): 两人都在画面一起做动作, 全程不说话.

    ⚠ Picture 顺序必须匹配 refs (倾听者先, 说话者后), 否则 H3 只渲 1 人.
    """
    spk = speaker(script, sid)
    fc = frame_chars(script, sid) or list(script["characters"].keys())
    other = next((c for c in fc if c != spk), spk)
    fps = [(script["characters"].get(c) or {}).get("fingerprint", c) for c in (other, spk)]
    scene = shot(script, sid).get("shot") or ""
    motion = shot(script, sid).get("motion") or ""
    return (f"<Picture 1> {fps[0]} 与 <Picture 2> {fps[1]} "
            f"双人无遮挡并列站立，两人都在画面中，{scene}，{motion}，"
            f"两人全程保持沉默不说话，嘴巴紧闭，一起做动作，表情自然")


def _build_dual_dialog_prompt(script: dict, sid: int | str) -> str:
    """双人对话 prompt (2026-08-07 新增): Ref2VA 双角色同框 + 说话人说中文台词.

    ⚠ 2026-08-07 修: 用"名字"当 <Picture N> 描述 → H3 只渲 1 人. 必须用**外观描述**
    (指纹) 当 <Picture N> + 强调"双人无遮挡并列站立" (对齐已验证的 h3_ref2va 双人格式).
    """
    spk = speaker(script, sid)
    others = [c for c in script["characters"] if c != spk]
    other = others[0] if others else spk
    line = clean_line(shot(script, sid)["line"] or "")  # 剥"（内心：）"壳, 防 H3 念出
    line = line.replace('"', "'").replace("\n", " ")
    scene = shot(script, sid)["shot"] or ""
    # ⚠ 2026-08-07: Picture 1 = 倾听者 (ref0), Picture 2 = 说话者 (ref1), 对齐 shot2 成功布局.
    return (f"<Picture 1> 对应【倾听者：{other}】，外观与 <Picture 1> 完全一致；"
            f"<Picture 2> 对应【说话人：{spk}】，外观与 <Picture 2> 完全一致。"
            f"双人无遮挡并列站立，{spk} 与 {other} 相互面对，均完整出现在画面中央，"
            f"身体保持端正自然姿态，{scene}。"
            f"【唯一说话者：{spk}】：仅 {spk} 开口说话，"
            f"嘴唇动作与输入的中文配音音频 100% 精准同步，嘴型自然张合，无延迟或错位；"
            f"清晰念出台词：「{line}」，绝对只说 1 次，无重复、无省略、无篡改；"
            f"说完台词后立即完全沉默，全程无任何额外杂音、气音或模糊发音。"
            f"【唯一倾听者：{other}】：全程保持绝对沉默，嘴巴始终紧闭，"
            f"无任何说话动作（包括微张、嘴型变化）；专注倾听 {spk}，"
            f"表情自然，适度点头回应，无多余肢体动作，身体姿态稳定。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", default="03", help="集数 (读剧本数据, 默认 03)")
    ap.add_argument("--shots", default=None, help="指定镜 (如 1,2,3; 默认剧本全部)")
    ap.add_argument("--duration", type=int, default=None,
                    help="镜时长秒 (默认读剧本 duration_s, 对齐 orchestrator)")
    ap.add_argument("--resolution", default="480p")
    ap.add_argument("--no-face-shots", default="",
                    help="不加 FACE_POS 的镜 (逗号分隔; 默认空=全加)")
    ap.add_argument("--kf-dir", default=None,
                    help="关键帧目录 (含 ep{ep}_shotNN_*.png, 默认 output/ep{ep}_real)")
    ap.add_argument("--out", default=None,
                    help="i2v 输出目录 (默认 output/ep{ep}_i2v_real/_seedance)")
    # H3 档位 (对齐 HANDOFF): 768x1344 正式 (25min/段) / 432x768 迭代 (6.3min/段)
    ap.add_argument("--h3-width", type=int, default=768,
                    help="H3 单人 i2v 宽 (默认 768, 迭代档 432)")
    ap.add_argument("--h3-height", type=int, default=1344,
                    help="H3 单人 i2v 高 (默认 1344, 迭代档 768)")
    ap.add_argument("--h3-turbo", action="store_true",
                    help="H3 Turbo LoRA (steps 20→8, 768x1344 从 25min → ~10min)")
    # 2026-08-15: 官方 h3-prompt-writing 六段式 prompt (对照实验验证四模式), legacy 保留回退
    ap.add_argument("--prompt-style", choices=["official", "legacy"], default="official",
                    help="H3 prompt 构建: official=官方六段式 (默认) / legacy=旧中文指令")
    # 2026-08-15 P0: 尾帧衔接 — 单人无对白镜用上镜尾帧当首帧, 跨镜不跳变 (LocalMiniDrama 参考)
    ap.add_argument("--tail-link", action="store_true",
                    help="尾帧衔接: 单人无对白镜 (H3 ImageToVideo) 用上一镜输出尾帧作首帧")
    # 2026-08-15: H3 音频偶发乱码/重复 (seed 相关) → 换 seed 重跑修复
    ap.add_argument("--seed", type=int, default=None, help="H3 采样 seed (默认 42, 换值避重复/乱码)")
    # 2026-08-19 R2V A/B: 单人对话镜双参考 [kf, 定妆照] (P1 构图 P2 身份)
    # ⚠ A/B 已判死 (脸漂移+幻觉道具, 见 HANDOFF 2026-08-19 晚): 默认 None 走回原单参考,
    #    想复测显式 --identity-ref <path> (未来 H3 官方 API 可再验).
    ap.add_argument("--identity-ref", default=None,
                    help="角色定妆照路径 (单人对话镜 refs=[kf,定妆照] 双参考; "
                         "R2V A/B 判死后默认关闭, 显式传路径才启用)")
    args = ap.parse_args()

    script = load_script(args.ep)
    ep = script["ep"]
    kf_dir = args.kf_dir or os.path.join(MANJU, "output", f"ep{ep}_real")
    out = args.out or os.path.join(MANJU, "output", f"ep{ep}_i2v_real", "_seedance")
    os.makedirs(out, exist_ok=True)
    no_face = {int(x) for x in (args.no_face_shots or "").split(",") if x.strip()}
    ident_ref = args.identity_ref   # 默认不启用 (判死); 显式传路径才双参考

    shots = [int(x) for x in (args.shots or ",".join(script["shots"].keys())).split(",")
             if x.strip()]
    prefix = f"ep{ep}_shot"
    ok, skip, fail = [], [], []
    for s in shots:
        out_path = os.path.join(out, f"{prefix}{s:02d}.mp4")
        if os.path.exists(out_path):
            print(f"[skip] shot{s:02d} 已存在: {os.path.basename(out_path)}", flush=True)
            skip.append(s)
            continue
        cands = sorted(glob.glob(os.path.join(kf_dir, f"{prefix}{s:02d}_*.png")))
        if not cands:
            print(f"[ERR] shot{s:02d} 关键帧缺: {prefix}{s:02d}_*.png in {kf_dir}", flush=True)
            fail.append(s)
            continue
        src = cands[-1]
        motion = shot(script, s)["motion"]
        if not motion:
            print(f"[ERR] shot{s:02d} 无 motion 意图 (剧本 shots[{s}].motion 缺)", flush=True)
            fail.append(s)
            continue
        lt = line_type(script, s)
        # prompt 按镜头类型拼: dialog 中文说话(内嵌台词) / voiceover 闭嘴 / ambient 无台词
        if lt == LINE_TYPE_DIALOG:
            line = clean_line(shot(script, s)["line"])
            spk = speaker(script, s)
            # 服装前置 + 英文动作 + 中文说话指令 (锁中文语音 + 锁台词 + 锁服装)
            prompt = (f"{_outfit_prefix(script, s)}, {motion}"
                      + _dialog_speech_cn(line, spk))
        elif lt == LINE_TYPE_VOICEOVER:
            prompt = motion + VOICEOVER_SILENT
        else:  # ambient
            prompt = motion
        if s not in no_face:
            prompt = prompt + FACE_POS
        print(f"[shot{s:02d}] KF={os.path.basename(src)} line_type={lt}", flush=True)
        try:
            # provider 由 MANJU_VIDEOGEN 切换; 参数按 provider 路由
            from lib.video_gen import PROVIDER
            dur = args.duration if args.duration is not None \
                else int(script.get("duration_s", 5))
            kw = {"duration": dur}
            if PROVIDER == "h3_local":
                # H3 单人 i2v: 首帧关键帧 + 竖屏 768x1344 (正式档) / 432x768 (迭代档)
                kw.update({"width": args.h3_width, "height": args.h3_height,
                           "mode": "single", "turbo": args.h3_turbo})
                if args.seed is not None:
                    kw["seed"] = args.seed
                fc = frame_chars(script, s)
                # 缓冲空镜 (织体第④律): frame_chars=[] → 无人纯场景, 无 ref 无 audio
                # ⚠ is_buffer 哨兵: buffer prompt 设完后必须跳过下游 elif 链
                #   (build_ambient/voiceover 会查 character() → None 崩, 2026-08-20 ep28 镜3)
                is_buffer = fc is not None and len(fc) == 0
                if is_buffer:
                    prompt = h3prompt.build_buffer_prompt(script, s)
                    kw["mode"] = "single"
                fc = [c for c in (fc or []) if c in script["characters"]
                      and (script["characters"][c].get("face_ref")
                           or script["characters"][c].get("body_ref"))]
                if not is_buffer and len(fc) >= 2:
                    # ⭐ 双人同框 (2026-08-07 混合方案): Ref2VA 双角色同框.
                    # 对话镜带说话人音频, 无对话镜纯同框互动.
                    spk_k = speaker(script, s)
                    other_k = next((c for c in fc if c != spk_k), spk_k)
                    refs = []
                    for ck in (other_k, spk_k):   # 倾听者先, 说话者后
                        ch = script["characters"][ck]
                        ref = ch.get("face_ref") or ch.get("body_ref")
                        if not ref or not os.path.exists(ref):
                            raise RuntimeError(f"双人同框缺角色 {ck} 的 ref: {ref!r}")
                        refs.append(ref)
                    kw.update({"mode": "dual", "ref_images": refs})
                    if lt == LINE_TYPE_DIALOG:
                        # 双人对话: 说话人音频 + 对话 prompt
                        line = clean_line(shot(script, s)["line"])
                        if not (line or "").strip():
                            print(f"[ERR] shot{s:02d} dialog 镜但 line 为空", flush=True)
                            fail.append(s)
                            continue
                        tmp_voice = os.path.join(out, f"_dialog_voice_{s:02d}.mp3")
                        if not os.path.exists(tmp_voice):
                            synthesize(line, tmp_voice,
                                       voice=character(script, s)["voice_id"], speed=1.0)
                        _pad_audio_to_duration(tmp_voice, dur)   # 补静音到镜头时长
                        kw["audio"] = tmp_voice
                        # 官方六段式 (默认) / legacy 旧中文指令 (回退)
                        prompt = (_build_dual_dialog_prompt(script, s)
                                  if args.prompt_style == "legacy"
                                  else h3prompt.build_dual_dialog_prompt(script, s)) + FACE_POS
                    else:
                        # 双人无对话 (voiceover/ambient): 纯同框互动.
                        # ⚠ 实测无音频只渲 1 人, 喂静音音频激活双人场景.
                        silent_voice = os.path.join(out, f"_dialog_voice_{s:02d}.mp3")
                        if not os.path.exists(silent_voice):
                            _make_silent_audio(silent_voice, dur)
                        kw["audio"] = silent_voice
                        prompt = (_build_dual_scene_prompt(script, s)
                                  if args.prompt_style == "legacy"
                                  else h3prompt.build_dual_scene_prompt(script, s, kind=lt)) + FACE_POS
                elif not is_buffer and lt == LINE_TYPE_DIALOG:
                    # 单人对话镜: 台词 TTS → 喂 H3 驱动口型.
                    # ⚠ 落 i2v out 目录 (不进 voice_dir), 组装靠 has_voice 判路由时
                    #   对话镜 has_voice=False → 保留 H3 内嵌语音, 不重复叠 TTS.
                    line = clean_line(shot(script, s)["line"])
                    if not (line or "").strip():
                        print(f"[ERR] shot{s:02d} dialog 镜但 line 为空", flush=True)
                        fail.append(s)
                        continue
                    tmp_voice = os.path.join(out, f"_dialog_voice_{s:02d}.mp3")
                    if not os.path.exists(tmp_voice):
                        synthesize(line, tmp_voice,
                                   voice=character(script, s)["voice_id"], speed=1.0)
                    _pad_audio_to_duration(tmp_voice, dur)   # 补静音到镜头时长
                    kw["audio"] = tmp_voice
                    # R2V A/B (2026-08-19): identity_ref → refs=[kf,定妆照] 双参考,
                    # prompt 用双参考变体 (P1 构图锚 / P2 身份锚, 脸跟 P2)
                    if ident_ref and os.path.exists(ident_ref):
                        kw["identity_ref"] = ident_ref
                    # 官方六段式 (默认): 覆盖外层 legacy prompt
                    if args.prompt_style == "official":
                        prompt = (h3prompt.build_dialog_prompt_identity(script, s)
                                  if kw.get("identity_ref")
                                  else h3prompt.build_dialog_prompt(script, s))
                        if s not in no_face:
                            prompt += FACE_POS
                elif not is_buffer and args.prompt_style == "official":
                    # 单人 voiceover/ambient (h3): 官方六段式沉默 prompt (⛔ 不写旁白字样)
                    prompt = (h3prompt.build_voiceover_prompt(script, s)
                              if lt == LINE_TYPE_VOICEOVER
                              else h3prompt.build_ambient_prompt(script, s))
                    if s not in no_face:
                        prompt += FACE_POS
            else:
                # 非 h3 provider (seedance/ltx): 保持 legacy prompt + seedance kw
                kw.update({"ratio": "9:16", "resolution": args.resolution})
            # 尾帧衔接 (--tail-link, P0): 单人无对白镜用上镜输出尾帧作首帧.
            # ⚠ 只对 H3 单人 ImageToVideo (非对白/非双人) 生效 — 对白/双人的 image 是身份参考,
            #    尾帧会漂身份. 上镜 mp4 缺 → 优雅回退 keyframe.
            if args.tail_link and PROVIDER == "h3_local" and lt != LINE_TYPE_DIALOG:
                fc_now = [c for c in (frame_chars(script, s) or [])
                          if c in script["characters"]]
                if len(fc_now) < 2:
                    prev_mp4 = os.path.join(out, f"{prefix}{s - 1:02d}.mp4")
                    if os.path.exists(prev_mp4):
                        tail_png = os.path.join(out, f"_tail_{s - 1:02d}.png")
                        if not os.path.exists(tail_png):
                            _extract_last_frame(prev_mp4, tail_png)
                        src = tail_png
                        print(f"[tail-link] shot{s:02d} 首帧=shot{s - 1:02d} 尾帧",
                              flush=True)
            generate(src, prompt, out_path, **kw)
            ok.append(s)
        except Exception as e:
            print(f"[FAIL] shot{s:02d}: {e}", flush=True)
            fail.append(s)
    print(f"\n[done] ok={ok} skip={skip} fail={fail}", flush=True)
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())