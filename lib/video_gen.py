"""视频生成 provider 抽象 (i2v)。镜像 lib/tts.py 模式。

MVP: 火山方舟 Seedance (doubao-seedance) i2v 首帧。本地 Wan2.2-5B 在 12GB 死路
(access violation + OOM catch-22, 见 memory manju-comfyui-torch-pin), 转 API。
切 provider 改 MANJU_VIDEOGEN, 主干零改动。

Seedance i2v 流程(异步): POST 提交任务 → task_id → 轮询 → video_url(24h 有效) → 下载转存。
API 参考: https://www.volcengine.com/docs/82379/1520757

注意: Seedance 2.0 系列**不支持含真人人脸的参考图**(须走授权/虚拟人像/模型自有产物),
且需余额 ≥¥200。故默认 1.0 pro(支持 i2v 首帧 + 无真人脸禁令 + 无门槛), 2.0 留作升级备选。
"""
import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request

ARK_BASE = os.environ.get("ARK_BASE", "https://ark.cn-beijing.volces.com/api/v3")
# 1.0 pro: 支持 i2v 首帧, 无 ¥200 门槛, 无 2.0 系列的真人脸禁令。版本号错时 API 报可用列表。
DEFAULT_MODEL = os.environ.get("SEEDANCE_MODEL", "doubao-seedance-1-0-pro-250528")


def _load_envfile():
    """零依赖读 manju/.env 注入 os.environ(不覆盖已有)。key 不入 git。"""
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.exists(p):
        return
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_envfile()

PROVIDER = os.environ.get("MANJU_VIDEOGEN", "seedance").lower()  # seedance | (未来 kling/minimax)


def _img_to_data_url(path):
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _ark_headers():
    api_key = os.environ.get("ARK_API_KEY")
    if not api_key:
        raise RuntimeError("Seedance 缺 ARK_API_KEY; 填 manju/.env")
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _ark_call(req, timeout=60):
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:800]
        raise RuntimeError(f"ARK HTTP {e.code}: {body}") from None


def submit_i2v(image_path, prompt, model=DEFAULT_MODEL, duration=5, ratio="9:16",
               resolution="480p", seed=-1):
    """提交 i2v 首帧任务, 返回 task_id。"""
    body = {
        "model": model,
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": _img_to_data_url(image_path)},
             "role": "first_frame"},
        ],
        "duration": duration,
        "ratio": ratio,
        "resolution": resolution,
        "seed": seed,
        "watermark": False,
        "generate_audio": False,   # 1.0 pro 无声(字段被忽略); 2.0/1.5 生效
    }
    req = urllib.request.Request(
        ARK_BASE + "/contents/generations/tasks",
        data=json.dumps(body).encode("utf-8"),
        headers=_ark_headers(), method="POST")
    resp = _ark_call(req)
    tid = resp.get("id")
    if not tid:
        raise RuntimeError(f"ARK 提交无 task_id: {resp}")
    return tid


def query_task(task_id):
    req = urllib.request.Request(
        ARK_BASE + f"/contents/generations/tasks/{task_id}",
        headers=_ark_headers(), method="GET")
    return _ark_call(req)


def _extract_video_url(resp):
    """兼容返回形态: content.video_url(str|dict) / content[].video_url / 顶层。
    Seedance 1.0 pro 实测: content.video_url 直接是 URL 字符串。"""
    c = resp.get("content")
    if isinstance(c, str) and c:
        return c
    if isinstance(c, dict):
        vu = c.get("video_url")
        if isinstance(vu, str) and vu:
            return vu
        if isinstance(vu, dict) and vu.get("url"):
            return vu["url"]
        if c.get("url"):
            return c["url"]
    if isinstance(c, list):
        for item in c:
            if isinstance(item, str) and item:
                return item
            if not isinstance(item, dict):
                continue
            vu = item.get("video_url")
            if isinstance(vu, str) and vu:
                return vu
            if isinstance(vu, dict) and vu.get("url"):
                return vu["url"]
            if item.get("type") == "video_url" and item.get("url"):
                return item["url"]
    return None


def wait_task(task_id, poll_interval=10, timeout=900):
    """轮询到 succeeded → 返回 video_url; failed/expired 抛错。"""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        resp = query_task(task_id)
        status = resp.get("status", "?")
        if status != last:
            print(f"[seedance] task={task_id} status={status}", flush=True)
            last = status
        if status == "succeeded":
            url = _extract_video_url(resp)
            if not url:
                raise RuntimeError(f"succeeded 但无 video_url: {resp}")
            return url
        if status in ("failed", "expired"):
            raise RuntimeError(f"task {status}: {resp.get('error') or resp}")
        time.sleep(poll_interval)
    raise TimeoutError(f"task {task_id} 超时 {timeout}s, last={last}")


def download(url, out):
    with urllib.request.urlopen(url, timeout=180) as r, open(out, "wb") as f:
        f.write(r.read())
    return out


def seedance_i2v(image, prompt, out, model=DEFAULT_MODEL, duration=5, ratio="9:16",
                 resolution="480p", poll_interval=10, timeout=900):
    """完整 i2v: 提交 → 轮询 → 下载。返回 out。"""
    tid = submit_i2v(image, prompt, model=model, duration=duration, ratio=ratio,
                     resolution=resolution)
    url = wait_task(tid, poll_interval=poll_interval, timeout=timeout)
    download(url, out)
    print(f"[seedance] -> {out}", flush=True)
    return out


# ---- LTX-2.3 本地 i2v (2026-08-06 免费 provider, 12G 显存单段 ~1.1s) ----

def _ffmpeg():
    for cand in (r"~/ffmpeg/ffmpeg.exe", r"~/ffmpeg/bin/ffmpeg.exe"):
        if os.path.exists(cand):
            return cand
    import shutil
    return shutil.which("ffmpeg") or "ffmpeg"


def _extract_last_frame(video_path, out_png):
    """抽视频末帧 (连续递进分段: 下段首帧用上段末帧).

    用 -sseof -0.05 (从末尾 50ms 处) 最稳, 避免 select 的 n 转义问题.
    """
    import subprocess
    subprocess.run(
        [_ffmpeg(), "-y", "-v", "error", "-sseof", "-0.05", "-i", video_path,
         "-frames:v", "1", out_png],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _concat_videos(segs, dst):
    """硬切拼接 LTX 分段 (concat demuxer -c copy 无损, 快)."""
    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as f:
        for s in segs:
            f.write(f"file '{os.path.abspath(s).replace(chr(39), chr(39)+chr(39)+chr(39))}'\n")
        lst = f.name
    try:
        subprocess.run(
            [_ffmpeg(), "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", dst],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        os.unlink(lst)
    return dst


def ltx_local_i2v(image, prompt, out, duration=5, segment_seconds=4.0,
                  segment_frames=97, width=480, height=832, frame_rate=24,
                  server="http://127.0.0.1:8188", seed_base=None):
    """LTX-2.3 本地 i2v (免费). 连续递进分段: 每段 4s (97帧, 12G 显存上限实测),
    用上段末帧作下段首帧. 5s 镜 = 2 段 (4s + 1s).

    Args:
        image: 首帧图路径 (关键帧).
        prompt: 运动描述 (LTX 轻微自然动, 非精确跟动作).
        out: 输出 mp4 路径.
        duration: 目标时长秒 (默认 5).
        segment_seconds/frames: 单段帧参数. ⚠ 实测: 480x832 latent 小,
            i2v 单段可达 97 帧 (4s, 200s), 远超 pose 的 33 帧上限.
        width/height/frame_rate: 分辨率 (对齐关键帧 480x832).
        server: ComfyUI 地址.
        seed_base: 每段 seed = seed_base + 段号 (保证段间略有差异).

    Returns: out. 零 API 成本, 比 Seedance 慢 (~4-5 分钟/镜) 但免费.
    """
    from .comfyui_client import ComfyUIClient, build_ltx23_i2v_workflow

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    client = ComfyUIClient(server)
    # 上传首帧 → 拿 input 名
    frame_name = client.upload_image(image)
    n_segments = max(1, int(-(-duration // segment_seconds)))  # ceil: 5/4=2 段
    base = seed_base if seed_base is not None else 42
    tmp_dir = os.path.join(os.path.dirname(os.path.abspath(out)), "_ltx_segs")
    os.makedirs(tmp_dir, exist_ok=True)

    segs = []
    cur_frame = frame_name
    for i in range(n_segments):
        seg_out = os.path.join(tmp_dir, f"seg{i:02d}.mp4")
        wf = build_ltx23_i2v_workflow(
            cur_frame, prompt, "",
            width=width, height=height, length=segment_frames,
            seed=base + i, frame_rate=frame_rate,
            filename_prefix=f"ltx23_i2v_seg{i:02d}")
        print(f"[ltx23] seg{i+1}/{n_segments} (首帧={cur_frame}) ...", flush=True)
        saved = client.generate(wf, tmp_dir)
        if not saved:
            raise RuntimeError(f"LTX i2v 段 {i} 无输出")
        segs.append(saved[0])
        # 连续递进: 下段首帧 = 本段末帧
        if i < n_segments - 1:
            last_png = os.path.join(tmp_dir, f"last{i:02d}.png")
            _extract_last_frame(segs[-1], last_png)
            cur_frame = client.upload_image(last_png)

    _concat_videos(segs, out)
    # 按目标时长裁剪 (分段会拼出整段倍数时长, 5s 需裁掉多余尾段)
    if duration and os.path.exists(out):
        import subprocess
        _tmp = out + ".tmp.mp4"
        subprocess.run(
            [_ffmpeg(), "-y", "-i", out, "-t", str(duration), "-c", "copy", _tmp],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.replace(_tmp, out)
    print(f"[ltx23] -> {out}  ({n_segments} 段拼接, {duration}s)", flush=True)
    return out


def h3_local_i2v(image, prompt, out, duration=5, width=768, height=1344,
                 mode="single", ref_images=None, seed=None,
                 server="http://127.0.0.1:8188", filename_prefix=None,
                 unet_gguf=None, ref_image_size="match", turbo=False,
                 audio=None, audio_slot="ref_audios", identity_ref=None):
    """H3 本地 i2v (免费, 2026-08-06). 单人 ImageToVideo 或双人 Ref2VA.

    Args:
        image: 单人模式 = 首帧关键帧图; 双人模式 = 主角色脸图 (或 None).
        prompt: 单人 = 场景+运动+风格描述; 双人 = "<Picture 1> 角色A 与 <Picture 2> 角色B, ...".
        out: 输出 mp4 路径.
        duration: 目标时长秒 (默认 5, H3 单段可达 5s+).
        width/height: 分辨率 (默认 768x1344 竖屏, H3 原生).
        mode: "single" (ImageToVideo) | "dual" (Ref2VA 多参考图).
        ref_images: 双人模式的角色参考图列表 (ComfyUI input 名, 每角色一张).
        seed: 采样 seed (默认随机).
        server: ComfyUI 地址.
        filename_prefix: SaveVideo 前缀 (默认 manju_h3 / manju_h3_dual).
        unet_gguf: 覆盖 UNET GGUF (单人 fl2va / 双人 ref2va).
        audio: 台词音频路径 (对话镜, H3 驱动口型). None = 无台词 (环境音).
        audio_slot: "ref_audios" (角色台词) | "ref_video_audios" (参考视频音频).
        identity_ref: 角色定妆照路径 (R2V A/B, 2026-08-19). 单人对话镜时
            refs=[kf, 定妆照] 双参考: P1 锁构图, P2 锁身份. None = 原单参考.

    Returns: out. 零 API 成本, 768x1344 正式档 (25min/段) / 432x768 迭代档 (6.3min/段).
    """
    from .comfyui_client import (
        ComfyUIClient, build_h3_i2v_workflow, build_h3_ref2va_workflow,
        h3_length_from_duration,
    )

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    client = ComfyUIClient(server)
    length = h3_length_from_duration(duration)
    base = seed if seed is not None else 42

    if mode == "dual":
        # 上传每张参考图 → input 名
        if not ref_images:
            raise ValueError("双人模式 (Ref2VA) 需要 ref_images 参考图列表")
        uploaded = [client.upload_image(r) for r in ref_images]
        # 台词音频 (对话镜) → 复制到 ComfyUI input (LoadAudio 读相对路径)
        audio_name = _copy_audio_to_comfy_input(audio) if audio else None
        wf = build_h3_ref2va_workflow(
            uploaded, prompt, width=width, height=height, length=length,
            seed=base, filename_prefix=filename_prefix or "manju_h3_dual",
            unet_gguf=unet_gguf, ref_image_size=ref_image_size, turbo=turbo,
            audio=audio_name, audio_slot=audio_slot,
        )
    else:
        # 单人: 上传首帧关键帧
        frame_name = client.upload_image(image)
        if audio:
            # 对话镜: Ref2VA 单参考图 + 音频驱动口型 (已验证 _h3_single_talk.json 路径;
            # H3 音视频联合生成, 唇型由音频音素驱动). 首帧关键帧作身份参考.
            # identity_ref (R2V A/B): 双参考 [kf, 定妆照], P1 构图 P2 身份.
            audio_name = _copy_audio_to_comfy_input(audio)
            if identity_ref:
                ident_name = client.upload_image(identity_ref)
                refs = [frame_name, ident_name]
            else:
                refs = [frame_name]
            wf = build_h3_ref2va_workflow(
                refs, prompt, width=width, height=height, length=length,
                seed=base, filename_prefix=filename_prefix or "manju_h3",
                unet_gguf=unet_gguf, ref_image_size=ref_image_size,
                turbo=turbo, audio=audio_name, audio_slot=audio_slot,
            )
        else:
            wf = build_h3_i2v_workflow(
                frame_name, prompt, width=width, height=height, length=length,
                seed=base, filename_prefix=filename_prefix or "manju_h3",
                unet_gguf=unet_gguf, turbo=turbo,
            )

    print(f"[h3_local] {mode} {width}x{height} len={length} (上传完成) ...", flush=True)
    # H3 高分辨率 (768x1344) 采样 25min+, 用长 timeout (默认 900s=15min 不够)
    saved = client.generate(wf, os.path.dirname(os.path.abspath(out)), timeout=3600)
    if not saved:
        raise RuntimeError("H3 本地生成无输出")
    # 移动产物到 out (SaveVideo 前缀文件名 vs 目标 out)
    _finalize_h3_output(saved[0], out)
    print(f"[h3_local] -> {out}", flush=True)
    return out


def _copy_audio_to_comfy_input(audio):
    """台词音频 (对话镜) → 复制到 ComfyUI input → 返回 input 文件名 (LoadAudio 读相对路径)."""
    import shutil
    audio_name = os.path.basename(audio)
    manju_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    comfy_input = os.path.join(os.path.dirname(manju_root), "ComfyUI", "input")
    os.makedirs(comfy_input, exist_ok=True)
    shutil.copy(audio, os.path.join(comfy_input, audio_name))
    return audio_name


def _finalize_h3_output(src, out):
    """把 SaveVideo 产物复制到目标 out 路径 (不同目录时)."""
    import shutil
    if os.path.abspath(src) != os.path.abspath(out):
        shutil.copyfile(src, out)


def generate(image, prompt, out, **kw):
    """按 PROVIDER 派发 (seedance=付费API; ltx_local/h3_local=免费本地)."""
    if PROVIDER == "seedance":
        return seedance_i2v(image, prompt, out, **kw)
    if PROVIDER == "ltx_local":
        return ltx_local_i2v(image, prompt, out, **kw)
    if PROVIDER == "h3_local":
        return h3_local_i2v(image, prompt, out, **kw)
    raise NotImplementedError(f"视频生成 provider {PROVIDER!r} 未实现 "
                              f"(seedance / ltx_local / h3_local)")
