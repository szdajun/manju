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


def generate(image, prompt, out, **kw):
    """按 PROVIDER 派发(目前仅 seedance; kling/minimax 待 stub)。"""
    if PROVIDER == "seedance":
        return seedance_i2v(image, prompt, out, **kw)
    raise NotImplementedError(f"视频生成 provider {PROVIDER!r} 未实现 (seedance)")
