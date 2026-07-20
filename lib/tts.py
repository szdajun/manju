"""TTS provider 抽象。

MVP 用 edge-tts(微软免费神经语音, 无 key 本地跑); 火山/MiniMax 留 stub, 拿到 key 填实现即可。
切 provider 只改环境变量 MANJU_TTS, 主干(voice_ep01 / make_ep01)零改动。

打工牛马主角 ~30 男, 疲惫丧音色 → zh-CN-YunxiNeural(年轻男声) + rate=-5%(略拖) + pitch=-2Hz(略低)。
"""
import asyncio
import json
import os
import urllib.request

DEFAULT_VOICE = "zh-CN-YunxiNeural"
DEFAULT_RATE = "-5%"       # 略拖 → 疲惫
DEFAULT_PITCH = "-2Hz"     # 略低 → 丧


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

PROVIDER = os.environ.get("MANJU_TTS", "edge").lower()  # edge | volcano | minimax


async def _edge(text, out, voice, rate, pitch):
    import edge_tts
    await edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).save(out)


def tts_edge(text, out, voice=DEFAULT_VOICE, rate=DEFAULT_RATE, pitch=DEFAULT_PITCH):
    asyncio.run(_edge(text, out, voice, rate, pitch))


def tts_volcano(text, out, **kw):
    raise NotImplementedError("火山 TTS 待 API key; 先 MANJU_TTS=edge 出草版")


def tts_minimax(text, out, voice=None, speed=0.9, vol=1.0, pitch=0):
    """MiniMax T2A v2。返回 out。

    国际站 api.minimaxi.com: sk-* key + Bearer, 端点 /v1/t2a_v2, 不要 GroupId。
    国内站 api.minimaxi.chat: JWT key + ?GroupId= 。MINIMAX_BASE 切换。
    data.audio 为 hex 编码 mp3, bytes.fromhex 解出。speed<1 略拖 → 疲惫。
    """
    import urllib.error
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise RuntimeError("MiniMax TTS 缺 MINIMAX_API_KEY; 填 manju/.env")
    voice = voice or os.environ.get("MINIMAX_VOICE", "male-qn-qingse")  # 青涩青年男声
    model = os.environ.get("MINIMAX_MODEL", "speech-02-hd")
    base = os.environ.get("MINIMAX_BASE", "https://api.minimaxi.com").rstrip("/")
    gid = os.environ.get("MINIMAX_GROUP_ID")
    url = f"{base}/v1/t2a_v2"
    if base.endswith(".chat") and gid:          # 国内站才挂 GroupId
        url += f"?GroupId={gid}"
    body = {
        "model": model,
        "text": text,
        "stream": False,
        "voice_setting": {"voice_id": voice, "speed": speed, "vol": vol, "pitch": pitch},
        "audio_setting": {"sample_rate": 32000, "format": "mp3", "channel": 1},
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            resp = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_txt = e.read().decode("utf-8", "replace")[:400]
        raise RuntimeError(f"MiniMax TTS HTTP {e.code}: {body_txt}") from None
    code = (resp.get("base_resp") or {}).get("status_code", 0)
    if code != 0:
        msg = (resp.get("base_resp") or {}).get("status_msg", "?")
        raise RuntimeError(f"MiniMax TTS 错误 {code}: {msg} (1027=内容安全 1004=鉴权)")
    audio_hex = (resp.get("data") or {}).get("audio")
    if not audio_hex:
        raise RuntimeError(f"MiniMax TTS 无 audio 字段: {resp}")
    with open(out, "wb") as f:
        f.write(bytes.fromhex(audio_hex))
    return out


def synthesize(text, out, **kw):
    """按 PROVIDER 派发。空文本抛错。返回 out。"""
    if not text or not text.strip():
        raise ValueError("TTS 空文本")
    if PROVIDER == "edge":
        tts_edge(text, out, **kw)
    elif PROVIDER == "volcano":
        tts_volcano(text, out, **kw)
    elif PROVIDER == "minimax":
        tts_minimax(text, out, **kw)
    else:
        raise ValueError(f"未知 TTS provider: {PROVIDER!r} (edge|volcano|minimax)")
    return out
