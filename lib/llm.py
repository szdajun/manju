"""lib/llm.py — 可插拔 LLM provider (2026-08-15).

CLAUDE.md "可插拔 provider" 架构: LLM 也应可切换 (剧本/实体提取/后续都用它).

provider 由 MANJU_LLM 环境变量切换 (默认 minimax):
  minimax: MiniMax-M2 (api.minimaxi.com, .env MINIMAX_API_KEY) — 豆包故障时主用
  doubao:  豆包 (scripts/_doubao_reasoning.ask_doubao) — 原默认, 端点 404 待修时备用

⚠ 豆包 `ask_doubao` 只认一个端点 (勿猜模型名); MiniMax 需遵守内容安全 (错误 1027).
"""
import json
import os
import urllib.request


def _load_envfile():
    """零依赖读 manju/.env 注入 os.environ (不覆盖已有). key 不入 git."""
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.exists(p):
        return
    with open(p, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#") or "=" not in ln:
                continue
            k, _, v = ln.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_envfile()


def ask_llm(prompt: str, *, max_tokens: int = 1500, temperature: float = 0.4,
            provider: str | None = None) -> str:
    """统一 LLM 入口. provider: minimax (默认) | doubao."""
    p = (provider or os.environ.get("MANJU_LLM", "minimax")).lower()
    if p == "doubao":
        from scripts._doubao_reasoning import ask_doubao  # scripts 路径
        return ask_doubao(prompt, max_tokens=max_tokens, temperature=temperature)
    if p == "minimax":
        return _ask_minimax(prompt, max_tokens=max_tokens, temperature=temperature)
    raise ValueError(f"未知 LLM provider: {provider!r} (minimax | doubao)")


def _ask_minimax(prompt: str, *, max_tokens: int, temperature: float) -> str:
    """MiniMax-M2 单轮对话 (OpenAI 兼容 /v1/text/chatcompletion_v2)."""
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise RuntimeError("MiniMax LLM 缺 MINIMAX_API_KEY; 填 manju/.env")
    base = os.environ.get("MINIMAX_BASE", "https://api.minimaxi.com").rstrip("/")
    payload = {
        "model": "MiniMax-M2",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    req = urllib.request.Request(
        f"{base}/v1/text/chatcompletion_v2",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=120).read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"MiniMax LLM HTTP {e.code}: {e.read().decode()[:300]}") from None
    choices = resp.get("choices") or []
    if not choices:
        raise RuntimeError(f"MiniMax LLM 无 choices: {str(resp)[:200]}")
    return (choices[0].get("message") or {}).get("content", "").strip()
