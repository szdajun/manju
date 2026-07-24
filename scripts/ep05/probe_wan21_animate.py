"""Probe ComfyUI for Wan2.1-Animate-14B native node + weight availability (P2 / T4).

CP2 pre-flight check. Exits 0 only when both:
  - WanVideoAnimateEmbeds + WanAnimateToVideo + WanVideoModelLoader / WanVideoVAELoader
    nodes are registered (ComfyUI-WanVideoWrapper installed).
  - At least one Wan2.1-Animate-14B diffusion model checkpoint is present under
    ../ComfyUI/models/diffusion_models/.

Writes a JSON summary next to this script and touches
  output/ep05_poc_v1/.probe_ok   (CP2 pass signal)
on success.

Run:  uv run python scripts/ep05/probe_wan21_animate.py
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

COMFYUI = "http://127.0.0.1:8188"
ANIMATE_NODES = ("WanVideoAnimateEmbeds", "WanAnimateToVideo", "WanVideoModelLoader", "WanVideoVAELoader")
DIFFUSION_DIR = Path(r"../ComfyUI/models/diffusion_models")
PLACEHOLDER = Path(r"../manju/output/ep05_poc_v1/.probe_ok")
SUMMARY = Path(__file__).parent / "_probe_summary.json"


def fetch_object_info() -> dict:
    raw = urllib.request.urlopen(f"{COMFYUI}/object_info", timeout=10).read()
    return json.loads(raw)


def main() -> int:
    try:
        info = fetch_object_info()
    except Exception as exc:  # noqa: BLE001
        print(f"ComfyUI {COMFYUI} unreachable: {exc}")
        return 1

    present = {n: n in info for n in ANIMATE_NODES}
    weights = sorted(p.name for p in DIFFUSION_DIR.glob("*.safetensors")) if DIFFUSION_DIR.exists() else []
    animate_w = [w for w in weights if "Animate" in w]

    summary = {"nodes_present": present, "weights": weights, "animate_weights": animate_w}
    SUMMARY.write_text(json.dumps(summary, indent=2))

    print("WanVideoWrapper nodes:")
    for n, ok in present.items():
        print(f"  {'OK' if ok else 'MISSING'}  {n}")
    print(f"Wan2.1-Animate diffusion weights in {DIFFUSION_DIR}: {animate_w or '(none)'}")

    nodes_ok = all(present.values())
    weights_ok = bool(animate_w)
    if nodes_ok and weights_ok:
        PLACEHOLDER.parent.mkdir(parents=True, exist_ok=True)
        PLACEHOLDER.touch()
        print(f"OK -> wrote {PLACEHOLDER}")
        return 0

    if not nodes_ok:
        print(
            "FAIL: WanVideoWrapper nodes missing. Install with:\n"
            "  cd ../ComfyUI/custom_nodes && "
            "git clone https://github.com/kijai/ComfyUI-WanVideoWrapper\n"
            "then restart ComfyUI 8188."
        )
    if not weights_ok:
        print(
            f"FAIL: no Wan2.1-Animate diffusion checkpoint under {DIFFUSION_DIR}.\n"
            "  huggingface-cli download Wan-AI/Wan2.1-Animate-14B "
            f"--local-dir {DIFFUSION_DIR}\n"
            "(HF anonymous download was 401/404 at probe time; gated)"
        )
    return 2


if __name__ == "__main__":
    sys.exit(main())