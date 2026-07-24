"""Generate the kid_dance SD1.5 (Counterfeit-V3.0) reference image through ComfyUI.

T2.b fix: SDXL OOM on RTX 3060 12GB. SD1.5 Counterfeit is anime-native (~4GB),
plenty of VRAM for LoRA training + sampling.
"""

from __future__ import annotations
import httpx
import sys
import time

SERVER = "http://127.0.0.1:8188"
PROMPT = (
    "masterpiece, best quality, anime cartoon style, 1girl, chibi, child, little girl age 6, "
    "very young child, baby face, round face, huge eyes, small body, "
    "asian little girl, large head small body chibi proportion 1:2, "
    "bright red sleeveless crop top, crimson red sleeveless tank top, white loose baggy harem pants, white sneakers, "
    "soft pastel park background, smug confident expression, one eyebrow raised crooked smile, "
    "cowboy shot from knees up, arms relaxed at sides, illustration"
)
NEG = (
    "mature, adult, teenager, teen, sensual, sexy, revealing, navel, abs, "
    "realistic, photograph, 3d, ugly, deformed, extra limbs, hat, sunglasses, mature body, "
    "big breast, slim waist"
)
SEED = 23
CHECKPOINT = "Counterfeit-V3.0_fix_fp16.safetensors"
WIDTH = 512
HEIGHT = 768
SEED = 11
STEPS = 28
CFG = 7.0

WF = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": CHECKPOINT},
    },
    "2": {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["1", 1], "text": PROMPT},
    },
    "3": {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["1", 1], "text": NEG},
    },
    "4": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": WIDTH, "height": HEIGHT, "batch_size": 1},
    },
    "5": {
        "class_type": "KSampler",
        "inputs": {
            "model": ["1", 0],
            "positive": ["2", 0],
            "negative": ["3", 0],
            "seed": SEED,
            "steps": STEPS,
            "cfg": CFG,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
            "latent_image": ["4", 0],
        },
    },
    "6": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
    },
    "7": {
        "class_type": "SaveImage",
        "inputs": {"images": ["6", 0], "filename_prefix": "ep05_ref"},
    },
}


def main() -> int:
    r = httpx.post(f"{SERVER}/prompt", json={"prompt": WF}, timeout=30)
    r.raise_for_status()
    pid = r.json()["prompt_id"]
    print(f"SUBMITTED: {pid} (ckpt={CHECKPOINT}, {WIDTH}x{HEIGHT}, seed={SEED})")

    start = time.time()
    while time.time() - start < 180:
        h = httpx.get(f"{SERVER}/history/{pid}", timeout=10).json()
        if pid in h and h[pid].get("outputs"):
            out = h[pid]["outputs"]["7"]["images"][0]
            print(f"WROTE_FN:{out['filename']} subfolder={out.get('subfolder','')}")
            return 0
        time.sleep(3)
    print("TIMEOUT 180s", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
