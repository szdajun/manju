"""Generate four SDXL keyframes through ComfyUI with LoRA and IPAdapter face lock."""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[2]
COMFYUI = "http://127.0.0.1:8188"
COMFYUI_DIR = Path("../ComfyUI")
CHECKPOINT = "sd_xl_base_1.0.safetensors"
LORA_NAME = "kid_dance_lora.safetensors"
LORA_WEIGHT = 0.6
IPADAPTER_WEIGHT = 0.85
REF_PATH = ROOT / "assets/characters/kid_dance/ref.png"
REF_INPUT_NAME = "ep05_kid_dance_ref.png"
PROMPTS_DIR = ROOT / "prompts/ep05"
OUT_DIR = ROOT / "output/ep05_poc_v1/keyframes"


def build_workflow(data: dict) -> dict:
    """Build the installed ComfyUI_IPAdapter_plus SDXL API workflow."""
    return {
        "ckpt": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": CHECKPOINT},
        },
        "lora": {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["ckpt", 0],
                "clip": ["ckpt", 1],
                "lora_name": LORA_NAME,
                "strength_model": LORA_WEIGHT,
                "strength_clip": LORA_WEIGHT,
            },
        },
        "ref": {"class_type": "LoadImage", "inputs": {"image": REF_INPUT_NAME}},
        "ip_loader": {
            "class_type": "IPAdapterUnifiedLoader",
            "inputs": {"model": ["lora", 0], "preset": "PLUS FACE (portraits)"},
        },
        "ip_apply": {
            "class_type": "IPAdapterAdvanced",
            "inputs": {
                "model": ["ip_loader", 0],
                "ipadapter": ["ip_loader", 1],
                "image": ["ref", 0],
                "weight": IPADAPTER_WEIGHT,
                "weight_type": "linear",
                "combine_embeds": "concat",
                "start_at": 0.0,
                "end_at": 1.0,
                "embeds_scaling": "V only",
            },
        },
        "positive": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["lora", 1], "text": data["prompt"]},
        },
        "negative": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["lora", 1], "text": data["negative"]},
        },
        "latent": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
        },
        "sampler": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["ip_apply", 0],
                "positive": ["positive", 0],
                "negative": ["negative", 0],
                "seed": data["seed"],
                "steps": 25,
                "cfg": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "latent_image": ["latent", 0],
            },
        },
        "decode": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["sampler", 0], "vae": ["ckpt", 2]},
        },
        "save": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["decode", 0],
                "filename_prefix": f"ep05_kf_{data['shot']:02d}",
            },
        },
    }


def wait_for_image(client: httpx.Client, prompt_id: str, timeout_s: int = 600) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        response = client.get(f"{COMFYUI}/history/{prompt_id}", timeout=15)
        response.raise_for_status()
        history = response.json().get(prompt_id)
        if history:
            if history.get("status", {}).get("status_str") == "error":
                raise RuntimeError(f"ComfyUI execution failed: {history['status']}")
            images = history.get("outputs", {}).get("save", {}).get("images", [])
            if images:
                return images[0]
        time.sleep(2)
    raise TimeoutError(f"ComfyUI did not finish {prompt_id} within {timeout_s}s")


def main() -> int:
    if not REF_PATH.is_file():
        raise FileNotFoundError(REF_PATH)
    prompt_paths = [PROMPTS_DIR / f"kf_{shot:02d}.yaml" for shot in range(4)]
    missing = [path for path in prompt_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing prompt YAML files: {missing}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    comfy_ref = COMFYUI_DIR / "input" / REF_INPUT_NAME
    shutil.copy2(REF_PATH, comfy_ref)

    with httpx.Client(trust_env=False) as client:
        stats = client.get(f"{COMFYUI}/system_stats", timeout=10)
        stats.raise_for_status()
        for yaml_path in prompt_paths:
            with yaml_path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle)
            response = client.post(
                f"{COMFYUI}/prompt", json={"prompt": build_workflow(data)}, timeout=30
            )
            if response.status_code >= 400:
                raise RuntimeError(f"ComfyUI rejected {yaml_path.name}: {response.text}")
            prompt_id = response.json()["prompt_id"]
            print(f"SUBMITTED {yaml_path.name}: {prompt_id}", flush=True)
            image = wait_for_image(client, prompt_id)
            params = {
                "filename": image["filename"],
                "subfolder": image.get("subfolder", ""),
                "type": image.get("type", "output"),
            }
            image_response = client.get(f"{COMFYUI}/view", params=params, timeout=30)
            image_response.raise_for_status()
            out_path = OUT_DIR / f"kf_{data['shot']:02d}.png"
            out_path.write_bytes(image_response.content)
            print(f"WROTE {out_path} ({len(image_response.content)} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
