"""Run one Wan2.2 TI2V-5B short image-to-video validation via ComfyUI.

Run: uv run python scripts/ep05/wan2_2_ti2v_test.py
"""
from __future__ import annotations

import json
import shutil
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

COMFYUI = "http://127.0.0.1:8188"
COMFYUI_ROOT = Path(r"../ComfyUI")
START_IMAGE = Path(r"../manju/output/ep05_poc_v1/keyframes/kf_00.png")
OUTPUT = Path(r"../manju/output/ep05_poc_v1/motion/wan2_2_test.mp4")
MODEL = "Wan2_2-TI2V-5B_fp8_e4m3fn_scaled_KJ.safetensors"
VAE = "Wan2_2_VAE_bf16.safetensors"
T5 = "umt5-xxl-enc-fp8_e4m3fn.safetensors"
PROMPT = (
    "anime cartoon child character, asian girl age 6-8, red twin tails hair, "
    "red sleeveless crop top, white loose baggy pants, white sneakers, dancing "
    "happily in soft pastel park, gentle motion, masterpiece, 24fps"
)
NEGATIVE = "blurry, low quality, deformed face, extra limbs, hat, sunglasses, dark"
WIDTH, HEIGHT, NUM_FRAMES, FPS = 832, 480, 24, 12


def request_json(path: str, data: bytes | None = None, content_type: str = "application/json") -> dict:
    request = urllib.request.Request(
        f"{COMFYUI}{path}", data=data, headers={"Content-Type": content_type}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def upload_image(path: Path) -> str:
    boundary = f"----wan22{uuid.uuid4().hex}"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
        f"filename=\"{path.name}\"\r\nContent-Type: image/png\r\n\r\n"
    ).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    result = request_json("/upload/image", body, f"multipart/form-data; boundary={boundary}")
    return result["name"]


def build_workflow(uploaded_image: str) -> dict:
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": uploaded_image}},
        "2": {
            "class_type": "WanVideoModelLoader",
            "inputs": {
                "model": MODEL,
                "base_precision": "bf16",
                "quantization": "disabled",
                "load_device": "offload_device",
                "attention_mode": "sdpa",
            },
        },
        "3": {
            "class_type": "WanVideoVAELoader",
            "inputs": {"model_name": VAE, "precision": "bf16", "use_cpu_cache": False},
        },
        "4": {
            "class_type": "LoadWanVideoT5TextEncoder",
            "inputs": {
                "model_name": T5,
                "precision": "bf16",
                "load_device": "offload_device",
                "quantization": "fp8_e4m3fn",
            },
        },
        "5": {
            "class_type": "WanVideoTextEncode",
            "inputs": {
                "positive_prompt": PROMPT,
                "negative_prompt": NEGATIVE,
                "t5": ["4", 0],
                "model_to_offload": ["2", 0],
                "force_offload": True,
                "device": "gpu",
            },
        },
        "6": {
            "class_type": "WanVideoImageToVideoEncode",
            "inputs": {
                "width": WIDTH,
                "height": HEIGHT,
                "num_frames": NUM_FRAMES,
                "noise_aug_strength": 0.0,
                "start_latent_strength": 1.0,
                "end_latent_strength": 1.0,
                "force_offload": True,
                "vae": ["3", 0],
                "start_image": ["1", 0],
                "tiled_vae": False,
            },
        },
        "7": {
            "class_type": "WanVideoSampler",
            "inputs": {
                "model": ["2", 0],
                "image_embeds": ["6", 0],
                "text_embeds": ["5", 0],
                "steps": 12,
                "cfg": 5.0,
                "shift": 8.0,
                "seed": 20260724,
                "force_offload": True,
                "scheduler": "unipc",
                "riflex_freq_index": 0,
                "rope_function": "comfy",
            },
        },
        "8": {
            "class_type": "WanVideoDecode",
            "inputs": {
                "vae": ["3", 0],
                "samples": ["7", 0],
                "enable_vae_tiling": False,
                "tile_x": 272,
                "tile_y": 272,
                "tile_stride_x": 144,
                "tile_stride_y": 128,
            },
        },
        "9": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["8", 0],
                "frame_rate": FPS,
                "loop_count": 0,
                "filename_prefix": "ep05_wan2_2_test",
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": True,
                "trim_to_audio": False,
                "pingpong": False,
                "save_output": True,
            },
        },
    }


def invoke_and_wait(workflow: dict, timeout_seconds: int = 3600) -> Path:
    response = request_json("/prompt", json.dumps({"prompt": workflow}).encode())
    if "error" in response:
        raise RuntimeError(json.dumps(response, ensure_ascii=False, indent=2))
    prompt_id = response["prompt_id"]
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        history = request_json(f"/history/{prompt_id}")
        if prompt_id in history:
            record = history[prompt_id]
            status = record.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(json.dumps(status, ensure_ascii=False, indent=2))
            outputs = record.get("outputs", {}).get("9", {})
            entries = outputs.get("gifs") or outputs.get("videos") or []
            if entries:
                item = entries[0]
                base = COMFYUI_ROOT / ("output" if item.get("type") == "output" else "temp")
                return base / item.get("subfolder", "") / item["filename"]
        time.sleep(5)
    raise TimeoutError(f"ComfyUI prompt {prompt_id} exceeded {timeout_seconds}s")


def main() -> int:
    if not START_IMAGE.is_file():
        raise FileNotFoundError(START_IMAGE)
    request_json("/system_stats")
    uploaded = upload_image(START_IMAGE)
    source = invoke_and_wait(build_workflow(uploaded))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, OUTPUT)
    print(f"Output: {OUTPUT}")
    print(f"Bytes: {OUTPUT.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
