"""Stage B batch: 8 张林夏立绘批量跑裤区 inpaint (写入 train_data 强学习灰侧线).

每个 base 立绘 -> 一次 pants inpaint (denoise 0.5 + pants mask) -> 输出新立绘

用法:
  ../manju/.venv/Scripts/python.exe scripts/_batch_inpaint_pants.py

产物:
  ../manju/output/ep03_lihui/inpaint_pants/linxia_inpaint_{stem}.png
  其中 stem 是原 stem (如 manju_linxia_real_00017_)
"""
import os
import sys
import json
import time
import urllib.request
import urllib.error

MANJU = "../manju"
sys.path.insert(0, MANJU)
from lib.comfyui_client import ComfyUIClient  # noqa

from PIL import Image

SERVER = "http://127.0.0.1:8188"
SRC = "../manju/output/ep03_lihui"
DST = "../manju/output/ep03_lihui/inpaint_pants"
PANTS_MASK = "../manju/output/ep03_lihui/inpaint_pants_mask.png"

PROMPT = "a photo of BLACK athletic pants with thin GRAY STRIPES on the outside of the legs running down the full length of the pants, photorealistic, sharp"
NEG = "blurry, lowres, jpeg artifacts, text, watermark, cartoon, anime, painting, deformed hands, white pants, light pants, gray pants"


def make_pants_mask(mask_path):
    img = Image.open(mask_path)
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    px = mask.load()
    for y in range(int(h * 0.55), int(h * 0.95)):
        for x in range(int(w * 0.20), int(w * 0.32)):
            px[x, y] = 255
        for x in range(int(w * 0.68), int(w * 0.80)):
            px[x, y] = 255
    mask.save(mask_path)


def build_workflow(base_name, mask_name, out_name, prompt, denoise=0.5):
    return {
        "ckpt": {"class_type": "CheckpointLoaderSimple",
                 "inputs": {"ckpt_name": "DreamShaper_8_pruned.safetensors"}},
        "base": {"class_type": "LoadImage", "inputs": {"image": base_name}},
        "mask": {"class_type": "LoadImageMask", "inputs": {"image": mask_name,
                                                            "channel": "red"}},
        "vae": {"class_type": "VAEEncodeForInpaint",
                "inputs": {"pixels": ["base", 0], "vae": ["ckpt", 2],
                           "mask": ["mask", 0], "grow_mask_by": 4}},
        "noise_mask": {"class_type": "SetLatentNoiseMask",
                       "inputs": {"samples": ["vae", 0], "mask": ["mask", 0]}},
        "pos": {"class_type": "CLIPTextEncode",
                "inputs": {"clip": ["ckpt", 1], "text": prompt}},
        "neg": {"class_type": "CLIPTextEncode",
                "inputs": {"clip": ["ckpt", 1], "text": NEG}},
        "ksampler": {"class_type": "KSampler",
                     "inputs": {"model": ["ckpt", 0],
                                "positive": ["pos", 0],
                                "negative": ["neg", 0],
                                "latent_image": ["noise_mask", 0],
                                "seed": 42, "steps": 28, "cfg": 7.5,
                                "sampler_name": "dpmpp_2m", "scheduler": "karras",
                                "denoise": denoise}},
        "decode": {"class_type": "VAEDecode",
                   "inputs": {"samples": ["ksampler", 0], "vae": ["ckpt", 2]}},
        "save": {"class_type": "SaveImage",
                 "inputs": {"images": ["decode", 0], "filename_prefix": out_name}},
    }


def submit(client, wf):
    payload = {"prompt": wf, "client_id": "batch-inpaint-pants"}
    req = urllib.request.Request(
        f"{SERVER}/prompt", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        r = urllib.request.urlopen(req, timeout=30)
        return json.loads(r.read().decode()).get("prompt_id")
    except urllib.error.HTTPError as e:
        print(f"[FAIL submit] {e.code} body={e.read().decode()[:200]}")
        return None


def wait(prompt_id, timeout_sec=120):
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            r = urllib.request.urlopen(f"{SERVER}/history/{prompt_id}", timeout=10)
            h = json.loads(r.read().decode())
            entry = list(h.values())[0]
            if entry.get("status", {}).get("completed"):
                return entry
        except Exception:
            pass
        time.sleep(3)
    return None


def fetch_save(entry):
    paths = []
    for k, v in entry.get("outputs", {}).items():
        for img in v.get("images", []):
            fn = img.get("filename")
            sub = img.get("subfolder", "")
            url = f"{SERVER}/view?filename={fn}&subfolder={sub}&type=output"
            local = os.path.join(DST, fn)
            urllib.request.urlretrieve(url, local)
            paths.append(local)
    return paths


def main():
    os.makedirs(DST, exist_ok=True)
    # 确保 pants mask 存在
    if not os.path.exists(PANTS_MASK):
        # 用 linxia_real.png 作 basis (其尺寸 512x912)
        make_pants_mask(PANTS_MASK)

    c = ComfyUIClient(SERVER)
    mask_name = c.upload_image(PANTS_MASK)
    print(f"[uploaded mask] {mask_name}")

    # 选定原图: linxia_real.png + 8 张候选 (seed 00017_~00024_ 各跑一次)
    bases = ["linxia_real.png"] + [f"manju_linxia_real_{n:05d}_.png" for n in range(17, 25)]
    bases = [b for b in bases if os.path.exists(os.path.join(SRC, b))]
    print(f"[bases] {len(bases)} images")

    for base in bases:
        base_path = os.path.join(SRC, base)
        try:
            base_name = c.upload_image(base_path)
        except Exception as e:
            print(f"[skip upload] {base}: {e}")
            continue
        out_name = "inpaint_pants_" + base.replace(".png", "")
        wf = build_workflow(base_name, mask_name, out_name, PROMPT, denoise=0.5)
        pid = submit(c, wf)
        if not pid:
            continue
        print(f"[submitted] {base} -> {pid}")
        entry = wait(pid, timeout_sec=180)
        if entry:
            paths = fetch_save(entry)
            print(f"[done] {base} -> {paths}")
        else:
            print(f"[timeout] {base}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
