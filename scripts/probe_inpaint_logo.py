"""Stage B: ComfyUI inpaint 探针 — 用 linxia_real.png 基底 + mask 左胸 logo 区
   inpaint "small white arc logo" 验证 inpaint 节点链可工作.

预期:
  - 写出 inpaint logo mask (白 1 黑 0 单色 PNG, 中心 cx~190, cy~340 区 80x80)
  - 提交 VAEEncodeForInpaint workflow
  - 出图 logo 区出现白弧标志 (其余身体应与原图几乎一致)

用法:
  ../manju/.venv/Scripts/python.exe scripts/_probe_inpaint_logo.py
"""
import os
import sys

MANJU = "../manju"
sys.path.insert(0, MANJU)
from lib.comfyui_client import ComfyUIClient  # noqa

import json
import urllib.request
from PIL import Image
import numpy as np

LINXIA_BASE = "../manju/output/ep03_lihui/linxia_real.png"
LOG = "../manju/output/ep03_lihui/inpaint_logo_mask.png"
OUT_DIR = "../manju/output/ep03_lihui/_inpaint_pilot"
SERVER = "http://127.0.0.1:8188"


def build_logo_mask(base_path, out_path):
    """左胸 logo 区 mask. 白=inpaint, 黑=keep."""
    img = Image.open(base_path)
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    pixels = mask.load()
    cx, cy, r = int(w * 0.39), int(h * 0.36), 50
    for y in range(max(0, cy - r), min(h, cy + r)):
        for x in range(max(0, cx - r), min(w, cx + r)):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                pixels[x, y] = 255
    mask.save(out_path)


def build_pants_stripe_mask(base_path, out_path):
    """裤区外侧灰线条纹 mask. 立绘 512x912:
    外侧裤腿 (image 左右 5-15%) 从 cy=580 到 cy=850.
    """
    img = Image.open(base_path)
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    pixels = mask.load()
    # 左裤腿外侧
    for y in range(int(h * 0.55), int(h * 0.95)):
        for x in range(int(w * 0.20), int(w * 0.32)):
            pixels[x, y] = 255
    # 右裤腿外侧
    for y in range(int(h * 0.55), int(h * 0.95)):
        for x in range(int(w * 0.68), int(w * 0.80)):
            pixels[x, y] = 255
    mask.save(out_path)


def build_inpaint_workflow(base_name, mask_name, out_name, prompt, denoise=0.7):
    """Standard ComfyUI inpaint workflow.

    denoise=0.7: 只在 mask 区作 inpaint,其它保留基底 (避免全身重画).
    """
    return {
        "ckpt": {"class_type": "CheckpointLoaderSimple",
                 "inputs": {"ckpt_name": "DreamShaper_8_pruned.safetensors"}},
        "base": {"class_type": "LoadImage", "inputs": {"image": base_name}},
        "mask": {"class_type": "LoadImageMask", "inputs": {"image": mask_name,
                                                            "channel": "red"}},
        "vae": {"class_type": "VAEEncodeForInpaint",
                "inputs": {"pixels": ["base", 0], "vae": ["ckpt", 2],
                           "mask": ["mask", 0], "grow_mask_by": 4}},
        "neg_inpaint": {"class_type": "SetLatentNoiseMask",
                        "inputs": {"samples": ["vae", 0], "mask": ["mask", 0]}},
        "pos": {"class_type": "CLIPTextEncode",
                "inputs": {"clip": ["ckpt", 1], "text": prompt}},
        "neg": {"class_type": "CLIPTextEncode",
                "inputs": {"clip": ["ckpt", 1],
                           "text": "blurry, lowres, jpeg artifacts, text, watermark, cartoon, anime, painting, deformed hands"}},
        "ksampler": {"class_type": "KSampler",
                     "inputs": {"model": ["ckpt", 0],
                                "positive": ["pos", 0],
                                "negative": ["neg", 0],
                                "latent_image": ["neg_inpaint", 0],
                                "seed": 42, "steps": 28, "cfg": 7.5,
                                "sampler_name": "dpmpp_2m", "scheduler": "karras",
                                "denoise": denoise}},
        "decode": {"class_type": "VAEDecode",
                   "inputs": {"samples": ["ksampler", 0], "vae": ["ckpt", 2]}},
        "save": {"class_type": "SaveImage",
                 "inputs": {"images": ["decode", 0], "filename_prefix": out_name}},
    }


def submit(client, wf):
    payload = {"prompt": wf, "client_id": "inpaint-stage-b"}
    import urllib.request, urllib.error, json
    req = urllib.request.Request(
        f"{SERVER}/prompt",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        r = urllib.request.urlopen(req, timeout=30)
        resp = json.loads(r.read().decode())
        return resp.get("prompt_id")
    except urllib.error.HTTPError as e:
        print(f"[FAIL] {e.code} body={e.read().decode()[:200]}")
        return None


def wait_for_completion(prompt_id, timeout_sec=300):
    """poll /history/{id} until status != pending."""
    import urllib.request, json, time
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            r = urllib.request.urlopen(f"{SERVER}/history/{prompt_id}", timeout=10)
            h = json.loads(r.read().decode())
            entry = list(h.values())[0]
            status = entry.get("status", {})
            if status.get("completed"):
                return entry
        except Exception:
            pass
        time.sleep(3)
    return None


def fetch_image(entry, save_dir):
    """下载 entry 的 save 节点图片到 save_dir,返回本地路径."""
    import urllib.request, os
    save_dir = save_dir
    outs = entry.get("outputs", {})
    saved_paths = []
    for k, v in outs.items():
        for img in v.get("images", []):
            fn = img.get("filename")
            sub = img.get("subfolder", "")
            url = f"{SERVER}/view?filename={fn}&subfolder={sub}&type=output"
            local = os.path.join(save_dir, fn)
            urllib.request.urlretrieve(url, local)
            saved_paths.append(local)
    return saved_paths


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(LINXIA_BASE):
        print(f"[err] missing base: {LINXIA_BASE}")
        return 1

    # 1) logo mask + inpaint
    build_logo_mask(LINXIA_BASE, LOG)
    build_pants_stripe_mask(LINXIA_BASE, os.path.join(os.path.dirname(LOG), "inpaint_pants_mask.png"))
    PLOG = "inpaint_pants_pilot.png"

    c = ComfyUIClient(SERVER)
    base_name = c.upload_image(LINXIA_BASE)
    logo_mask_name = c.upload_image(LOG)
    pants_mask_name = c.upload_image(os.path.join(os.path.dirname(LOG), "inpaint_pants_mask.png"))
    print(f"[uploaded] base={base_name}, logo_mask={logo_mask_name}, pants_mask={pants_mask_name}")

    logo_wf = build_inpaint_workflow(
        base_name=base_name, mask_name=logo_mask_name,
        out_name="inpaint_logo_probe2",
        prompt="a photo of misty-blue athletic top with a small WHITE ARC LOGO on the left chest, the logo is a small curved white arc shape, photorealistic, sharp",
        denoise=0.6,
    )
    pid_l = submit(c, logo_wf)
    print(f"[logo inpaint submitted] {pid_l}")

    pants_wf = build_inpaint_workflow(
        base_name=base_name, mask_name=pants_mask_name,
        out_name="inpaint_pants_probe2",
        prompt="a photo of BLACK athletic pants with thin GRAY STRIPES on the outside of the legs running down the full length of the pants, photorealistic, sharp",
        denoise=0.5,
    )
    pid_p = submit(c, pants_wf)
    print(f"[pants inpaint submitted] {pid_p}")

    # 等待二者完成
    for pid, label in [(pid_l, "logo"), (pid_p, "pants")]:
        if not pid:
            continue
        print(f"[wait {label}] {pid}")
        entry = wait_for_completion(pid, timeout_sec=120)
        if entry:
            paths = fetch_image(entry, OUT_DIR)
            print(f"[{label} done] {paths}")
        else:
            print(f"[{label} timeout]")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)


if __name__ == "__main__":
    sys.exit(main() or 0)
