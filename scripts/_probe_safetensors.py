"""Reproduce load_torch_file crash in isolation (temp diagnostic).

ComfyUI 0.22 crashes at comfy/utils.py:135 f.get_tensor(k) when loading
umt5-xxl-enc-fp8_e4m3fn on torch 2.7.1+cu126. Test if safetensors get_tensor
on the real files crashes outside ComfyUI.
"""
import gc
import traceback
import torch
import safetensors

print(f"torch={torch.__version__} safetensors={safetensors.__version__} numpy={__import__('numpy').__version__}")

FILES = [
    ("T5(umt5 fp8)", "../ComfyUI/models/text_encoders/umt5-xxl-enc-fp8_e4m3fn.safetensors"),
    ("5B diff(fp8_scaled)", "../ComfyUI/models/diffusion_models/Wan2_2-TI2V-5B_fp8_e4m3fn_scaled_KJ.safetensors"),
    ("VAE(bf16)", "../ComfyUI/models/vae/Wan2_2_VAE_bf16.safetensors"),
]

for label, path in FILES:
    print(f"\n[{label}] {path.split('/')[-1]}")
    try:
        import os
        if not os.path.isfile(path):
            print("    FILE MISSING"); continue
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            print(f"    {len(keys)} keys; first={keys[0][:48]}")
            for i, k in enumerate(keys[:3]):
                t = f.get_tensor(k)
                print(f"    get_tensor[{i}] {k[:40]} dtype={t.dtype} shape={tuple(t.shape)}")
                _ = t.flatten()[:3].tolist()   # touch storage (== __getitem__ path)
                print(f"      storage read OK")
        print(f"    {label} LOAD OK")
    except Exception as e:
        print(f"    {label} EXCEPTION: {repr(e)}"); traceback.print_exc()
    gc.collect()

print("\n[DONE]")
