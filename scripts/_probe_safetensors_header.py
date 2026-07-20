"""Read safetensors HEADER only (pure python, no get_tensor, will NOT segfault).

Phase-1 evidence: dtype/shape/data_offsets of the crashing tensor
`blocks.0.cross_attn.k.bias` (5B file) vs working T5 tensors.

safetensors layout: [u64 LE header_len][header_len bytes JSON][tensor data...]
"""
import json
import struct
import os

FILES = {
    "5B": "../ComfyUI/models/diffusion_models/Wan2_2-TI2V-5B_fp8_e4m3fn_scaled_KJ.safetensors",
    "T5": "../ComfyUI/models/text_encoders/umt5-xxl-enc-fp8_e4m3fn.safetensors",
    "VAE": "../ComfyUI/models/vae/Wan2_2_VAE_bf16.safetensors",
}

for label, path in FILES.items():
    if not os.path.isfile(path):
        print(f"[{label}] MISSING"); continue
    sz = os.path.getsize(path)
    print(f"\n[{label}] {os.path.basename(path)}  size={sz/1e9:.2f}GB")
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n).decode("utf-8"))
    total = len(header) - (1 if "__metadata__" in header else 0)
    print(f"    header_len={n}  tensor_count={total}")
    # dtypes histogram
    from collections import Counter
    dt = Counter(v.get("dtype") for k, v in header.items() if k != "__metadata__")
    print(f"    dtypes={dict(dt)}")
    # first 6 tensors by file order (sort by data_offsets start)
    items = sorted(
        ((k, v) for k, v in header.items() if k != "__metadata__"),
        key=lambda kv: kv[1]["data_offsets"][0],
    )
    for k, v in items[:6]:
        s, e = v["data_offsets"]
        print(f"    {k:48s} dtype={v['dtype']:18s} shape={str(v['shape']):20s} "
              f"bytes={e-s:>10d} (abs {s:>13d}..{e:>13d})")
    # specifically the crashing key
    ck = "blocks.0.cross_attn.k.bias"
    if ck in header:
        v = header[ck]
        s, e = v["data_offsets"]
        print(f"    >>CRASH KEY {ck}: dtype={v['dtype']} shape={v['shape']} bytes={e-s} abs[{s}..{e}]")
        print(f"       tensor byte-size as fraction of file: {(e-s)/sz*100:.4f}%")
print("\n[DONE header parse]")
