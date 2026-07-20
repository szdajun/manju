"""Phase-1: isolate WHICH dtype/shape crashes get_tensor on this torch build.

Synthetic safetensors round-trip. If F32 [3072] crashes here too, dtype is the
root cause. If only the real 5B file crashes, it's file-specific (mmap/flaky).
"""
import gc
import traceback
import torch
from safetensors import safe_open
from safetensors.torch import save_file

print(f"torch={torch.__version__} safetensors OK")
import tempfile, os
tmp = tempfile.mkdtemp()

CASES = [
    ("F32_1d_3072",  {"b": torch.zeros(3072, dtype=torch.float32)}),        # == 5B crash tensor
    ("BF16_1d_4096", {"b": torch.zeros(4096, dtype=torch.bfloat16)}),       # == T5 working tensor
    ("F8E4M3_2d",    {"w": torch.zeros((3072, 3072), dtype=torch.float8_e4m3fn)}),  # 5B weight dtype
    ("F32_2d_big",   {"w": torch.zeros((3072, 3072), dtype=torch.float32)}),
]
for name, tensors in CASES:
    p = os.path.join(tmp, f"{name}.safetensors")
    save_file(tensors, p)
    try:
        with safe_open(p, framework="pt", device="cpu") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                _ = t.flatten()[:3].tolist()
        print(f"  [{name}] OK")
    except Exception as e:
        print(f"  [{name}] EXC: {repr(e)}"); traceback.print_exc()
    gc.collect()
print("\n[SYNTHETIC DONE]")
