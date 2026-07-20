"""Isolate torch module.to(device) behavior (temp diagnostic, v2: cache-cleaned).

v2: empty_cache between tests, smaller fp8 allocs so OOM doesn't mask the result.
Goal after torch 2.13->2.7.1 downgrade: confirm bf16 AND fp8 module.to(cpu) are clean.
"""
import gc
import sys
import traceback
import torch

print(f"torch={torch.__version__} cuda={torch.version.cuda} avail={torch.cuda.is_available()}")
dev = torch.device("cuda")


def clean():
    gc.collect()
    torch.cuda.empty_cache()


print("\n[1] bf16 large module .cuda() then .to('cpu') ...")
try:
    m = torch.nn.Sequential(*[torch.nn.Linear(2048, 2048, bias=False) for _ in range(400)])
    m.to(dev); m.to("cpu")
    print("    bf16 module .to(cpu) OK"); del m; clean()
except Exception as e:
    print("    bf16 EXCEPTION:", repr(e)); traceback.print_exc(); clean()


print("\n[2] fp8 tensor on cuda .to('cpu') ...")
try:
    x = torch.randn(4096, 4096, device=dev).to(torch.float8_e4m3fn)
    y = x.to("cpu")
    print(f"    fp8 tensor .to(cpu) OK -> {y.dtype} {y.device}"); del x, y; clean()
except Exception as e:
    print("    fp8 tensor EXCEPTION:", repr(e)); traceback.print_exc(); clean()


print("\n[3] fp8-weight linear module on cuda .to('cpu') (== force_offload path) ...")
try:
    lin = torch.nn.Linear(2048, 2048, bias=False, device=dev)
    lin.weight.data = lin.weight.data.to(torch.float8_e4m3fn)
    lin.to("cpu")
    print(f"    fp8-weight module .to(cpu) OK -> {lin.weight.dtype} {lin.weight.device}"); del lin; clean()
except Exception as e:
    print("    fp8 module EXCEPTION:", repr(e)); traceback.print_exc(); clean()

print("\n[ALL DONE] no access violation")
