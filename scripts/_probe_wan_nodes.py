"""Temp probe: print WanVideoWrapper node input schemas from ComfyUI /object_info.
Delete after build_wan_i2v_workflow is locked."""
import json
import urllib.request

OI = json.load(urllib.request.urlopen("http://127.0.0.1:8188/object_info", timeout=30))

WANT = [
    "WanVideoModelLoader",
    "WanVideoBlockSwap",
    "WanVideoVAELoader",
    "LoadWanVideoT5TextEncoder",
    "WanVideoTextEncode",
    "WanVideoEncode",
    "WanVideoEmptyEmbeds",
    "WanVideoSampler",
    "WanVideoDecode",
    "VHS_VideoCombine",
]

all_wan = sorted(k for k in OI if "wanvideo" in k.lower() or "wan" in k.lower())[:60]
print("=== nodes containing 'wan' (first 60) ===")
print("\n".join(all_wan))

for n in WANT:
    if n not in OI:
        print(f"\n### {n}: *** NOT FOUND ***")
        continue
    inp = OI[n]["input"]
    print(f"\n### {n}")
    for section in ("required", "optional"):
        d = inp.get(section, {})
        if d:
            print(f"  [{section}]")
            for k, v in d.items():
                print(f"    {k} = {v}")
