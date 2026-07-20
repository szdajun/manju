"""Probe WanVideoWrapper T5/TextEncode node schemas from /object_info (temp)."""
import json
import urllib.request

SERVER = "http://127.0.0.1:8188"
NODES = ["LoadWanVideoT5TextEncoder", "WanVideoTextEncode",
         "WanVideoModelLoader", "WanVideoBlockSwap"]


def main():
    try:
        with urllib.request.urlopen(SERVER + "/object_info", timeout=10) as r:
            info = json.loads(r.read().decode())
    except Exception as e:
        print(f"[DEAD] /object_info failed: {e}")
        return
    print("[ALIVE] server responds\n")
    for n in NODES:
        spec = info.get(n)
        if not spec:
            print(f"### {n}: NOT FOUND\n")
            continue
        inp = spec.get("input", {}).get("required", {})
        print(f"### {n}")
        for k, v in inp.items():
            print(f"  {k}: {v}")
        print()


if __name__ == "__main__":
    main()
