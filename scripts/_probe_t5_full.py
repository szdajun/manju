"""Dump FULL input spec (required+optional) for LoadWanVideoT5TextEncoder (temp)."""
import json
import urllib.request

with urllib.request.urlopen("http://127.0.0.1:8188/object_info/LoadWanVideoT5TextEncoder",
                            timeout=10) as r:
    spec = json.loads(r.read().decode())["LoadWanVideoT5TextEncoder"]
for sect in ("required", "optional"):
    print(f"### {sect}")
    for k, v in (spec.get("input", {}).get(sect, {}) or {}).items():
        print(f"  {k}: {v}")
