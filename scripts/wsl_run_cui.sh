#!/bin/bash
# WSL 跑 ComfyUI(复用 /mnt/f/wkspace/ComfyUI 仓库+权重), port 8189(避 Windows 8188 冲突)。
# Windows 端 gen_wan.py 走 localhost:8189 连(WSL2 NAT 默认 localhost 转发)。
# 模型在 /mnt/f(Windows fs 经 9p, 加载慢但布局同 Windows ComfyUI, Wan 权重直接可用)。
set -e
CUI=/mnt/f/wkspace/ComfyUI
PY=~/cui_wsl/bin/python

cd "$CUI"
exec "$PY" main.py --listen 127.0.0.1 --port 8189
