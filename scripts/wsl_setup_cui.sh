#!/bin/bash
# WSL Linux venv for ComfyUI + Wan2.2-5B (躲 Windows torch .to() access violation)。
# 复刻 Windows env (torch 2.7.1+cu126) 唯一变量换 OS → 若 block_swap 在 WSL 通, 证实是 Windows bug。
# venv 放 WSL 原生 fs (~/cui_wsl, 快), ComfyUI 仓库/权重复用 /mnt/f (慢加载但可用)。
set -e
CUI=/mnt/f/wkspace/ComfyUI
VENV=~/cui_wsl

if [ ! -d "$VENV" ]; then
  echo "=== creating venv $VENV (python3.12) ==="
  python3.12 -m venv "$VENV"
fi
source "$VENV/bin/activate"

# 覆盖全局 tuna 镜像(WSL 下 flaky: SSL EOF/timeout), 走 aliyun(实测 200 稳)。
# 不改 ~/.config/pip/pip.conf(用户别的用途可能靠 tuna), 仅本 shell 生效。
# torch 仍走 pytorch.org cu126 索引(aliyun 不带 cu126 变体), 见下。
export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
export PIP_TRUSTED_HOST=mirrors.aliyun.com

pip install --upgrade pip wheel setuptools -q

echo "=== installing torch 2.7.1+cu126 (Linux) ==="
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu126

echo "=== torch CUDA check ==="
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| avail', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

echo "=== installing ComfyUI requirements ==="
pip install -r "$CUI/requirements.txt"

echo "=== installing WanVideoWrapper requirements ==="
pip install -r "$CUI/custom_nodes/ComfyUI-WanVideoWrapper/requirements.txt"

echo "=== pin numpy<2 + safetensors 0.4.3 (memory mandate) ==="
pip install "numpy<2" "safetensors==0.4.3"

echo "=== DONE setup ==="
python -c "import torch, numpy, safetensors; print('torch', torch.__version__, 'numpy', numpy.__version__, 'safetensors', safetensors.__version__, 'cuda', torch.cuda.is_available())"
