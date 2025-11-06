#!/usr/bin/env bash
set -Eeuo pipefail

wget -qO- https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env


sudo apt update && sudo apt install -y build-essential
sudo apt install python3.12-venv

# Install system build/env tools (Ubuntu/Debian):
sudo apt update && sudo apt install -y build-essential

# Create and activate virtual environment
uv venv --python 3.12 && source .venv/bin/activate \
        && uv pip install -r requirements/requirements.txt \
        && uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126 \
        && uv pip install torch-geometric==2.6.1 \
        && uv pip install hf_transfer\
        && uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu126.html

echo "Installation complete."