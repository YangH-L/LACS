#!/usr/bin/env bash
# Environment setup on the GPU server (run from the project root).
set -euo pipefail

ENV_NAME=${ENV_NAME:-lacs}
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" python=3.11
  fi
  conda activate "$ENV_NAME"
fi

pip install -r requirements.txt
# PyTorch with a CUDA 12.8+ build (needed for RTX 5090 / Blackwell)
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-gpu.txt

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
python -m lacs build-cpp
echo "environment ready"
