#!/bin/bash
# One-time environment setup on the SPIT login node (no GPU needed).
#   bash slurm/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."

module load python/3.11.14
[ -d .venv ] || uv venv .venv
source .venv/bin/activate

# Default Linux torch wheel bundles CUDA 12.x runtime -- fine for the RTX 6000 Ada.
uv pip install torch
uv pip install -r Approach1_EEGNet/requirements.txt   # identical in all three folders

python -c "import torch, numpy, pyedflib, yaml, sklearn; print('torch', torch.__version__, 'ok')"
