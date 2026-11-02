#!/bin/bash
# Verify -> train (all seeds) -> evaluate, for one approach on one GPU.
#   sbatch --export=ALL,APPROACH=Approach6_STFTCNN slurm/train.sbatch
#   sbatch --export=ALL,APPROACH=Approach6_STFTCNN,CONFIG=config_noage.yaml slurm/train.sbatch
# Resubmitting the same command resumes from last.pt and skips finished seeds.
#SBATCH --job-name=caueeg_train
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm/logs/%x_%j.out

set -euo pipefail
: "${APPROACH:?set APPROACH, e.g. --export=ALL,APPROACH=Approach6_STFTCNN}"
CONFIG="${CONFIG:-config.yaml}"

cd "$SLURM_SUBMIT_DIR"
module load python/3.11.14
source .venv/bin/activate
export PYTHONUNBUFFERED=1          # live per-epoch lines in the log

echo "job $SLURM_JOB_ID on $(hostname): $APPROACH / $CONFIG"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

cd "$APPROACH"
python scripts/00_verify.py   --config "$CONFIG"
python scripts/02_train.py    --config "$CONFIG"
python scripts/03_evaluate.py --config "$CONFIG"
