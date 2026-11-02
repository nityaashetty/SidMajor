#!/bin/bash
# CPU-only: EDF -> fp16 memmap in _cache/eeg_fp16 (shared by all approaches).
# Skip this job if you uploaded _cache/ from your laptop.
#SBATCH --job-name=caueeg_cache
#SBATCH --partition=general
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load python/3.11.14
source .venv/bin/activate

cd Approach1_EEGNet
python scripts/01_build_cache.py
