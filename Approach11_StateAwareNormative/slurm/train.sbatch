#!/bin/bash
# Approach 11 on the SPIT cluster (one GPU). Run from the Approach11 folder:
#   sbatch slurm/train.sbatch
#   sbatch --export=ALL,RUN=no_events,SET="segments.use_events=false" slurm/train.sbatch
# Resubmitting resumes from last.pt and skips finished seeds. The store/features steps are
# skipped when their outputs are already present and up to date.
#SBATCH --job-name=a11_sanf
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load python/3.11.14
source ../.venv/bin/activate
pip install -q -r requirements.txt
export PYTHONUNBUFFERED=1
RUN="${RUN:-sanf_net}"
SET="${SET:-}"
python scripts/01_build_store.py --set data.workers=12
[ -f artifacts/features/biomarkers.csv ] || python scripts/02_build_features.py --set data.workers=12
[ -f artifacts/features/tabular_probs.npz ] || python scripts/03_tabular_baseline.py
python scripts/00_verify.py
python scripts/04_train.py --name "$RUN" --set train.num_workers=8 $SET
python scripts/05_evaluate.py --name "$RUN"
