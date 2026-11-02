#!/bin/bash
# Submit the cache build (unless already built) then all three approaches.
# Two GPUs on the node -> two trainings run at once, the third queues.
#   bash slurm/submit_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p slurm/logs

DEP=""
if [ ! -f _cache/eeg_fp16/norm_stats.npz ]; then
    CID=$(sbatch --parsable slurm/build_cache.sbatch)
    echo "cache job $CID"
    DEP="--dependency=afterok:$CID"
fi

for A in Approach1_EEGNet Approach6_STFTCNN Approach4_TemporalMIL; do
    JID=$(sbatch --parsable $DEP --job-name="tr_${A%%_*}" \
          --export=ALL,APPROACH=$A slurm/train.sbatch)
    echo "$A -> job $JID"
done
echo "watch: squeue -u \$USER   |   tail -f slurm/logs/*.out"
