#!/bin/bash
# Second GPU queue (review fixes). Starts only after run_ablations.sh has finished, so the two never
# share the GPU. Every step is resumable / skips finished work; safe to re-run.
cd "$(dirname "$0")" || exit 1
until grep -q "ALL DONE" ablations.log; do sleep 60; done
step() { echo "=== $(date +%T) $*"; "$@" || echo "!!! failed: $*"; }
step python train_cv.py --name base_eegnet_cv5   --model eegnet   --set segments.use_events=false
step python train_cv.py --name base_resnet1d_cv5 --model resnet1d --set segments.use_events=false
step python external_cv.py sanf_net_cv5
for r in 1 2 3 4; do
  step python train_cv.py --name sanf_net_cv5_r$r --model sanf --cv-seed $r
done
step python metrics.py
echo "=== $(date +%T) QUEUE2 DONE"
