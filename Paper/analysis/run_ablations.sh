#!/bin/bash
# Paper ablations: each is a full 5-fold CV run of SANF-Net with one component removed.
# Finished folds are skipped (04_train resumes), so this can be re-run after an interruption.
# After each run, export_oof.py writes its out-of-fold + test predictions; metrics.py runs at the end.
HERE="$(cd "$(dirname "$0")" && pwd)"
A11="$HERE/../../Approach11_StateAwareNormative"
run() {  # name, overrides...
  local name=$1; shift
  echo "=== $(date +%T) $name : $*"
  (cd "$A11" && python scripts/04_train.py --cv 5 --name "$name" --set "$@") || { echo "!!! $name train failed"; return; }
  (cd "$HERE" && python export_oof.py "$name") || echo "!!! $name export failed"
}
run abl_no_events_cv5    segments.use_events=false
run abl_no_normative_cv5 features.normative=false
run abl_no_bio_cv5       model.use_biomarkers=false
run abl_no_age_cv5       model.use_age=false
run abl_eeg_only_cv5     model.use_biomarkers=false model.use_age=false
run abl_no_ordinal_cv5   model.ordinal=false
run abl_no_film_cv5      model.use_state_film=false
run abl_mean_pool_cv5    model.pool=mean
(cd "$HERE" && python metrics.py > results/metrics.out 2>&1) || echo "!!! metrics failed"
echo "=== $(date +%T) ALL DONE"
