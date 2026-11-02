# Train all 5 seeds (resumes from last.pt if interrupted), then evaluate both checkpoints.
# Usage:  powershell -ExecutionPolicy Bypass -File run_attempt5.ps1
Set-Location $PSScriptRoot
$Host.UI.RawUI.WindowTitle = "Attempt 5 - EEGNet regularised"

python -u scripts/02_train.py 2>&1 | Tee-Object -FilePath train_log.txt -Append
if ($LASTEXITCODE -ne 0) { Write-Host "`nTRAINING FAILED (exit $LASTEXITCODE)" -ForegroundColor Red; exit 1 }

Write-Host "`n===== EVALUATE final.pt (EMA, primary) =====" -ForegroundColor Cyan
python -u scripts/03_evaluate.py 2>&1 | Tee-Object -FilePath eval_final_log.txt
Write-Host "`n===== EVALUATE best.pt (best val epoch, Approach-1 rule) =====" -ForegroundColor Cyan
python -u scripts/03_evaluate.py --checkpoint best.pt 2>&1 | Tee-Object -FilePath eval_best_log.txt

Write-Host "`nALL DONE. Results: outputs/eegnet_attempt5/results.json and results_best.json" -ForegroundColor Green
