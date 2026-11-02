# Live status of the paper ablation runs. Read-only; close the window any time (training is unaffected).
#   powershell -NoExit -File watch_progress.ps1
$log = Join-Path $PSScriptRoot "ablations.log"
$runs = Join-Path $PSScriptRoot "..\..\Approach11_StateAwareNormative\artifacts\runs"
$names = "abl_no_events_cv5", "abl_no_normative_cv5", "abl_no_bio_cv5", "abl_no_age_cv5",
         "abl_eeg_only_cv5", "abl_no_ordinal_cv5", "abl_no_film_cv5", "abl_mean_pool_cv5"
$Host.UI.RawUI.WindowTitle = "SANF-Net ablations - live status"

while ($true) {
    $out = New-Object System.Collections.Generic.List[string]
    $out.Add("SANF-Net ablations   $(Get-Date -Format 'HH:mm:ss')   (refreshes every 5 s, Ctrl+C to close)")
    $out.Add("")
    $gpu = nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits
    if ($gpu) {
        $g = $gpu.Split(",") | ForEach-Object { $_.Trim() }
        $bar = "#" * [int]([int]$g[1] / 5) + "." * (20 - [int]([int]$g[1] / 5))
        $out.Add("GPU  $($g[0])")
        $out.Add("     util [$bar] $($g[1])%   mem $($g[2]) / $($g[3]) MiB   $($g[4]) C   $($g[5]) W")
    }
    $out.Add("")
    $out.Add("Runs (folds finished / 5):")
    $done = 0
    foreach ($n in $names) {
        $k = 0
        foreach ($f in 0..4) { if (Test-Path (Join-Path $runs "$n\fold$f\final.pt")) { $k++ } }
        $exported = Test-Path (Join-Path $PSScriptRoot "oof\$n.csv")
        if ($k -eq 5) { $done++ }
        $mark = if ($exported) { "done" } elseif ($k -gt 0) { "training" } else { "queued" }
        $out.Add(("  {0,-22} [{1}{2}] {3}/5  {4}" -f $n, ("#" * $k), ("." * (5 - $k)), $k, $mark))
    }
    $left = (8 - $done) * 30
    $out.Add("")
    $out.Add("Rough time left: ~$([math]::Floor($left / 60)) h $($left % 60) min (about 30 min per run)")
    $out.Add("")
    $out.Add("Latest log lines:")
    if (Test-Path $log) {
        Get-Content $log -Tail 6 | ForEach-Object {
            $line = if ($_.Length -gt 118) { $_.Substring(0, 118) } else { $_ }
            $out.Add("  $line")
        }
        if (Select-String -Path $log -Pattern "ALL DONE" -Quiet) { $out.Add(""); $out.Add("  >>> ALL DONE <<<") }
    }
    Clear-Host
    $out | ForEach-Object { Write-Host $_ }
    Start-Sleep -Seconds 5
}
