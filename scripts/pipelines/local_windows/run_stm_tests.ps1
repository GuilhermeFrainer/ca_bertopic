Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "Starting STM Test Experiments for all datasets" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan

# No Yelp for now
#$experiments = @("anes/anes_stm_test", "fed/fed_stm_test", "gadarian/gadarian_stm_test", "trump/trump_stm_test", "yelp/yelp_stm_test")
$experiments = @("anes/anes_stm_test", "fed/fed_stm_test", "gadarian/gadarian_stm_test", "trump/trump_stm_test")
$total = $experiments.Count
$current = 1

foreach ($exp in $experiments) {
    Write-Host ""
    Write-Host "[$current/$total] Running $exp..." -ForegroundColor Yellow
    uv run python scripts/experiments/run_stm.py --exp $exp
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: $exp failed with exit code $LASTEXITCODE" -ForegroundColor Red
    }
    $current++
}

Write-Host ""
Write-Host "====================================================" -ForegroundColor Green
Write-Host "All STM test experiments completed." -ForegroundColor Green
Write-Host "====================================================" -ForegroundColor Green
