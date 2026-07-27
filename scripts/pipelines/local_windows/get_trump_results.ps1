# PowerShell script to generate all results specifically for the Trump dataset
# This bypasses any 'info0' filtering since those experiments may not be available for Trump yet.

<#
# OPTIONAL: Run these if you haven't generated the raw results for Trump yet:
uv run scripts/run_optimizer.py --exp trump_opt_mv_spectral
uv run scripts/run_optimizer.py --exp trump_opt_mv_co_reg_spectral
#>

# Set output directory
$outputDir = Join-Path $env:USERPROFILE "Downloads\Trump-Results"
$tablesDir = Join-Path $outputDir "tables"
$figuresDir = Join-Path $outputDir "figures"

# Create directories
if (-not (Test-Path $outputDir)) { New-Item -ItemType Directory -Path $outputDir -Force | Out-Null }
if (-not (Test-Path $tablesDir)) { New-Item -ItemType Directory -Path $tablesDir -Force | Out-Null }
if (-not (Test-Path $figuresDir)) { New-Item -ItemType Directory -Path $figuresDir -Force | Out-Null }

Write-Host "Generating Trump results to: $outputDir" -ForegroundColor Cyan

$dataset = "trump"

# 1. Cleveland Dot Plots
Write-Host "Generating Cleveland Dot Plot (Best)..."
$clevelandBestPath = Join-Path $figuresDir "${dataset}_cleveland.pdf"
uv run scripts/find_best_models.py --dataset $dataset --cleveland "$clevelandBestPath" --suppress-nulls

Write-Host "Generating Cleveland Dot Plot (Average)..."
$clevelandAvgPath = Join-Path $figuresDir "${dataset}_cleveland_avg.pdf"
uv run scripts/find_best_models.py --dataset $dataset --cleveland "$clevelandAvgPath" --average --suppress-nulls

# 2. Parallel Lines Plot (Dump)
Write-Host "Generating Parallel Lines Plot (Dump)..."
$parallelPath = Join-Path $figuresDir "${dataset}_parallel.pdf"
uv run scripts/find_best_models.py --dataset $dataset --parallel "$parallelPath" --dump --suppress-nulls

# 3. Star Plots
Write-Host "Generating Star Plot (Best)..."
$starBestPath = Join-Path $figuresDir "${dataset}_star_best.pdf"
uv run scripts/find_best_models.py --dataset $dataset --star-plot "$starBestPath" --suppress-nulls

Write-Host "Generating Star Plot (Average)..."
$starAvgPath = Join-Path $figuresDir "${dataset}_star_avg.pdf"
uv run scripts/find_best_models.py --dataset $dataset --star-plot "$starAvgPath" --average --suppress-nulls

# 4. LaTeX Tables
Write-Host "Generating LaTeX Table (Best)..."
$tableBestPath = Join-Path $tablesDir "${dataset}_table_best.tex"
uv run scripts/find_best_models.py --dataset $dataset --latex "$tableBestPath" --suppress-nulls

Write-Host "Generating LaTeX Table (Average)..."
$tableAvgPath = Join-Path $tablesDir "${dataset}_table_avg.tex"
uv run scripts/find_best_models.py --dataset $dataset --latex "$tableAvgPath" --average --suppress-nulls

# 5. LaTeX Table (Dump)
Write-Host "Generating LaTeX Table (Dump)..."
$tableDumpPath = Join-Path $tablesDir "${dataset}_table_dump.tex"
uv run scripts/find_best_models.py --dataset $dataset --latex "$tableDumpPath" --dump

Write-Host "`nTrump results generated successfully in $outputDir" -ForegroundColor Cyan
