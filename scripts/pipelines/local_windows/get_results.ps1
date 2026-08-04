param (
    [switch]$MergeInfo0,
    [switch]$OnlyInfo0,
    [switch]$Release
)

# Set output directory
$releaseDir = "E:\30-39 Estudos\36 Mestrado\36.14 Dissertation"
if ($Release -and $releaseDir -ne "") {
    $outputDir = $releaseDir
} else {
    $outputDir = Join-Path $env:USERPROFILE "Downloads\CA-BERTopic-Results"
}

$tablesDir = Join-Path $outputDir "tables"
$figuresDir = Join-Path $outputDir "figures"

# Create directories
if (-not (Test-Path $outputDir)) { New-Item -ItemType Directory -Path $outputDir -Force | Out-Null }
if (-not (Test-Path $tablesDir)) { New-Item -ItemType Directory -Path $tablesDir -Force | Out-Null }
if (-not (Test-Path $figuresDir)) { New-Item -ItemType Directory -Path $figuresDir -Force | Out-Null }

Write-Host "Generating results to: $outputDir" -ForegroundColor Cyan

# Prepare flags
$mergeFlags = @()
if ($MergeInfo0) { $mergeFlags += "--merge-info0" }
if ($OnlyInfo0) { $mergeFlags += "--only-info0" }

# 0. Generate Label Table
Write-Host "Generating Model Label Table..." -ForegroundColor Yellow
$labelTablePath = Join-Path $tablesDir "model_labels.tex"
uv run scripts/analysis/find_best_models.py --label-table @mergeFlags | Out-File -FilePath "$labelTablePath" -Encoding utf8

# List of datasets
$datasets = @("fed", "yelp", "trump", "anes", "gadarian")

foreach ($dataset in $datasets) {
    Write-Host "`n============================================================" -ForegroundColor Green
    Write-Host " PROCESSING DATASET: $dataset" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green

    # 1. Cleveland Dot Plots
    Write-Host "Generating Cleveland Dot Plot (Best)..."
    $clevelandBestPath = Join-Path $figuresDir "${dataset}_cleveland.pdf"
    uv run scripts/analysis/find_best_models.py --dataset $dataset --cleveland "$clevelandBestPath" --suppress-nulls @mergeFlags

    Write-Host "Generating Cleveland Dot Plot (Average)..."
    $clevelandAvgPath = Join-Path $figuresDir "${dataset}_cleveland_avg.pdf"
    uv run scripts/analysis/find_best_models.py --dataset $dataset --cleveland "$clevelandAvgPath" --average --suppress-nulls @mergeFlags

    # 2. Parallel Lines Plot (Dump)
    Write-Host "Generating Parallel Lines Plot (Dump)..."
    $parallelPath = Join-Path $figuresDir "${dataset}_parallel.pdf"
    uv run scripts/analysis/find_best_models.py --dataset $dataset --parallel "$parallelPath" --dump --suppress-nulls @mergeFlags

    # 3. Star Plots
    Write-Host "Generating Star Plot (Best)..."
    $starBestPath = Join-Path $figuresDir "${dataset}_star_best.pdf"
    uv run scripts/analysis/find_best_models.py --dataset $dataset --star-plot "$starBestPath" --suppress-nulls @mergeFlags  

    Write-Host "Generating Star Plot (Average)..."
    $starAvgPath = Join-Path $figuresDir "${dataset}_star_avg.pdf"
    uv run scripts/analysis/find_best_models.py --dataset $dataset --star-plot "$starAvgPath" --average --suppress-nulls @mergeFlags

    # 4. LaTeX Tables
    Write-Host "Generating LaTeX Table (Best)..."
    $tableBestPath = Join-Path $tablesDir "${dataset}_table_best.tex"
    uv run scripts/analysis/find_best_models.py --dataset $dataset --latex "$tableBestPath" --suppress-nulls @mergeFlags     

    Write-Host "Generating LaTeX Table (Average)..."
    $tableAvgPath = Join-Path $tablesDir "${dataset}_table_avg.tex"
    uv run scripts/analysis/find_best_models.py --dataset $dataset --latex "$tableAvgPath" --average --suppress-nulls @mergeFlags

    # 5. LaTeX Table (Dump)
    Write-Host "Generating LaTeX Table (Dump)..."
    $tableDumpPath = Join-Path $tablesDir "${dataset}_table_dump.tex"
    uv run scripts/analysis/find_best_models.py --dataset $dataset --latex "$tableDumpPath" --dump
}

Write-Host "`nAll results generated successfully in $outputDir" -ForegroundColor Cyan
