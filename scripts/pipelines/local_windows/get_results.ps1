param (
    [switch]$Figures,
    [switch]$MergeInfo0,
    [switch]$OnlyInfo0,
    [switch]$Release,
    [ValidateSet("all", "standard", "stemmed", "no_stopword_removal", "with_stopwords", "no_stopword")]
    [string]$ResultType = "all"
)

# Normalize aliases to primary result type 'no_stopword_removal'
if ($ResultType -eq "no_stopword" -or $ResultType -eq "with_stopwords") {
    $ResultType = "no_stopword_removal"
}

# Set output directory
$releaseDir = "E:\30-39 Estudos\36 Mestrado\36.14 Dissertation"
if ($Release -and $releaseDir -ne "") {
    $outputDir = $releaseDir
} else {
    $outputDir = Join-Path $env:USERPROFILE "Downloads\CA-BERTopic-Results"
}

$tablesDir = Join-Path $outputDir "tables"
$figuresDir = Join-Path $outputDir "figures"

# Prepare flags
$mergeFlags = @()
if ($MergeInfo0) { $mergeFlags += "--merge-info0" }
if ($OnlyInfo0) { $mergeFlags += "--only-info0" }

# List of result types to process
$resultTypesToProcess = if ($ResultType -eq "all") { @("standard", "stemmed", "no_stopword_removal") } else { @($ResultType) }

# List of datasets
$datasets = @("fed", "yelp", "trump", "anes", "gadarian")

foreach ($resType in $resultTypesToProcess) {
    Write-Host "`n============================================================" -ForegroundColor Magenta
    Write-Host " PROCESSING RESULT TYPE: $($resType.ToUpper())" -ForegroundColor Magenta
    Write-Host "============================================================" -ForegroundColor Magenta

    $typeTablesDir = Join-Path $tablesDir $resType
    $typeFiguresDir = Join-Path $figuresDir $resType

    if (-not (Test-Path $typeTablesDir)) { New-Item -ItemType Directory -Path $typeTablesDir -Force | Out-Null }
    if (-not (Test-Path $typeFiguresDir)) { New-Item -ItemType Directory -Path $typeFiguresDir -Force | Out-Null }

    # 0. Generate Label Table
    Write-Host "Generating Model Label Table ($resType)..." -ForegroundColor Yellow
    $labelTablePath = Join-Path $typeTablesDir "model_labels.tex"
    uv run scripts/analysis/find_best_models.py --label-table --result-type $resType @mergeFlags | Out-File -FilePath "$labelTablePath" -Encoding utf8

    # 0b. Generate HDBSCAN Noise Coverage Table
    Write-Host "Generating HDBSCAN Noise Coverage Table ($resType)..." -ForegroundColor Yellow
    $noiseCoverageTablePath = Join-Path $typeTablesDir "hdbscan_noise_coverage.tex"
    $noiseFlags = @()
    if ($MergeInfo0) { $noiseFlags += "--merge-info0" }
    uv run scripts/analysis/calculate_noise_coverage.py --result-type $resType --output-latex "$noiseCoverageTablePath" @noiseFlags

    foreach ($dataset in $datasets) {
        Write-Host "`n------------------------------------------------------------" -ForegroundColor Green
        Write-Host " DATASET: $dataset | TYPE: $resType" -ForegroundColor Green
        Write-Host "------------------------------------------------------------" -ForegroundColor Green

        if ($Figures) {
            # 1. Cleveland Dot Plots
            Write-Host "Generating Cleveland Dot Plot (Best)..."
            $clevelandBestPath = Join-Path $typeFiguresDir "${dataset}_cleveland.pdf"
            uv run scripts/analysis/find_best_models.py --dataset $dataset --result-type $resType --cleveland "$clevelandBestPath" --suppress-nulls @mergeFlags

            Write-Host "Generating Cleveland Dot Plot (Average)..."
            $clevelandAvgPath = Join-Path $typeFiguresDir "${dataset}_cleveland_avg.pdf"
            uv run scripts/analysis/find_best_models.py --dataset $dataset --result-type $resType --cleveland "$clevelandAvgPath" --average --suppress-nulls @mergeFlags

            # 2. Parallel Lines Plot (Dump)
            Write-Host "Generating Parallel Lines Plot (Dump)..."
            $parallelPath = Join-Path $typeFiguresDir "${dataset}_parallel.pdf"
            uv run scripts/analysis/find_best_models.py --dataset $dataset --result-type $resType --parallel "$parallelPath" --dump --suppress-nulls @mergeFlags

            # 3. Star Plots
            Write-Host "Generating Star Plot (Best)..."
            $starBestPath = Join-Path $typeFiguresDir "${dataset}_star_best.pdf"
            uv run scripts/analysis/find_best_models.py --dataset $dataset --result-type $resType --star-plot "$starBestPath" --suppress-nulls @mergeFlags  

            Write-Host "Generating Star Plot (Average)..."
            $starAvgPath = Join-Path $typeFiguresDir "${dataset}_star_avg.pdf"
            uv run scripts/analysis/find_best_models.py --dataset $dataset --result-type $resType --star-plot "$starAvgPath" --average --suppress-nulls @mergeFlags
        }

        # 4. LaTeX Tables
        Write-Host "Generating LaTeX Table (Best)..."
        $tableBestPath = Join-Path $typeTablesDir "${dataset}_table_best.tex"
        uv run scripts/analysis/find_best_models.py --dataset $dataset --result-type $resType --latex "$tableBestPath" --suppress-nulls @mergeFlags     

        Write-Host "Generating LaTeX Table (Best - Full with PCA & K-Means)..."
        $tableBestFullPath = Join-Path $typeTablesDir "${dataset}_table_best_full.tex"
        uv run scripts/analysis/find_best_models.py --dataset $dataset --result-type $resType --latex "$tableBestFullPath" --suppress-nulls --exclude-clustering none --exclude-dim-red none @mergeFlags

        Write-Host "Generating LaTeX Table (Average)..."
        $tableAvgPath = Join-Path $typeTablesDir "${dataset}_table_avg.tex"
        uv run scripts/analysis/find_best_models.py --dataset $dataset --result-type $resType --latex "$tableAvgPath" --average --suppress-nulls @mergeFlags

        Write-Host "Generating LaTeX Table (Average - Full with PCA & K-Means)..."
        $tableAvgFullPath = Join-Path $typeTablesDir "${dataset}_table_avg_full.tex"
        uv run scripts/analysis/find_best_models.py --dataset $dataset --result-type $resType --latex "$tableAvgFullPath" --average --suppress-nulls --exclude-clustering none --exclude-dim-red none @mergeFlags

        # 5. LaTeX Table (Dump)
        Write-Host "Generating LaTeX Table (Dump)..."
        $tableDumpPath = Join-Path $typeTablesDir "${dataset}_table_dump.tex"
        uv run scripts/analysis/find_best_models.py --dataset $dataset --result-type $resType --latex "$tableDumpPath" --dump
    }
}

Write-Host "`nAll results generated successfully in $outputDir" -ForegroundColor Cyan
