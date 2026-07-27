param (
    [Parameter(Mandatory=$true)]
    [string]$Version
)

$ErrorActionPreference = "Stop"

# Docker Image Variables
$ImageName = "cast"
$TagVersion = "${ImageName}:stm-lite-$Version"
$TagLatest = "${ImageName}:stm-lite-latest"

# File System Variables
$DownloadsDir = Join-Path $env:USERPROFILE "Downloads"
$TarName = "${ImageName}_stm-lite-v${Version}.tar"
$TarPath = Join-Path $DownloadsDir $TarName

# Rsync and Remote Variables
$RemoteUser = "gdsfrainer"
$RemoteHost = "gppd-hpc.inf.ufrgs.br"
$RemotePath = "~/docker_images"
$SshKeyPath = "~/.ssh/pcad_ufrgs" # Path inside WSL as per original command

Write-Host "`n>>> Building Docker image: $TagVersion" -ForegroundColor Cyan
docker build -f Dockerfile.stm -t $TagVersion .

Write-Host "`n>>> Tagging image as $TagLatest" -ForegroundColor Cyan
docker tag $TagVersion $TagLatest

Write-Host "`n>>> Saving image to $TarPath" -ForegroundColor Cyan
if (!(Test-Path $DownloadsDir)) {
    New-Item -ItemType Directory -Path $DownloadsDir | Out-Null
}
docker save $TagVersion -o $TarPath

Write-Host "`n>>> Converting path for WSL using wslpath" -ForegroundColor Cyan
# Replace backslashes with forward slashes to prevent stripping when passing to WSL
$NormalizedPath = $TarPath.Replace('\', '/')
$WslTarPath = wsl wslpath -u $NormalizedPath

if ([string]::IsNullOrWhiteSpace($WslTarPath)) {
    Write-Error "Failed to convert Windows path to WSL path. wslpath returned no output."
    exit 1
}
$WslTarPath = $WslTarPath.Trim()

Write-Host "`n>>> Transferring image to remote server via wsl rsync" -ForegroundColor Cyan
$RemoteDest = "${RemoteUser}@${RemoteHost}:$RemotePath"
$SshCommand = "ssh -i $SshKeyPath"

# Run rsync through WSL
wsl rsync -azvP -e "$SshCommand" $WslTarPath $RemoteDest

Write-Host "`nProcess completed successfully!" -ForegroundColor Green
