# Define paths and command
$targetDir = "C:\Users\guisf\Downloads"
$remotePath = "gdsfrainer@gppd-hpc.inf.ufrgs.br:~/slurm"
$sshKey = "~/.ssh/pcad_ufrgs"

# Ensure you are in the local destination folder
Set-Location -Path $targetDir

# Execute rsync directly inside WSL
wsl rsync -avP --exclude='models' -e "ssh -i $sshKey" $remotePath .
