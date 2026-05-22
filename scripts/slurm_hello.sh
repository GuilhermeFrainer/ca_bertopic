#!/bin/bash
#SBATCH --job-name=hello_r_docker
#SBATCH --partition=cidia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=slurm_log/%x_%j.out
#SBATCH --error=slurm_log/%x_%j.err

# ==============================================================================
# HOW TO USE THIS SCRIPT:
# 1. On your local machine, build the Docker image:
#    docker build -f Dockerfile.stm -t r_hello_world:v1 .
# 
# 2. Save the image to a tar file:
#    docker save -o r_hello_world_v1.tar r_hello_world:v1
# 
# 3. Transfer the tar file to the cluster:
#    Put the tar file in your $HOME/docker_images/ directory on the cluster.
#    (Create the directory if it doesn't exist: mkdir -p ~/docker_images)
# 
# 4. Transfer this Slurm script to the cluster.
# 
# 5. Submit the job on the cluster:
#    sbatch slurm_hello.sh
# ==============================================================================

# 1. Create logs directory for Slurm output
mkdir -p slurm_log

# 2. Define image info
IMAGE_NAME="r_hello_world"
VERSION="v1"

echo "Starting job at $(date)..."

# 3. Ensure the specific Docker image is loaded on the node.
# Compute nodes might not have internet access or might have empty image caches.
# We load it from a pre-saved .tar file just in case.
if ! docker image inspect ${IMAGE_NAME}:${VERSION} >/dev/null 2>&1; then
    echo "Image ${IMAGE_NAME}:${VERSION} not found on this node."
    echo "Loading from \$HOME/docker_images/${IMAGE_NAME}_${VERSION}.tar..."
    docker load < \$HOME/docker_images/${IMAGE_NAME}_${VERSION}.tar
fi

# 4. Run the Docker container
# --rm removes the container after it finishes running
echo "Running Docker container..."
docker run --rm ${IMAGE_NAME}:${VERSION}

echo "Job finished at $(date)."
