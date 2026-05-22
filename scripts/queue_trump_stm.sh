#!/bin/bash

# ==============================================================================
# Master Script to queue the last 5 models for the Trump STM experiment
# ==============================================================================

# 1. Experiment Details (Hardcoded for speed/reliability)
DATASET="trump"
FORMULA="~ log(favorites + 1) + log(retweets + 1) + date + as.factor(device) + as.factor(is_retweet) + as.factor(is_deleted) + as.factor(is_flagged)"
SEED=36201624
IMAGE_NAME="cast"
VERSION="stm-lite-v0.1.0"

# The last 5 K values from trump_stm.yaml
K_VALUES=(500 550 600 650 700)

echo "Queuing 5 STM models for dataset: $DATASET"

for k in "${K_VALUES[@]}"; do
    model_id="stm_k${k}"
    echo "  -> Dispatching $model_id"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=stm_${DATASET}_k${k}
#SBATCH --partition=cidia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=slurm_log/%x_%j.out
#SBATCH --error=slurm_log/%x_%j.err

# 1. Setup SCRATCH Workspace
mkdir -p \$SCRATCH/ca_bertopic
mkdir -p \$SCRATCH/ca_bertopic/{data/processed,results,models,logs}

# 2. Sync Code and Data
rsync -av --exclude='data/' --exclude='models/' --exclude='results/' --exclude='logs/' \
    \$HOME/ca_bertopic/ \$SCRATCH/ca_bertopic/

rsync -av \$HOME/ca_bertopic/data/processed/trump_stm_data.rds \
    \$SCRATCH/ca_bertopic/data/processed/

# 3. Ensure Docker image is loaded
if ! docker image inspect ${IMAGE_NAME}:${VERSION} >/dev/null 2>&1; then
    docker load < \$HOME/docker_images/${IMAGE_NAME}_${VERSION}.tar
fi

# 4. RUN TRAINING via Docker
# We create a specific output folder for this K to avoid overwrites
# We expand these in the master script so they are hardcoded in the Slurm job
OUTPUT_DIR="results/${DATASET}_${model_id}"
MODEL_PATH="models/${DATASET}_${model_id}.rds"

mkdir -p \$SCRATCH/ca_bertopic/${OUTPUT_DIR}

docker run --rm \
    -v \$SCRATCH/ca_bertopic:/app/ca_bertopic \
    -w /app/ca_bertopic \
    -e RENV_PATHS_LIBRARY=/app/renv/library \
    ${IMAGE_NAME}:${VERSION} \
    Rscript scripts/train_stm.R \
    --rds_path data/processed/trump_stm_data.rds \
    --k ${k} \
    --output_dir ${OUTPUT_DIR} \
    --seed ${SEED} \
    --model_path ${MODEL_PATH} \
    --prevalence_formula "${FORMULA}"

# 5. Sync Results back to HOME/slurm
mkdir -p \$HOME/slurm/{results,models,logs}
rsync -av \$SCRATCH/ca_bertopic/${OUTPUT_DIR}/ \$HOME/slurm/results/${model_id}/
rsync -av \$SCRATCH/ca_bertopic/${MODEL_PATH}   \$HOME/slurm/models/
rsync -av \$SCRATCH/ca_bertopic/logs/           \$HOME/slurm/logs/
EOF

done

echo "------------------------------------------------"
echo "All 5 jobs have been dispatched to the scheduler."
