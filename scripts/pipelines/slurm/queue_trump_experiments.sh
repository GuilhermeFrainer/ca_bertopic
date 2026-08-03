#!/bin/bash

# ==============================================================================
# Master Script to queue all standard Trump experiments on SLURM
# Generates one separate job for each model instance index (1 to 15).
# ==============================================================================

# Models to run
MODELS=("aligned_umap" "append_umap" "baseline" "mv_co_reg_spectral" "mv_co_reg_spectral_info0" "mv_spectral" "mv_spectral_info0" "umap_spectral")

# Ensure slurm log directory exists
mkdir -p slurm_log

# 1. Queue CA-BERTopic experiments (Non-STM)
# Each of the 8 models has 5 standard configurations (nr_topics = 10, 20, 30, 40, 50) and 3 seeds (15 total runs)
for model in "${MODELS[@]}"; do
    for model_idx in {1..15}; do
        job_name="ca_bertopic_trump_${model}_m${model_idx}"
        echo "Queuing job: $job_name"

        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=cidia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm_log/%x_%j.out
#SBATCH --error=slurm_log/%x_%j.err

echo "Job started at \$(date) on \$(hostname)"

# 1. Setup SCRATCH Workspace
mkdir -p \$SCRATCH/ca_bertopic
mkdir -p \$SCRATCH/ca_bertopic/{data/processed,results,models,logs,output,tables}

# 2. Sync Code base
rsync -av --exclude='data/' --exclude='models/' --exclude='results/' --exclude='logs/' \\
    --exclude='output/' --exclude='tables/' --exclude='.venv/' --exclude='.git/' \\
    \$HOME/ca_bertopic/ \$SCRATCH/ca_bertopic/

cd \$SCRATCH/ca_bertopic

# 3. Sync specific data file (Trump embeddings)
rsync -a \$HOME/ca_bertopic/data/processed/trump_embeddings.parquet \$SCRATCH/ca_bertopic/data/processed/

# 4. Export UV path
export PATH="\$HOME/.local/bin:\$PATH"

# 5. Run specific model instance
uv run python scripts/experiments/run_optimizer.py --exp trump/trump_standard_${model} --model ${model_idx}

# 6. Sync results back to HOME/slurm
mkdir -p \$HOME/slurm/{results,logs,output,tables,models}
rsync -a \$SCRATCH/ca_bertopic/results/ \$HOME/slurm/results/
rsync -a \$SCRATCH/ca_bertopic/logs/ \$HOME/slurm/logs/
rsync -a \$SCRATCH/ca_bertopic/output/ \$HOME/slurm/output/
rsync -a \$SCRATCH/ca_bertopic/tables/ \$HOME/slurm/tables/
rsync -a \$SCRATCH/ca_bertopic/models/ \$HOME/slurm/models/

echo "Job finished at \$(date)"
EOF
    done
done

# 2. Queue STM experiments via Docker
# For Trump STM standard, K values are 10, 20, 30, 40, 50 and 3 seeds
K_VALUES=(10 20 30 40 50)
SEEDS=(36201624 62613654 57116123)
IMAGE_NAME="cast"
VERSION="stm-lite-v0.1.0"
FORMULA="~ log(favorites + 1) + log(retweets + 1) + date + as.factor(device) + as.factor(is_retweet) + as.factor(is_deleted) + as.factor(is_flagged)"

for k in "${K_VALUES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        model_id="stm_k${k}_seed${seed}"
        job_name="stm_trump_k${k}_s${seed}"
        echo "Queuing job: $job_name"

        output_dir="results/trump_${model_id}"
        model_path="models/trump_${model_id}.rds"

        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=cidia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm_log/%x_%j.out
#SBATCH --error=slurm_log/%x_%j.err

echo "Job started at \$(date) on \$(hostname)"

# 1. Setup SCRATCH Workspace
mkdir -p \$SCRATCH/ca_bertopic
mkdir -p \$SCRATCH/ca_bertopic/{data/processed,results,models,logs}

# 2. Sync Code base
rsync -av --exclude='data/' --exclude='models/' --exclude='results/' --exclude='logs/' \
    --exclude='.venv/' --exclude='.git/' \
    \$HOME/ca_bertopic/ \$SCRATCH/ca_bertopic/

# 3. Sync specific data file (Trump STM data)
rsync -a \$HOME/ca_bertopic/data/processed/trump_stm_data.rds \$SCRATCH/ca_bertopic/data/processed/

# 4. Ensure Docker image is loaded
if ! docker image inspect ${IMAGE_NAME}:${VERSION} >/dev/null 2>&1; then
    docker load < \$HOME/docker_images/${IMAGE_NAME}_${VERSION}.tar
fi

# 5. Run training via Docker
mkdir -p \$SCRATCH/ca_bertopic/${output_dir}
docker run --rm \
    -v \$SCRATCH/ca_bertopic:/app/ca_bertopic \
    -w /app/ca_bertopic \
    -e RENV_PATHS_LIBRARY=/app/renv/library \
    ${IMAGE_NAME}:${VERSION} \
    Rscript scripts/r_scripts/train_stm.R \
    --rds_path "data/processed/trump_stm_data.rds" \
    --k "${k}" \
    --output_dir "${output_dir}" \
    --seed "${seed}" \
    --model_path "${model_path}" \
    --prevalence_formula "${FORMULA}"

# 6. Sync results back to HOME/slurm
mkdir -p \$HOME/slurm/{results,models,logs}
rsync -a \$SCRATCH/ca_bertopic/${output_dir}/ \$HOME/slurm/results/${model_id}/
rsync -a \$SCRATCH/ca_bertopic/${model_path}   \$HOME/slurm/models/
rsync -a \$SCRATCH/ca_bertopic/logs/           \$HOME/slurm/logs/

echo "Job finished at \$(date)"
EOF
    done
done

echo "------------------------------------------------"
echo "All Trump jobs have been dispatched to the scheduler."
