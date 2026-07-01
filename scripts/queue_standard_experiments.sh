#!/bin/bash

# ==============================================================================
# Master Script to queue all standard experiments on SLURM
# Generates one separate job for each dataset and model combination.
# Ignores the Trump dataset.
# ==============================================================================

# Datasets to run (ignoring trump)
DATASETS=("anes" "fed" "gadarian" "yelp")

# Models to run
MODELS=("aligned_umap" "append_umap" "baseline" "mv_co_reg_spectral" "mv_co_reg_spectral_info0" "mv_spectral" "mv_spectral_info0" "umap_spectral" "stm")

# Ensure slurm log directory exists
mkdir -p slurm_log

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo "Queuing job for Dataset: $dataset | Model: $model"

        # Determine resource requirements and copy/run commands
        if [ "$model" = "stm" ]; then
            job_name="stm_${dataset}"
            mem="16G"
            cpus=4
            time_limit="24:00:00"

            # Determine dataset prefixes for STM files
            if [ "$dataset" = "yelp" ]; then
                data_prefix="yelp_s10000"
            else
                data_prefix="$dataset"
            fi

            # Data copy commands for STM (RDS, BoW, and Embeddings)
            copy_commands="rsync -a \$HOME/ca_bertopic/data/processed/${data_prefix}_stm_data.rds \$SCRATCH/ca_bertopic/data/processed/
rsync -a \$HOME/ca_bertopic/data/processed/${data_prefix}_bow.parquet \$SCRATCH/ca_bertopic/data/processed/
rsync -a \$HOME/ca_bertopic/data/processed/${data_prefix}_embeddings.parquet \$SCRATCH/ca_bertopic/data/processed/"

            # Run command for STM
            run_command="uv run python scripts/run_stm.py --exp ${dataset}/${dataset}_standard_stm"
        else
            job_name="ca_bertopic_${dataset}_${model}"
            mem="32G"
            cpus=4
            time_limit="24:00:00"

            # Data copy commands for non-STM
            # For Yelp, we use the 10k presampled embeddings file to avoid copying and loading the full 16GB dataset.
            if [ "$dataset" = "yelp" ]; then
                copy_commands="rsync -a \$HOME/ca_bertopic/data/processed/yelp_s10000_embeddings.parquet \$SCRATCH/ca_bertopic/data/processed/yelp_embeddings.parquet"
            else
                copy_commands="rsync -a \$HOME/ca_bertopic/data/processed/${dataset}_embeddings.parquet \$SCRATCH/ca_bertopic/data/processed/"
            fi

            # Run command for non-STM
            run_command="uv run python scripts/run_optimizer.py --exp ${dataset}/${dataset}_standard_${model}"
        fi

        # Submit to SLURM using a here-doc
        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=cidia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=${mem}
#SBATCH --cpus-per-task=${cpus}
#SBATCH --time=${time_limit}
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

# 3. Sync specific data files needed for this job
${copy_commands}

# 4. Export UV path so that it may be used
export PATH="\$HOME/.local/bin:\$PATH"

# 5. Run training and evaluation
${run_command}

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

echo "------------------------------------------------"
echo "All jobs have been dispatched to the scheduler."
