#!/bin/bash

# ==============================================================================
# Master Script to queue all standard experiments on SLURM
# Generates one separate job for each dataset and model combination.
# Ignores the Trump dataset.
# ==============================================================================

# ------------------------------------------------------------------------------
# Default Configurations & Resource Allocations
# ------------------------------------------------------------------------------
PROJECT_NAME="ca_bertopic"

DEFAULT_DATASETS=("anes" "fed" "gadarian" "yelp")

DEFAULT_MEM="32G"
DEFAULT_CPUS=4
DEFAULT_TIME="24:00:00"

STM_MEM="16G"
STM_CPUS=4
STM_TIME="24:00:00"

ALL_MODELS=(
    "aligned_umap"
    "aligned_umap_mv_k_means"
    "aligned_umap_mv_spherical_k_means"
    "append_umap"
    "append_umap_mv_co_reg_spectral"
    "append_umap_mv_co_reg_spectral_info0"
    "append_umap_mv_k_means"
    "append_umap_mv_spectral"
    "append_umap_mv_spectral_info0"
    "append_umap_mv_spherical_k_means"
    "baseline"
    "k_means"
    "mv_co_reg_spectral"
    "mv_co_reg_spectral_info0"
    "mv_k_means"
    "mv_spectral"
    "mv_spectral_info0"
    "mv_spherical_k_means"
    "pca_k_means"
    "pca_mv_co_reg_spectral"
    "pca_mv_k_means"
    "pca_mv_spectral"
    "pca_mv_spherical_k_means"
    "stm"
    "tritopic"
    "umap_spectral"
)

# ------------------------------------------------------------------------------
# Help Function
# ------------------------------------------------------------------------------
show_help() {
    cat << EOF
Usage: queue_exp.sh [OPTIONS]

Options:
  -d, --dataset, --datasets DATASETS  Comma-separated list of datasets to run.
                                      Available: anes, fed, gadarian, yelp.
                                      Default: all 4 datasets.

  -m, --model, --models MODELS        Comma-separated list of models or model categories to run.
                                      Categories: baseline, stm, spectral, kmeans, spherical, pca, umap.
                                      Default: all 21 standard models.

  -x, --exclude PATTERNS              Comma-separated keywords or categories to exclude.
                                      Example: -x pca,k_means (excludes PCA & K-Means models).

  -b, --split, --breakdown            Split each experiment into separate Slurm jobs for each
                                      topic-count and seed combination (default: 15 runs per model).

      --runs, --model-idx RUNS        Comma-separated list or range of model configuration
                                      indices to run when split is active (default: 1-15).
                                      Examples: --runs 1..15, --runs 1,2,5, --runs 1-5.

  -s, --stemmed                       Run experiments on stemmed dataset versions (e.g. fed_stemmed).

      --keep-rep-stopwords            Keep English stop words in topic representations (default: removed).

  -t, --test                          Run a minimal test set (baseline & stm).
                                      Defaults to 'fed' dataset if -d is not specified.

  -n, --dry-run                       Preview jobs to be submitted without executing sbatch.

  -l, --list                          List resolved dataset & model combinations and exit.

  -y, --yes                           Skip confirmation prompt when submitting >10 jobs.

      --mem MEMORY                    Override memory per job (e.g. 16G, 64G).

      --cpus CPUS                     Override CPUs per task (e.g. 4, 8).

      --time TIME                     Override time limit per job (e.g. 12:00:00).

  -h, --help                          Show this help message.

Examples:
  ./queue_exp.sh -d fed
  ./queue_exp.sh -d yelp -m mv_spectral -b
  ./queue_exp.sh -d fed --split --runs 1..5
  ./queue_exp.sh -d fed --stemmed
  ./queue_exp.sh -d fed --test
  ./queue_exp.sh -x pca,k_means
  ./queue_exp.sh -d gadarian,anes -m baseline,stm,mv_spectral -b -n
EOF
}

# ------------------------------------------------------------------------------
# Argument Parsing
# ------------------------------------------------------------------------------
RAW_DATASETS=""
RAW_MODELS=""
RAW_EXCLUDES=""
RAW_RUNS=""
SPLIT_EXPERIMENTS=false
DEFAULT_MODEL_INDICES=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
USE_STEMMED=false
KEEP_REP_STOPWORDS=false
IS_TEST=false
DRY_RUN=false
LIST_ONLY=false
AUTO_YES=false

MEM_OVERRIDE=""
CPUS_OVERRIDE=""
TIME_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--dataset|--datasets)
            RAW_DATASETS="$2"
            shift 2
            ;;
        -m|--model|--models)
            RAW_MODELS="$2"
            shift 2
            ;;
        -x|--exclude)
            RAW_EXCLUDES="$2"
            shift 2
            ;;
        -b|--split|--breakdown)
            SPLIT_EXPERIMENTS=true
            shift
            ;;
        --runs|--model-idx)
            SPLIT_EXPERIMENTS=true
            RAW_RUNS="$2"
            shift 2
            ;;
        -s|--stemmed)
            USE_STEMMED=true
            shift
            ;;
        --keep-rep-stopwords)
            KEEP_REP_STOPWORDS=true
            shift
            ;;
        -t|--test)
            IS_TEST=true
            shift
            ;;
        -n|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -l|--list)
            LIST_ONLY=true
            shift
            ;;
        -y|--yes)
            AUTO_YES=true
            shift
            ;;
        --mem)
            MEM_OVERRIDE="$2"
            shift 2
            ;;
        --cpus)
            CPUS_OVERRIDE="$2"
            shift 2
            ;;
        --time)
            TIME_OVERRIDE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Error: Unknown option '$1'"
            show_help
            exit 1
            ;;
    esac
done

# ------------------------------------------------------------------------------
# Resolve Model Indices for Split Mode
# ------------------------------------------------------------------------------
MODEL_INDICES=("${DEFAULT_MODEL_INDICES[@]}")
if [ -n "$RAW_RUNS" ]; then
    MODEL_INDICES=()
    IFS=',' read -ra RUN_PARTS <<< "$RAW_RUNS"
    for part in "${RUN_PARTS[@]}"; do
        part_clean=$(echo "$part" | xargs)
        if [[ "$part_clean" =~ ^([0-9]+)\.\.([0-9]+)$ ]] || [[ "$part_clean" =~ ^([0-9]+)\-([0-9]+)$ ]]; then
            start_idx="${BASH_REMATCH[1]}"
            end_idx="${BASH_REMATCH[2]}"
            for ((idx=start_idx; idx<=end_idx; idx++)); do
                MODEL_INDICES+=("$idx")
            done
        elif [[ "$part_clean" =~ ^[0-9]+$ ]]; then
            MODEL_INDICES+=("$part_clean")
        else
            echo "Warning: Unrecognized run index or range '$part_clean'. Ignoring."
        fi
    done
fi

# ------------------------------------------------------------------------------
# Resolve Datasets
# ------------------------------------------------------------------------------
TARGET_DATASETS=()
if [ -n "$RAW_DATASETS" ]; then
    IFS=',' read -ra ADDR <<< "$RAW_DATASETS"
    for ds in "${ADDR[@]}"; do
        ds_clean=$(echo "$ds" | xargs)
        if [ -n "$ds_clean" ]; then
            TARGET_DATASETS+=("$ds_clean")
        fi
    done
elif [ "$IS_TEST" = true ]; then
    TARGET_DATASETS=("fed")
else
    TARGET_DATASETS=("${DEFAULT_DATASETS[@]}")
fi

# ------------------------------------------------------------------------------
# Resolve Models
# ------------------------------------------------------------------------------
INITIAL_MODELS=()
if [ "$IS_TEST" = true ] && [ -z "$RAW_MODELS" ]; then
    INITIAL_MODELS=("baseline" "stm")
elif [ -n "$RAW_MODELS" ]; then
    IFS=',' read -ra ADDR <<< "$RAW_MODELS"
    for item in "${ADDR[@]}"; do
        item_clean=$(echo "$item" | xargs)
        if [ -n "$item_clean" ]; then
            matched=false
            case "$item_clean" in
                kmeans|k_means)
                    for m in "${ALL_MODELS[@]}"; do
                        if [[ "$m" == *"k_means"* ]]; then INITIAL_MODELS+=("$m"); matched=true; fi
                    done
                    ;;
                spherical)
                    for m in "${ALL_MODELS[@]}"; do
                        if [[ "$m" == *"spherical"* ]]; then INITIAL_MODELS+=("$m"); matched=true; fi
                    done
                    ;;
                pca)
                    for m in "${ALL_MODELS[@]}"; do
                        if [[ "$m" == *"pca"* ]]; then INITIAL_MODELS+=("$m"); matched=true; fi
                    done
                    ;;
                spectral)
                    for m in "${ALL_MODELS[@]}"; do
                        if [[ "$m" == *"spectral"* ]]; then INITIAL_MODELS+=("$m"); matched=true; fi
                    done
                    ;;
                umap)
                    for m in "${ALL_MODELS[@]}"; do
                        if [[ "$m" == *"umap"* ]]; then INITIAL_MODELS+=("$m"); matched=true; fi
                    done
                    ;;
            esac

            if [ "$matched" = false ]; then
                for m in "${ALL_MODELS[@]}"; do
                    if [ "$m" = "$item_clean" ] || [[ "$m" == *"$item_clean"* ]]; then
                        INITIAL_MODELS+=("$m")
                        matched=true
                    fi
                done
            fi

            if [ "$matched" = false ]; then
                echo "Warning: Model or category '$item_clean' did not match any known model."
            fi
        fi
    done
else
    INITIAL_MODELS=("${ALL_MODELS[@]}")
fi

# Remove duplicates from INITIAL_MODELS
TARGET_MODELS=()
for m in "${INITIAL_MODELS[@]}"; do
    already_added=false
    for tm in "${TARGET_MODELS[@]}"; do
        if [ "$tm" = "$m" ]; then already_added=true; break; fi
    done
    if [ "$already_added" = false ]; then
        TARGET_MODELS+=("$m")
    fi
done

# ------------------------------------------------------------------------------
# Apply Exclusions
# ------------------------------------------------------------------------------
EXCLUDE_LIST=()
if [ -n "$RAW_EXCLUDES" ]; then
    IFS=',' read -ra ADDR <<< "$RAW_EXCLUDES"
    for ex in "${ADDR[@]}"; do
        ex_clean=$(echo "$ex" | xargs)
        if [ -n "$ex_clean" ]; then
            if [ "$ex_clean" = "kmeans" ]; then ex_clean="k_means"; fi
            EXCLUDE_LIST+=("$ex_clean")
        fi
    done
fi

FINAL_MODELS=()
for m in "${TARGET_MODELS[@]}"; do
    excluded=false
    for ex in "${EXCLUDE_LIST[@]}"; do
        if [[ "$m" == *"$ex"* ]]; then
            excluded=true
            break
        fi
    done
    if [ "$excluded" = false ]; then
        FINAL_MODELS+=("$m")
    fi
done

if [ ${#FINAL_MODELS[@]} -eq 0 ]; then
    echo "Error: No models selected after applying filters and exclusions."
    exit 1
fi

if [ ${#TARGET_DATASETS[@]} -eq 0 ]; then
    echo "Error: No datasets selected."
    exit 1
fi

# ------------------------------------------------------------------------------
# Calculate & Print Summary
# ------------------------------------------------------------------------------
NUM_NON_STM_MODELS=0
for m in "${FINAL_MODELS[@]}"; do
    if [ "$m" != "stm" ]; then
        NUM_NON_STM_MODELS=$((NUM_NON_STM_MODELS + 1))
    fi
done

if [ "$SPLIT_EXPERIMENTS" = true ]; then
    TOTAL_JOBS=$(( ${#TARGET_DATASETS[@]} * NUM_NON_STM_MODELS * ${#MODEL_INDICES[@]} ))
else
    TOTAL_JOBS=$(( ${#TARGET_DATASETS[@]} * NUM_NON_STM_MODELS ))
fi

echo "================================================================="
echo " Experiment Submission Plan"
echo "================================================================="
echo " Datasets (${#TARGET_DATASETS[@]}):  ${TARGET_DATASETS[*]}"
echo " Models (${#FINAL_MODELS[@]}):    ${FINAL_MODELS[*]}"
if [ "$SPLIT_EXPERIMENTS" = true ]; then
    echo " Mode:         SPLIT / BREAKDOWN (${#MODEL_INDICES[@]} runs per model: ${MODEL_INDICES[*]})"
else
    echo " Mode:         STANDARD (1 job per model)"
fi
echo " Total Jobs:   $TOTAL_JOBS"
if [ "$USE_STEMMED" = true ]; then
    echo " Variant:      STEMMED (using clean_text_stemmed)"
fi
if [ -n "$RAW_EXCLUDES" ]; then
    echo " Exclusions:   $RAW_EXCLUDES"
fi
if [ "$DRY_RUN" = true ]; then
    echo " Dry Run:      YES (no jobs will be submitted)"
fi
echo "================================================================="

if [ "$LIST_ONLY" = true ]; then
    echo ""
    echo "Resolved Jobs List:"
    for dataset in "${TARGET_DATASETS[@]}"; do
        exp_dir="${dataset}"
        if [ "$USE_STEMMED" = true ]; then exp_dir="${dataset}_stemmed"; fi
        for model in "${FINAL_MODELS[@]}"; do
            if [ "$model" = "stm" ]; then continue; fi
            if [ "$SPLIT_EXPERIMENTS" = true ]; then
                for model_idx in "${MODEL_INDICES[@]}"; do
                    echo " - Dataset: $dataset (Config dir: $exp_dir) | Model: $model | Run: #$model_idx"
                done
            else
                echo " - Dataset: $dataset (Config dir: $exp_dir) | Model: $model (All runs)"
            fi
        done
    done
    exit 0
fi

# Confirm if submitting many jobs
if [ "$DRY_RUN" = false ] && [ "$AUTO_YES" = false ] && [ $TOTAL_JOBS -gt 10 ]; then
    read -p "You are about to submit $TOTAL_JOBS jobs to SLURM. Continue? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# Ensure slurm log directory exists
mkdir -p slurm_log

JOB_COUNT=0

for dataset in "${TARGET_DATASETS[@]}"; do
    exp_dir="${dataset}"
    job_dataset="${dataset}"
    if [ "$USE_STEMMED" = true ]; then
        exp_dir="${dataset}_stemmed"
        job_dataset="${dataset}_stemmed"
    fi

    for model in "${FINAL_MODELS[@]}"; do
        if [ "$model" = "stm" ]; then
            echo "Warning: Model 'stm' requested for dataset '$dataset', but STM jobs are currently disabled. Skipping."
            continue
        fi

        mem="${MEM_OVERRIDE:-$DEFAULT_MEM}"
        cpus="${CPUS_OVERRIDE:-$DEFAULT_CPUS}"
        time_limit="${TIME_OVERRIDE:-$DEFAULT_TIME}"

        # Data copy commands for non-STM
        # For Yelp, we use the 10k presampled embeddings file to avoid copying and loading the full 16GB dataset.
        if [ "$dataset" = "yelp" ]; then
            copy_commands="rsync -a \$HOME/${PROJECT_NAME}/data/processed/yelp_s10000_embeddings.parquet data/processed/yelp_embeddings.parquet"
        else
            copy_commands="rsync -a \$HOME/${PROJECT_NAME}/data/processed/${dataset}_embeddings.parquet data/processed/"
        fi

        # Common representation stopwords flag
        rep_flag="--remove-rep-stopwords"
        if [ "$KEEP_REP_STOPWORDS" = true ]; then
            rep_flag="--keep-rep-stopwords"
        fi

        if [ "$SPLIT_EXPERIMENTS" = true ]; then
            for model_idx in "${MODEL_INDICES[@]}"; do
                JOB_COUNT=$((JOB_COUNT + 1))
                job_name="${job_dataset}_${model}_m${model_idx}"
                run_command="uv run python scripts/experiments/run_optimizer.py --exp ${exp_dir}/${dataset}_standard_${model} --model ${model_idx} ${rep_flag}"

                if [ "$DRY_RUN" = true ]; then
                    echo "[$JOB_COUNT/$TOTAL_JOBS] [DRY RUN] Job: $job_name | Dataset: $dataset | Model: $model (Run #$model_idx) | Mem: $mem | CPUs: $cpus | Time: $time_limit"
                else
                    echo "[$JOB_COUNT/$TOTAL_JOBS] Queuing job for Dataset: $dataset | Model: $model | Run: #$model_idx"

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

# 1. Setup Job-Isolated SCRATCH Workspace & Cleanup Trap
JOB_SCRATCH="\$SCRATCH/${PROJECT_NAME}_\${SLURM_JOB_ID}"

cleanup() {
    trap - EXIT INT TERM
    echo "Cleaning up temporary scratch directory: \${JOB_SCRATCH}"
    cd "\$HOME" || cd /tmp
    if [ -n "\${JOB_SCRATCH}" ] && [ -d "\${JOB_SCRATCH}" ]; then
        rm -rf "\${JOB_SCRATCH}"
        if [ ! -d "\${JOB_SCRATCH}" ]; then
            echo "Scratch directory successfully removed."
        else
            echo "Warning: Failed to completely remove \${JOB_SCRATCH}."
        fi
    fi
}
trap cleanup EXIT INT TERM

mkdir -p "\${JOB_SCRATCH}"/{data/processed,results,models,logs,output,tables}

# 2. Sync Code base
rsync -av --exclude='data/' --exclude='models/' --exclude='results/' --exclude='logs/' \\
    --exclude='output/' --exclude='tables/' --exclude='.venv/' --exclude='.git/' \\
    \$HOME/${PROJECT_NAME}/ "\${JOB_SCRATCH}/"

cd "\${JOB_SCRATCH}"

# 3. Sync specific data files needed for this job
${copy_commands}

# 4. Export UV path so that it may be used
export PATH="\$HOME/.local/bin:\$PATH"

# 5. Run training and evaluation for single model run
${run_command}

# 6. Sync results back to HOME/slurm
mkdir -p \$HOME/slurm/{results,logs,output,tables,models}
rsync -a "\${JOB_SCRATCH}/results/" \$HOME/slurm/results/
rsync -a "\${JOB_SCRATCH}/logs/" \$HOME/slurm/logs/
rsync -a "\${JOB_SCRATCH}/output/" \$HOME/slurm/output/
rsync -a "\${JOB_SCRATCH}/tables/" \$HOME/slurm/tables/
rsync -a "\${JOB_SCRATCH}/models/" \$HOME/slurm/models/

echo "Job finished at \$(date)"
EOF
                fi
            done
        else
            JOB_COUNT=$((JOB_COUNT + 1))
            job_name="${job_dataset}_${model}"
            run_command="uv run python scripts/experiments/run_optimizer.py --exp ${exp_dir}/${dataset}_standard_${model} ${rep_flag}"

            if [ "$DRY_RUN" = true ]; then
                echo "[$JOB_COUNT/$TOTAL_JOBS] [DRY RUN] Job: $job_name | Dataset: $dataset | Model: $model | Mem: $mem | CPUs: $cpus | Time: $time_limit"
            else
                echo "[$JOB_COUNT/$TOTAL_JOBS] Queuing job for Dataset: $dataset | Model: $model"

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

# 1. Setup Job-Isolated SCRATCH Workspace & Cleanup Trap
JOB_SCRATCH="\$SCRATCH/${PROJECT_NAME}_\${SLURM_JOB_ID}"

cleanup() {
    trap - EXIT INT TERM
    echo "Cleaning up temporary scratch directory: \${JOB_SCRATCH}"
    cd "\$HOME" || cd /tmp
    if [ -n "\${JOB_SCRATCH}" ] && [ -d "\${JOB_SCRATCH}" ]; then
        rm -rf "\${JOB_SCRATCH}"
        if [ ! -d "\${JOB_SCRATCH}" ]; then
            echo "Scratch directory successfully removed."
        else
            echo "Warning: Failed to completely remove \${JOB_SCRATCH}."
        fi
    fi
}
trap cleanup EXIT INT TERM

mkdir -p "\${JOB_SCRATCH}"/{data/processed,results,models,logs,output,tables}

# 2. Sync Code base
rsync -av --exclude='data/' --exclude='models/' --exclude='results/' --exclude='logs/' \\
    --exclude='output/' --exclude='tables/' --exclude='.venv/' --exclude='.git/' \\
    \$HOME/${PROJECT_NAME}/ "\${JOB_SCRATCH}/"

cd "\${JOB_SCRATCH}"

# 3. Sync specific data files needed for this job
${copy_commands}

# 4. Export UV path so that it may be used
export PATH="\$HOME/.local/bin:\$PATH"

# 5. Run training and evaluation
${run_command}

# 6. Sync results back to HOME/slurm
mkdir -p \$HOME/slurm/{results,logs,output,tables,models}
rsync -a "\${JOB_SCRATCH}/results/" \$HOME/slurm/results/
rsync -a "\${JOB_SCRATCH}/logs/" \$HOME/slurm/logs/
rsync -a "\${JOB_SCRATCH}/output/" \$HOME/slurm/output/
rsync -a "\${JOB_SCRATCH}/tables/" \$HOME/slurm/tables/
rsync -a "\${JOB_SCRATCH}/models/" \$HOME/slurm/models/

echo "Job finished at \$(date)"
EOF
            fi
        fi
    done
done

echo "------------------------------------------------"
if [ "$DRY_RUN" = true ]; then
    echo "Dry run complete ($TOTAL_JOBS jobs simulated)."
else
    echo "All $TOTAL_JOBS jobs have been dispatched to the scheduler."
fi

