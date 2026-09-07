#!/bin/bash
# ==============================================================================
# Canonical SLURM Worker Script for Standard and Split Experiments
#
# Arguments:
#   $1: DATASET          (e.g., fed, yelp, anes, gadarian)
#   $2: EXP_TARGET       (e.g., fed/fed_standard_baseline)
#   $3: REP_FLAG         (--remove-rep-stopwords or --keep-rep-stopwords)
#   $4: MODEL_IDX        (optional: integer run index for split mode)
#
# Note: Any command-line resource flags passed to sbatch (e.g. --mem, --cpus-per-task,
# --time, --job-name) automatically override the fallback #SBATCH defaults below.
# ==============================================================================

#SBATCH --partition=cidia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm_log/%x_%j.out
#SBATCH --error=slurm_log/%x_%j.err

PROJECT_NAME="${PROJECT_NAME:-ca_bertopic}"

DATASET="${1:-$DATASET}"
EXP_TARGET="${2:-$EXP_TARGET}"
REP_FLAG="${3:-$REP_FLAG}"
MODEL_IDX="${4:-$MODEL_IDX}"

if [ -z "$DATASET" ] || [ -z "$EXP_TARGET" ]; then
    echo "ERROR: Missing required arguments. Usage: $0 <DATASET> <EXP_TARGET> [REP_FLAG] [MODEL_IDX]" >&2
    exit 1
fi

if [ -z "$REP_FLAG" ]; then
    REP_FLAG="--remove-rep-stopwords"
fi

echo "Job started at $(date) on $(hostname)"

# 1. Setup Job-Isolated SCRATCH Workspace & Cleanup Trap
JOB_SCRATCH_ROOT="$SCRATCH/job_${SLURM_JOB_ID}"
JOB_SCRATCH="${JOB_SCRATCH_ROOT}/${PROJECT_NAME}"

cleanup() {
    trap - EXIT INT TERM
    echo "Cleaning up temporary scratch directory: ${JOB_SCRATCH_ROOT}"
    cd "$HOME" || cd /tmp
    if [ -n "${JOB_SCRATCH_ROOT}" ] && [ -d "${JOB_SCRATCH_ROOT}" ]; then
        rm -rf "${JOB_SCRATCH_ROOT}"
        if [ ! -d "${JOB_SCRATCH_ROOT}" ]; then
            echo "Scratch directory successfully removed."
        else
            echo "Warning: Failed to completely remove ${JOB_SCRATCH_ROOT}."
        fi
    fi
}
trap cleanup EXIT INT TERM

mkdir -p "${JOB_SCRATCH}"/{data/processed,results,models,logs,output,tables}

# 2. Sync Code base
rsync -av --exclude='data/' --exclude='models/' --exclude='results/' --exclude='logs/' \
    --exclude='output/' --exclude='tables/' --exclude='.venv/' --exclude='.git/' \
    "$HOME/${PROJECT_NAME}/" "${JOB_SCRATCH}/"

FAST_TRITOPIC_SRC=""
if [ -d "$HOME/fast-tritopic" ]; then
    FAST_TRITOPIC_SRC="$HOME/fast-tritopic"
elif [ -d "$HOME/fast_tritopic" ]; then
    FAST_TRITOPIC_SRC="$HOME/fast_tritopic"
fi

if [ -n "$FAST_TRITOPIC_SRC" ]; then
    mkdir -p "${JOB_SCRATCH_ROOT}/fast-tritopic"
    rsync -avL --exclude='.venv/' --exclude='.git/' \
        "$FAST_TRITOPIC_SRC/" "${JOB_SCRATCH_ROOT}/fast-tritopic/"
fi

cd "${JOB_SCRATCH}"

# 3. Sync specific data files needed for this job
# For Yelp, we use the 10k presampled embeddings file to avoid copying and loading the full 16GB dataset.
if [ "$DATASET" = "yelp" ]; then
    rsync -a "$HOME/${PROJECT_NAME}/data/processed/yelp_s10000_embeddings.parquet" data/processed/yelp_embeddings.parquet
else
    rsync -a "$HOME/${PROJECT_NAME}/data/processed/${DATASET}_embeddings.parquet" data/processed/
fi

# 4. Export UV path and environment configuration
export PATH="$HOME/.local/bin:$PATH"
export UV_LINK_MODE="copy"

# Link pre-built virtual environment from HOME if available to avoid
# 10GB package copying and concurrent rebuild race conditions on scratch
if [ -d "$HOME/${PROJECT_NAME}/.venv" ]; then
    ln -sfn "$HOME/${PROJECT_NAME}/.venv" "${JOB_SCRATCH}/.venv"
    UV_SYNC_OPT="--no-sync"
else
    UV_SYNC_OPT=""
fi

# 5. Run training and evaluation
if [ -n "$MODEL_IDX" ] && [ "$MODEL_IDX" != "all" ] && [ "$MODEL_IDX" != "-" ]; then
    uv run ${UV_SYNC_OPT} python scripts/experiments/run_optimizer.py --exp "${EXP_TARGET}" --model "${MODEL_IDX}" ${REP_FLAG}
else
    uv run ${UV_SYNC_OPT} python scripts/experiments/run_optimizer.py --exp "${EXP_TARGET}" ${REP_FLAG}
fi

RUN_EXIT=$?
if [ $RUN_EXIT -ne 0 ]; then
    echo "ERROR: Experiment execution failed with exit code $RUN_EXIT" >&2
    exit $RUN_EXIT
fi

# 6. Sync results back to HOME/slurm
mkdir -p "$HOME/slurm"/{results,logs,output,tables,models}
rsync -a "${JOB_SCRATCH}/results/" "$HOME/slurm/results/"
rsync -a "${JOB_SCRATCH}/logs/" "$HOME/slurm/logs/"
rsync -a "${JOB_SCRATCH}/output/" "$HOME/slurm/output/"
rsync -a "${JOB_SCRATCH}/tables/" "$HOME/slurm/tables/"
rsync -a "${JOB_SCRATCH}/models/" "$HOME/slurm/models/"

echo "Job finished at $(date)"
