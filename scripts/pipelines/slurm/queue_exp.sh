#!/bin/bash
# ==============================================================================
# Compatibility wrapper for queue_exp.py
# Dispatches to the Python implementation in scripts/experiments/queue_exp.py
# ==============================================================================

export PATH="$HOME/.local/bin:$PATH"
PROJECT_NAME="${PROJECT_NAME:-ca_bertopic}"
REPO_ROOT="$HOME/${PROJECT_NAME}"

# If not on the cluster, fall back to relative path for local runs
if [ ! -f "${REPO_ROOT}/scripts/experiments/queue_exp.py" ]; then
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi

exec uv run python "${REPO_ROOT}/scripts/experiments/queue_exp.py" "$@"
