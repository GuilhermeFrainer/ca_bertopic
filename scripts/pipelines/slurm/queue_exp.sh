#!/bin/bash
# ==============================================================================
# Compatibility wrapper for queue_exp.py
# Dispatches to the Python implementation in scripts/experiments/queue_exp.py
# ==============================================================================

export PATH="$HOME/.local/bin:$PATH"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if command -v uv >/dev/null 2>&1; then
    exec uv run python "${REPO_ROOT}/scripts/experiments/queue_exp.py" "$@"
else
    exec python "${REPO_ROOT}/scripts/experiments/queue_exp.py" "$@"
fi
