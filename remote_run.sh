#!/usr/bin/env bash
#
# remote_run.sh — Drop-in replacement for "uv run train.py".
# Syncs train.py to a remote GPU machine, runs training, streams results back.
#
# Usage:
#   bash remote_run.sh > run.log 2>&1        # exactly like the local workflow
#   bash remote_run.sh --check               # connectivity + GPU check only
#   bash remote_run.sh --prepare             # run prepare.py on remote (data setup)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$SCRIPT_DIR/remote.toml"

# ---------------------------------------------------------------------------
# Config parsing (simple grep-based, no TOML library needed)
# ---------------------------------------------------------------------------

get_config() {
    local key="$1"
    local default="${2:-}"
    local val
    val=$(grep -E "^${key}\s*=" "$CONFIG" 2>/dev/null | head -1 | sed 's/^[^=]*=\s*//' | sed 's/\s*#.*//' | tr -d '"' | tr -d "'")
    if [ -z "$val" ]; then
        echo "$default"
    else
        echo "$val"
    fi
}

HOST=$(get_config "host")
USER=$(get_config "user" "ubuntu")
KEY_PATH=$(get_config "key_path" "~/.ssh/id_rsa")
WORKSPACE=$(get_config "workspace" "/workspace/autoresearch")
CACHE_DIR=$(get_config "cache_dir" "/root/.cache/autoresearch")
NUM_GPUS=$(get_config "num_gpus" "1")
USE_CONTAINER=$(get_config "use_container" "false")
IMAGE=$(get_config "image")
CONNECT_TIMEOUT=$(get_config "connect_timeout" "30")
RUN_TIMEOUT=$(get_config "run_timeout" "900")

# Expand tilde in key path
KEY_PATH="${KEY_PATH/#\~/$HOME}"

SSH_OPTS="-o ConnectTimeout=$CONNECT_TIMEOUT -o StrictHostKeyChecking=accept-new -o BatchMode=yes -i $KEY_PATH"
SSH_CMD="ssh $SSH_OPTS ${USER}@${HOST}"
SCP_CMD="scp $SSH_OPTS"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

if [ -z "$HOST" ]; then
    echo "ERROR: 'host' is not set in $CONFIG" >&2
    echo "Provision a GPU machine and set the IP/hostname in remote.toml" >&2
    exit 1
fi

if [ ! -f "$KEY_PATH" ]; then
    echo "ERROR: SSH key not found at $KEY_PATH" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Phase 1: Connectivity check
# ---------------------------------------------------------------------------

check_connection() {
    echo "Checking connectivity to ${USER}@${HOST}..." >&2
    if ! $SSH_CMD "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" 2>/dev/null; then
        echo "ERROR: Cannot reach ${USER}@${HOST} or no GPU detected" >&2
        exit 1
    fi
    echo "Connection OK, GPU available." >&2
}

if [ "${1:-}" = "--check" ]; then
    check_connection
    exit 0
fi

# ---------------------------------------------------------------------------
# Phase 2: Sync code to remote
# ---------------------------------------------------------------------------

sync_code() {
    echo "Syncing code to ${HOST}:${WORKSPACE}..." >&2
    $SSH_CMD "mkdir -p $WORKSPACE" 2>/dev/null

    rsync -az --delete \
        -e "ssh $SSH_OPTS" \
        --exclude '.git' \
        --exclude '.venv' \
        --exclude '__pycache__' \
        --exclude 'run.log' \
        --exclude 'results.tsv' \
        --exclude 'remote.toml' \
        --exclude '.remote_state' \
        --exclude 'worktrees' \
        --exclude 'dev' \
        "$SCRIPT_DIR/" "${USER}@${HOST}:${WORKSPACE}/"

    echo "Code synced." >&2
}

# ---------------------------------------------------------------------------
# Phase 3: Ensure data exists on remote
# ---------------------------------------------------------------------------

ensure_data() {
    echo "Checking remote data..." >&2
    if $SSH_CMD "test -d ${CACHE_DIR}/data && test -f ${CACHE_DIR}/tokenizer/tokenizer.pkl" 2>/dev/null; then
        echo "Remote data OK." >&2
    else
        echo "Data not found on remote. Running prepare.py..." >&2
        $SSH_CMD "cd $WORKSPACE && uv run prepare.py" >&2
        echo "Data preparation complete." >&2
    fi
}

if [ "${1:-}" = "--prepare" ]; then
    check_connection
    sync_code
    ensure_data
    exit 0
fi

# ---------------------------------------------------------------------------
# Phase 4: Run training
# ---------------------------------------------------------------------------

run_training() {
    local train_cmd

    if [ "$NUM_GPUS" -gt 1 ]; then
        train_cmd="uv run torchrun --nproc_per_node=$NUM_GPUS train.py"
    else
        train_cmd="uv run train.py"
    fi

    local remote_cmd
    if [ "$USE_CONTAINER" = "true" ] && [ -n "$IMAGE" ]; then
        remote_cmd="docker run --rm --gpus all \
            -v ${CACHE_DIR}:/root/.cache/autoresearch \
            -v ${WORKSPACE}:/workspace/autoresearch \
            ${IMAGE} $train_cmd"
    else
        remote_cmd="cd $WORKSPACE && $train_cmd"
    fi

    # Run with timeout. stdout/stderr stream directly to our stdout/stderr,
    # so the caller's redirection (> run.log 2>&1) captures everything.
    timeout "$RUN_TIMEOUT" $SSH_CMD "$remote_cmd"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

check_connection
sync_code
ensure_data
run_training
