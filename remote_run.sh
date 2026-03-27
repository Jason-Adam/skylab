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
    val=$(grep -E "^${key}\s*=" "$CONFIG" 2>/dev/null | head -1 \
        | sed -E 's/^[^=]*=[[:space:]]*([^#]*).*/\1/' | tr -d '"' | tr -d "'" \
        | sed 's/[[:space:]]*$//')
    if [ -z "$val" ]; then
        echo "$default"
    else
        echo "$val"
    fi
}

# Validate that a config value contains only safe path characters
validate_path() {
    local name="$1" val="$2"
    if [[ ! "$val" =~ ^[a-zA-Z0-9_./:~-]+$ ]]; then
        echo "ERROR: '$name' contains unsafe characters: $val" >&2
        exit 1
    fi
}

HOST=$(get_config "host")
REMOTE_USER=$(get_config "user" "ubuntu")
KEY_PATH=$(get_config "key_path" "~/.ssh/id_rsa")
WORKSPACE=$(get_config "workspace" "/workspace/autoresearch")
CACHE_DIR=$(get_config "cache_dir" "~/.cache/autoresearch")
NUM_GPUS=$(get_config "num_gpus" "1")
USE_CONTAINER=$(get_config "use_container" "false")
IMAGE=$(get_config "image")
CONNECT_TIMEOUT=$(get_config "connect_timeout" "30")
RUN_TIMEOUT=$(get_config "run_timeout" "900")

# Expand tilde in key path
KEY_PATH="${KEY_PATH/#\~/$HOME}"

# Validate inputs that get interpolated into shell commands
validate_path "host" "$HOST" || true  # host may be empty (caught below)
validate_path "workspace" "$WORKSPACE"
validate_path "cache_dir" "$CACHE_DIR"
validate_path "key_path" "$KEY_PATH"
[ -n "$IMAGE" ] && validate_path "image" "$IMAGE"

# Build SSH command as a function to ensure proper quoting
run_ssh() {
    ssh -o "ConnectTimeout=$CONNECT_TIMEOUT" \
        -o "StrictHostKeyChecking=accept-new" \
        -o "BatchMode=yes" \
        -i "$KEY_PATH" \
        "${REMOTE_USER}@${HOST}" "$@"
}

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
    echo "Checking connectivity to ${REMOTE_USER}@${HOST}..." >&2
    if ! run_ssh "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" 2>/dev/null; then
        echo "ERROR: Cannot reach ${REMOTE_USER}@${HOST} or no GPU detected" >&2
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
    run_ssh "mkdir -p '${WORKSPACE}'" 2>/dev/null

    rsync -az --delete \
        -e "ssh -o ConnectTimeout=$CONNECT_TIMEOUT -o StrictHostKeyChecking=accept-new -o BatchMode=yes -i $KEY_PATH" \
        --exclude '.git' \
        --exclude '.venv' \
        --exclude '__pycache__' \
        --exclude 'run.log' \
        --exclude 'results.tsv' \
        --exclude 'remote.toml' \
        --exclude '.remote_state' \
        --exclude 'worktrees' \
        --exclude 'dev' \
        "$SCRIPT_DIR/" "${REMOTE_USER}@${HOST}:${WORKSPACE}/"

    echo "Code synced." >&2
}

# ---------------------------------------------------------------------------
# Phase 3: Ensure data exists on remote
# ---------------------------------------------------------------------------

ensure_data() {
    echo "Checking remote data..." >&2
    if run_ssh "test -d '${CACHE_DIR}/data' && test -f '${CACHE_DIR}/tokenizer/tokenizer.pkl'" 2>/dev/null; then
        echo "Remote data OK." >&2
    else
        echo "Data not found on remote. Running prepare.py..." >&2
        run_ssh "cd '${WORKSPACE}' && uv run prepare.py" >&2
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
        train_cmd="uv run torchrun --nproc_per_node=${NUM_GPUS} train.py"
    else
        train_cmd="uv run train.py"
    fi

    # Timeout is enforced on the remote side so the training process is killed
    # even if the SSH connection drops (avoids orphaned GPU jobs).
    local remote_cmd
    if [ "$USE_CONTAINER" = "true" ] && [ -n "$IMAGE" ]; then
        remote_cmd="timeout ${RUN_TIMEOUT} docker run --rm --gpus all \
            -v '${CACHE_DIR}':/root/.cache/autoresearch \
            -v '${WORKSPACE}':/workspace/autoresearch \
            '${IMAGE}' ${train_cmd}"
    else
        remote_cmd="cd '${WORKSPACE}' && timeout ${RUN_TIMEOUT} ${train_cmd}"
    fi

    local exit_code=0
    run_ssh "$remote_cmd" || exit_code=$?

    if [ "$exit_code" -eq 124 ]; then
        echo "ERROR: Run timed out after ${RUN_TIMEOUT} seconds" >&2
        exit 124
    elif [ "$exit_code" -ne 0 ]; then
        exit "$exit_code"
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

check_connection
sync_code
ensure_data
run_training
