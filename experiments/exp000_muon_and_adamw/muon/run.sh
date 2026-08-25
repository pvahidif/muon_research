#!/usr/bin/env bash

set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$RUN_DIR/../../.." && pwd)"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}"

cd "$REPO_ROOT"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
    src/muon_research/scripts/run_optim_rules.py \
    --run_path "$RUN_DIR" \
    "$@"
