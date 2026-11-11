#!/bin/bash
# Source from /share/zhangh/sc4001-gp-c/project. Never modify Assignment environments.
export GP_BASE=/share/zhangh/sc4001-gp-c
export TMPDIR="$GP_BASE/tmp"
export HF_HOME="$GP_BASE/cache/huggingface"
export UV_CACHE_DIR="$GP_BASE/cache/uv"
export PYTHONPATH="$PWD/member-c/src:$PWD/member-a/protocol/locked_src:$PWD/member-b/protocol/locked_src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PATH="$GP_BASE/env/bin:$PATH"
