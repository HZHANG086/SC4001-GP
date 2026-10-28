#!/bin/bash
set -euo pipefail
if test "${CONDA_PREFIX##*/}" != sc4001; then
    echo 'Activate sc4001 before installing its runtime hooks.' >&2
    exit 1
fi
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
cp scripts/runtime-activate.sh "$CONDA_PREFIX/etc/conda/activate.d/sc4001_runtime.sh"
cp scripts/runtime-deactivate.sh "$CONDA_PREFIX/etc/conda/deactivate.d/sc4001_runtime.sh"
