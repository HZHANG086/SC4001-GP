#!/bin/bash
# Environment setup only. Run training/checks through SLURM.
# Existing sc4001 is never overwritten or deleted by this script.
set -euo pipefail
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
SC4001_BOOTSTRAP="$HOME/.conda/envs/sc4001-bootstrap"
SC4001_TARGET="$HOME/.conda/envs/sc4001"
SC4001_SHARED=/tc1apps/2_conda_env/CZ4042_v5
if test -e "$SC4001_TARGET"; then
    echo "Refusing to overwrite $SC4001_TARGET. Diagnose or preserve it first." >&2
    exit 1
fi
if ! test -x "$SC4001_BOOTSTRAP/bin/python"; then
    CONDA_PKGS_DIRS="$HOME/.conda/sc4001-pkgs" conda create --solver classic \
        --repodata-fn current_repodata.json --override-channels -c conda-forge \
        -p "$SC4001_BOOTSTRAP" python=3.11 pip -y
fi
"$SC4001_BOOTSTRAP/bin/python" -m pip install conda-pack==0.8.1 setuptools==80.9.0
mkdir -p work
"$SC4001_BOOTSTRAP/bin/conda-pack" -p "$SC4001_SHARED" --format tar \
    --dest-prefix "$SC4001_TARGET" -o work/sc4001-environment.tar
mkdir -p "$SC4001_TARGET"
tar -xf work/sc4001-environment.tar -C "$SC4001_TARGET"
conda activate "$SC4001_TARGET"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python -I -c 'import os,sys; assert sys.prefix == os.environ["CONDA_PREFIX"], sys.prefix'
python -m pip install --no-user mkl==2024.0.0
python -m pip install --no-user -e .
bash scripts/install-runtime-hooks.sh
python -m pip check
python -m pip freeze > requirements-remote.txt
conda env export > environment-remote.yml
printf '\nEnvironment ready for SLURM validation.\n'
