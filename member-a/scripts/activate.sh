#!/bin/bash
# Source this file from an sbatch job after changing to project root.
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sc4001
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
