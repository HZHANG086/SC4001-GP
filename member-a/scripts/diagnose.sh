#!/bin/bash
# Read-only head-node checks; do not run nvidia-smi here.
set -u
hostname
pwd
module list 2>&1
module load anaconda
command -v conda
conda env list
MyTCinfo
sinfo -o '%P %a %l %D'
squeue -u "$USER"
df -h .
quota -s 2>/dev/null || true
