#!/bin/bash
# Source from an sbatch job after changing to member-b/.  SC4001_CODE=formal uses the frozen
# copies (protocol/locked_src); anything else uses the development sources (src/).
source ../member-a/scripts/activate.sh
if [ "${SC4001_CODE:-dev}" = formal ]; then
    B_SRC="$PWD/protocol/locked_src"
else
    B_SRC="$PWD/src"
fi
export PYTHONPATH="$B_SRC:$(cd ../member-a && pwd)/protocol/locked_src"
export PYTHONDONTWRITEBYTECODE=1
