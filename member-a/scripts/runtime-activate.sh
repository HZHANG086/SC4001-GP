# Conda activation hook: isolate this project's packages and numerical libraries.
export _SC4001_SAVED_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
export _SC4001_SAVED_PYTHONNOUSERSITE="${PYTHONNOUSERSITE-}"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
