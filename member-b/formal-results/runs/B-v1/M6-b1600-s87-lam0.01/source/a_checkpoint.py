"""Scoped NumPy RNG allowlist for this project's locally generated checkpoints."""

def rng_serialization_context():
    import numpy as np
    import torch
    return torch.serialization.safe_globals([
        np.core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.uint32)),
    ])
