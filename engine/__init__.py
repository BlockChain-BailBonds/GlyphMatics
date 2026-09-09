"""Compatibility package for the legacy GlyphMatics tensor engine.

The canonical high-level API lives in :mod:`glyphmatics`, but historical
callers import ``engine.glyphmatics_core`` directly. This package preserves
that import path while correcting the legacy rank-compression contract.
"""

import numpy as np

from . import glyphmatics_core as _core


def _rank_capped_merge(a: np.ndarray, b: np.ndarray, rank_cap: int = 64) -> np.ndarray:
    """Merge two tensors into a rank-capped embedding.

    Legacy ``merge_op`` reconstructed the entire concatenated matrix, so its
    second dimension ignored ``rank_cap``.  It also required equal leading
    dimensions even though several bridge tensors naturally use different
    bond sizes.  Pad the smaller bond axis with zeros, compute one SVD, and
    return the compact ``U*S`` embedding with at most ``rank_cap`` columns.
    """
    A = np.asarray(a).reshape(a.shape[0], -1)
    B = np.asarray(b).reshape(b.shape[0], -1)
    rows = max(A.shape[0], B.shape[0])
    if A.shape[0] < rows:
        A = np.pad(A, ((0, rows - A.shape[0]), (0, 0)))
    if B.shape[0] < rows:
        B = np.pad(B, ((0, rows - B.shape[0]), (0, 0)))
    M = np.concatenate([A, B], axis=1)
    U, S, _ = np.linalg.svd(M, full_matrices=False)
    k = max(1, min(int(rank_cap), len(S), U.shape[1]))
    return U[:, :k] * S[:k]


# Patch the legacy module itself so both
# ``from engine import merge_op`` and
# ``from engine.glyphmatics_core import merge_op`` receive the corrected API.
_core.merge_op = _rank_capped_merge

from .glyphmatics_core import *  # noqa: E402,F401,F403

merge_op = _rank_capped_merge
