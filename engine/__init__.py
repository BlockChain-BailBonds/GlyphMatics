"""Compatibility package for the legacy GlyphMatics tensor engine.

The canonical high-level API lives in :mod:`glyphmatics`, but historical
callers import ``engine.glyphmatics_core`` directly. This package preserves
that import path while correcting legacy tensor-contract edge cases.
"""

import numpy as np

from . import glyphmatics_core as _core


def _rank_capped_merge(a: np.ndarray, b: np.ndarray, rank_cap: int = 64) -> np.ndarray:
    """Merge two tensors into a rank-capped embedding."""
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


def _dimension_safe_quantum_gravity_bridge(
    psi: np.ndarray, gravity: np.ndarray
) -> tuple[np.ndarray, float]:
    """Bridge unequal quantum/gravity tensors without an invalid contraction."""
    link = _rank_capped_merge(
        _core.kron(psi, gravity),
        _core.kron(_core.comm, _core.ent),
    )
    psi2 = np.asarray(psi).reshape(psi.shape[0], -1)
    gravity2 = np.asarray(gravity).reshape(gravity.shape[0], -1)
    rows = min(psi2.shape[0], gravity2.shape[0])
    cols = min(psi2.shape[1], gravity2.shape[1])
    delta_e = float(np.sum(psi2[:rows, :cols] * gravity2[:rows, :cols]))
    return link, delta_e


_core.merge_op = _rank_capped_merge
_core.quantum_gravity_bridge = _dimension_safe_quantum_gravity_bridge

from .glyphmatics_core import *  # noqa: E402,F401,F403

merge_op = _rank_capped_merge
quantum_gravity_bridge = _dimension_safe_quantum_gravity_bridge
