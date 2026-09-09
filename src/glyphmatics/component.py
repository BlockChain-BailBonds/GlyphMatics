# src/glyphmatics/component.py
from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np
from engine.glyphmatics_core import (
    quantum_gravity_bridge, gravity_arc_bridge, arc_defense_bridge,
    defense_policy_bridge, policy_quantum_bridge, quasicrystal_universal,
    unification_score, arc_meta_policy_solve,
    Ψ_quant, G_μν, Γ_arc, Θ_def, Π_meta, Φ_qc
)


def _consistent_color_map(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
) -> Dict[int, int] | None:
    """Infer a deterministic per-color substitution when training pairs support one."""
    mapping: Dict[int, int] = {}
    saw_change = False
    for inp, out in train_pairs:
        if inp.shape != out.shape:
            return None
        for source in np.unique(inp):
            targets = np.unique(out[inp == source])
            if len(targets) != 1:
                return None
            src = int(source)
            dst = int(targets[0])
            previous = mapping.get(src)
            if previous is not None and previous != dst:
                return None
            mapping[src] = dst
            saw_change = saw_change or src != dst
    return mapping if saw_change else None


def _apply_color_map(grid: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    original = np.asarray(grid)
    out = original.copy()
    for source, target in mapping.items():
        out[original == source] = target
    return out


class GlyphMaticsEngine:
    """High-level interface to the unified tensor engine."""

    def run_universe(self) -> Dict[str, object]:
        link, dE = quantum_gravity_bridge(Ψ_quant, G_μν)
        γ = gravity_arc_bridge(G_μν, Γ_arc)
        Θs = arc_defense_bridge(Γ_arc, Θ_def)
        Πu = defense_policy_bridge(Θs, Π_meta, R=γ, η=0.05)
        dΨdt = policy_quantum_bridge(Πu, Ψ_quant)
        Φr = quasicrystal_universal(Φ_qc)
        U = unification_score(Ψ_quant, G_μν, Γ_arc, Θ_def, Πu, Φr)

        demo_in = np.array([[1, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=np.uint8)
        demo_out = np.array([[2, 2, 0], [0, 0, 0], [0, 0, 0]], dtype=np.uint8)
        demo_pred = self.solve_arc([(demo_in, demo_out)], [demo_in])[0]

        return {
            "ΔE_qg": round(dE, 6),
            "Γ_arc_flow": round(γ, 6),
            "Π_trace": round(float(np.trace(Πu)), 6),
            "dΨ_dt_shape": tuple(dΨdt.shape),
            "U_score": round(U, 6),
            "arc_demo_ok": bool(np.array_equal(demo_pred, demo_out)),
        }

    def solve_arc(
        self,
        train_pairs: List[Tuple[np.ndarray, np.ndarray]],
        test_inputs: List[np.ndarray]
    ) -> List[np.ndarray]:
        color_map = _consistent_color_map(train_pairs)
        if color_map is not None:
            return [_apply_color_map(grid, color_map) for grid in test_inputs]
        return arc_meta_policy_solve(train_pairs, test_inputs)


class UnifiedARCComponent(GlyphMaticsEngine):
    """Backward-compatible name for the public ARC component API."""


__all__ = [
    "GlyphMaticsEngine",
    "UnifiedARCComponent",
    "Ψ_quant", "G_μν", "Γ_arc", "Θ_def", "Π_meta", "Φ_qc",
    "unification_score", "quantum_gravity_bridge",
]
