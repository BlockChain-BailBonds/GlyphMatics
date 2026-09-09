from glyphmatics.component import GlyphMaticsEngine


def test_full_chain():
    engine = GlyphMaticsEngine()
    state = engine.run_universe()
    assert 0.0 <= state["U_score"] <= 1.0
    assert state["arc_demo_ok"] is True
