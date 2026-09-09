from glyphmatics.component import GlyphMaticsEngine


def test_full_chain():
    engine = GlyphMaticsEngine()
    state = engine.run_universe()
    assert state["U_score"] > 0.999
    assert state["arc_demo_ok"] is True
