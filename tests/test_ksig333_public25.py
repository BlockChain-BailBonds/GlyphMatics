from __future__ import annotations

import importlib.util
from pathlib import Path


def load_module():
    path = Path(__file__).parents[1] / "experiments" / "ksig333_public25.py"
    spec = importlib.util.spec_from_file_location("ksig333_public25", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_game(root: Path, slug: str, body: str) -> None:
    path = root / slug / "v1" / f"{slug}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def test_ksig333_round_trip_and_non_self_expansion(tmp_path: Path) -> None:
    ksig = load_module()
    root = tmp_path / "environment_files"

    shared = '''\nfrom enum import Enum\nclass GameAction(Enum):\n    ACTION1 = 1\n    ACTION2 = 2\n\ndef step(action):\n    score = 0\n    x, y = 1, 2\n    if action == GameAction.ACTION1:\n        x = x + 1\n    if action == GameAction.ACTION2:\n        y = y - 1\n    if x == y:\n        score += 1\n    return score\n'''
    variant = shared + '''\ndef render_frame(frame):\n    frame = frame + 1\n    return frame\n'''
    write_game(root, "ar25", shared)
    write_game(root, "bp35", variant)
    write_game(root, "cd82", shared)

    out = tmp_path / "out"
    score = ksig.build_public25(root, out, ["ar25", "bp35", "cd82"])
    assert score["round_trip_exact"] is True
    assert score["arc_gameplay_score"] is None
    assert 0.0 <= score["ksig_structural_transfer_score"] <= 1.0

    graph = ksig.decode_sigil(out / "public25.ksig.png")
    assert graph["glyph_count"] == 333
    assert graph["possible_directed_edges"] == 333 * 332
    assert len(graph["nodes"]) == 333
    assert all(edge["source"] != edge["target"] for edge in graph["active_edges"])

    cache = ksig.activate_cache(graph, ["action:ACTION1", "spatial:arith_add"])
    assert cache["active_glyphs"]
    assert set(cache["seed_glyphs"]).issubset(set(cache["active_glyphs"]))


def test_tamper_detection(tmp_path: Path) -> None:
    ksig = load_module()
    root = tmp_path / "environment_files"
    source = '''\ndef step(action):\n    if action:\n        return 1\n    return 0\n'''
    write_game(root, "ar25", source)
    write_game(root, "bp35", source)
    out = tmp_path / "out"
    ksig.build_public25(root, out, ["ar25", "bp35"])

    from PIL import Image

    path = out / "public25.ksig.png"
    image = Image.open(path).convert("RGB")
    pixels = list(image.getdata())
    r, g, b = pixels[100]
    pixels[100] = (r ^ 0x01, g, b)
    image.putdata(pixels)
    tampered = out / "tampered.png"
    image.save(tampered)

    try:
        ksig.decode_sigil(tampered)
    except ValueError:
        pass
    else:
        raise AssertionError("tampered Knowledge Sigil unexpectedly decoded")
