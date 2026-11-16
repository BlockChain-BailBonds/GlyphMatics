from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import struct
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

PUBLIC_GAMES = (
    "ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "lf52",
    "lp85", "ls20", "m0r0", "r11l", "re86", "s5i5", "sb26", "sc25", "sk48",
    "sp80", "su15", "tn36", "tr87", "tu93", "vc33", "wa30",
)

GLYPH_COUNT = 333
POSSIBLE_DIRECTED_EDGES = GLYPH_COUNT * (GLYPH_COUNT - 1)
MAGIC = b"KSIG3331"
BITS_PER_CHANNEL = 2

ZONE_RGB = {
    "causal": (220, 45, 45),
    "spatial": (55, 95, 220),
    "action": (55, 175, 85),
    "goal": (235, 190, 45),
    "abstraction": (145, 70, 190),
    "temporal": (235, 120, 35),
    "perception": (45, 185, 205),
    "invariant": (235, 235, 235),
    "uncertain": (135, 135, 135),
    "negative": (20, 20, 20),
}

USAGE_BY_ZONE = {
    "causal": "PREDICT",
    "spatial": "PLAN",
    "action": "ACT",
    "goal": "PLAN",
    "abstraction": "COMPRESS",
    "temporal": "PREDICT",
    "perception": "OBSERVE",
    "invariant": "VERIFY",
    "uncertain": "COMPARE",
    "negative": "VERIFY",
}

KEYWORD_ZONES = (
    ("move", "spatial"), ("position", "spatial"), ("rotate", "spatial"),
    ("flip", "spatial"), ("camera", "perception"), ("render", "perception"),
    ("frame", "perception"), ("sprite", "perception"), ("collid", "causal"),
    ("interact", "causal"), ("trigger", "causal"), ("condition", "causal"),
    ("action", "action"), ("step", "action"), ("input", "action"),
    ("win", "goal"), ("goal", "goal"), ("complete", "goal"),
    ("level", "goal"), ("reward", "goal"), ("score", "goal"),
    ("reset", "temporal"), ("timer", "temporal"), ("count", "temporal"),
    ("tick", "temporal"), ("visible", "invariant"), ("blocking", "invariant"),
    ("random", "uncertain"), ("unknown", "uncertain"), ("fail", "negative"),
    ("lose", "negative"), ("dead", "negative"),
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dotted_name(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def zone_for_text(text: str) -> str:
    lower = text.lower()
    for keyword, zone in KEYWORD_ZONES:
        if keyword in lower:
            return zone
    return "abstraction"


def feature_zone(feature: str) -> str:
    if ":" in feature:
        prefix = feature.split(":", 1)[0]
        if prefix in ZONE_RGB:
            return prefix
    return zone_for_text(feature)


def assign_glyph(feature: str) -> int:
    digest = hashlib.sha256(feature.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % GLYPH_COUNT


class MechanismExtractor(ast.NodeVisitor):
    """Extract implementation-independent mechanism motifs from public game code."""

    def __init__(self) -> None:
        self.features: Counter[str] = Counter()

    def add(self, zone: str, name: str, count: int = 1) -> None:
        self.features[f"{zone}:{name}"] += count

    def visit_If(self, node: ast.If) -> Any:
        text = ast.dump(node.test, annotate_fields=False, include_attributes=False)
        if "ACTION" in text or "GameAction" in text:
            self.add("action", "conditional_action_branch")
        else:
            self.add("causal", "conditional_transition")
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> Any:
        self.add("temporal", "bounded_iteration")
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> Any:
        self.add("temporal", "state_loop")
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        self.add("causal", type(node.op).__name__.lower())
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> Any:
        for op in node.ops:
            self.add("causal", f"compare_{type(op).__name__.lower()}")
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        op = type(node.op).__name__.lower()
        zone = "spatial" if op in {"add", "sub", "mult", "floordiv", "mod"} else "abstraction"
        self.add(zone, f"arith_{op}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        attr = node.attr
        if attr.startswith("ACTION") and attr[6:].isdigit():
            self.add("action", attr)
        else:
            zone = zone_for_text(attr)
            if zone != "abstraction":
                self.add(zone, f"attr_{attr.lower()}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        name = dotted_name(node.func)
        leaf = name.rsplit(".", 1)[-1].lower() if name else "call"
        zone = zone_for_text(name)
        if zone == "abstraction":
            if leaf in {"get", "set", "append", "add", "remove", "update", "copy"}:
                self.add("abstraction", f"container_{leaf}")
            else:
                self.add("abstraction", "generic_call")
        else:
            self.add(zone, f"call_{leaf}")
        if leaf in {"get_data", "set_data"} and node.args and isinstance(node.args[0], ast.Constant):
            value = node.args[0].value
            if isinstance(value, str):
                self.add(zone_for_text(value), "persistent_state_key")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> Any:
        value = node.value
        if isinstance(value, str):
            lower = value.lower()
            if lower.startswith("action") and lower[6:].isdigit():
                self.add("action", lower.upper())
            elif any(token in lower for token in ("goal", "win", "complete", "reward")):
                self.add("goal", "goal_label")
        elif isinstance(value, (int, float)) and value in range(16):
            self.add("perception", "grid_palette_value")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        zone = zone_for_text(node.name)
        if zone != "abstraction":
            self.add(zone, f"function_{node.name.lower()}")
        self.generic_visit(node)


@dataclass(frozen=True)
class GameKnowledge:
    slug: str
    version: str
    source_path: str
    source_sha256: str
    features: dict[str, int]

    @property
    def glyphs(self) -> set[int]:
        return {assign_glyph(feature) for feature in self.features}


def find_game_source(root: Path, slug: str) -> Path:
    candidates = sorted(root.glob(f"{slug}/*/{slug}.py"))
    if not candidates:
        candidates = sorted(root.glob(f"**/{slug}/*/{slug}.py"))
    if not candidates:
        raise FileNotFoundError(f"No source found for public game {slug} under {root}")
    return candidates[-1]


def extract_game(path: Path, slug: str) -> GameKnowledge:
    raw = path.read_bytes()
    tree = ast.parse(raw.decode("utf-8"), filename=str(path))
    extractor = MechanismExtractor()
    extractor.visit(tree)
    return GameKnowledge(
        slug=slug,
        version=path.parent.name,
        source_path=str(path),
        source_sha256=sha256_hex(raw),
        features=dict(sorted(extractor.features.items())),
    )


def dominant_zone(features: Iterable[str]) -> str:
    counts = Counter(feature_zone(feature) for feature in features)
    return counts.most_common(1)[0][0] if counts else "uncertain"


def tier_from_frequency(count: int, corpus_size: int) -> int:
    ratio = count / max(1, corpus_size)
    if ratio >= 0.80:
        return 0
    if ratio >= 0.50:
        return 1
    if ratio >= 0.25:
        return 2
    if count >= 2:
        return 3
    return 4


def build_graph(games: list[GameKnowledge]) -> dict[str, Any]:
    feature_to_games: dict[str, set[str]] = defaultdict(set)
    glyph_features: dict[int, set[str]] = defaultdict(set)
    glyph_game_counts: Counter[int] = Counter()
    edge_counts: Counter[tuple[int, int]] = Counter()

    for game in games:
        glyphs = sorted(game.glyphs)
        for feature in game.features:
            feature_to_games[feature].add(game.slug)
            glyph_features[assign_glyph(feature)].add(feature)
        for glyph in glyphs:
            glyph_game_counts[glyph] += 1
        for source in glyphs:
            for target in glyphs:
                if source != target:
                    edge_counts[(source, target)] += 1

    nodes: list[dict[str, Any]] = []
    for glyph in range(GLYPH_COUNT):
        features = sorted(glyph_features.get(glyph, set()))
        zone = dominant_zone(features)
        count = glyph_game_counts.get(glyph, 0)
        nodes.append({
            "glyph": glyph,
            "id": f"G{glyph:03d}",
            "zone": zone,
            "usage": USAGE_BY_ZONE[zone],
            "tier": tier_from_frequency(count, len(games)) if count else 5,
            "game_frequency": count,
            "features": features,
        })

    edges: list[dict[str, Any]] = []
    for (source, target), count in sorted(edge_counts.items()):
        target_zone = nodes[target]["zone"]
        edges.append({
            "source": source,
            "target": target,
            "weight": count,
            "confidence": round(count / max(1, len(games)), 6),
            "color_zone": target_zone,
            "usage": nodes[source]["usage"],
        })

    graph: dict[str, Any] = {
        "format": "ksig333-public-knowledge-v1",
        "glyph_count": GLYPH_COUNT,
        "expansion_rule": "each glyph can expand to every other glyph except itself",
        "possible_directed_edges": POSSIBLE_DIRECTED_EDGES,
        "source_scope": "ARC-AGI-3 public demonstration set only",
        "games": [{
            "slug": game.slug,
            "version": game.version,
            "source_sha256": game.source_sha256,
            "feature_count": sum(game.features.values()),
            "unique_features": len(game.features),
            "active_glyphs": len(game.glyphs),
        } for game in games],
        "nodes": nodes,
        "active_edges": edges,
        "features": [{
            "feature": feature,
            "glyph": assign_glyph(feature),
            "zone": feature_zone(feature),
            "games": sorted(slugs),
        } for feature, slugs in sorted(feature_to_games.items())],
    }
    graph["graph_sha256"] = sha256_hex(canonical_json_bytes(graph))
    return graph


def leave_one_out(games: list[GameKnowledge]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    all_slugs = {game.slug for game in games}
    for holdout in games:
        train = [game for game in games if game.slug != holdout.slug]
        train_features = set().union(*(set(game.features) for game in train))
        holdout_features = set(holdout.features)
        train_glyphs = set().union(*(game.glyphs for game in train))
        holdout_glyphs = holdout.glyphs

        train_pairs: set[tuple[int, int]] = set()
        for game in train:
            glyphs = game.glyphs
            train_pairs.update((a, b) for a in glyphs for b in glyphs if a != b)
        holdout_pairs = {(a, b) for a in holdout_glyphs for b in holdout_glyphs if a != b}

        f_cov = len(holdout_features & train_features) / max(1, len(holdout_features))
        g_cov = len(holdout_glyphs & train_glyphs) / max(1, len(holdout_glyphs))
        r_cov = len(holdout_pairs & train_pairs) / max(1, len(holdout_pairs))
        score = 0.4 * f_cov + 0.3 * g_cov + 0.3 * r_cov
        rows.append({
            "holdout": holdout.slug,
            "trained_on": sorted(all_slugs - {holdout.slug}),
            "feature_coverage": round(f_cov, 6),
            "glyph_coverage": round(g_cov, 6),
            "relation_coverage": round(r_cov, 6),
            "ksig_structural_transfer_score": round(score, 6),
        })

    mean = sum(row["ksig_structural_transfer_score"] for row in rows) / max(1, len(rows))
    return {
        "metric": "leave-one-public-game-out structural transfer; NOT an ARC gameplay score",
        "games": rows,
        "mean_ksig_structural_transfer_score": round(mean, 6),
    }


def _payload_blob(payload: bytes) -> bytes:
    compressed = zlib.compress(payload, 9)
    return MAGIC + bytes.fromhex(sha256_hex(payload)) + struct.pack(">Q", len(compressed)) + struct.pack(">I", zlib.crc32(compressed) & 0xFFFFFFFF) + compressed


def _image_size_for_blob(blob: bytes) -> int:
    bits = len(blob) * 8
    pixels = math.ceil(bits / (3 * BITS_PER_CHANNEL))
    side = max(1024, math.ceil(math.sqrt(pixels)))
    return int(math.ceil(side / 64) * 64)


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def render_sigil(graph: dict[str, Any], out_path: Path) -> dict[str, Any]:
    payload = canonical_json_bytes(graph)
    blob = _payload_blob(payload)
    size = _image_size_for_blob(blob)
    image = Image.new("RGB", (size, size), (248, 248, 248))
    draw = ImageDraw.Draw(image)
    center = size / 2
    max_radius = size * 0.44
    ring_step = max_radius / 6

    draw.ellipse((center - 18, center - 18, center + 18, center + 18), outline=(20, 20, 20), width=2)
    draw.text((12, 12), "KSIG-333 PUBLIC-25", fill=(10, 10, 10), font=_font())
    draw.text((12, 28), graph["graph_sha256"][:24], fill=(70, 70, 70), font=_font())

    node_positions: dict[int, tuple[float, float]] = {}
    for node in graph["nodes"]:
        glyph = int(node["glyph"])
        tier = int(node["tier"])
        angle = (2 * math.pi * glyph / GLYPH_COUNT) - math.pi / 2
        radius = ring_step * (min(tier, 5) + 1)
        x = center + radius * math.cos(angle)
        y = center + radius * math.sin(angle)
        node_positions[glyph] = (x, y)
        zone = str(node["zone"])
        color = ZONE_RGB[zone]
        rr = 5 + max(0, 5 - tier)
        box = (x - rr, y - rr, x + rr, y + rr)
        deg = math.degrees(angle)
        span = 2 + min(10, int(node["game_frequency"]))
        draw.arc(box, start=deg - span, end=deg + span + 180, fill=color, width=2)
        if node["game_frequency"]:
            draw.point((int(x), int(y)), fill=color)

    strongest = sorted(graph["active_edges"], key=lambda item: (-item["weight"], item["source"], item["target"]))[: min(1500, len(graph["active_edges"]))]
    for edge in strongest:
        a = node_positions[int(edge["source"])]
        b = node_positions[int(edge["target"])]
        draw.line((a[0], a[1], b[0], b[1]), fill=ZONE_RGB[str(edge["color_zone"])], width=1)

    pixels = list(image.getdata())
    capacity_bytes = (len(pixels) * 3 * BITS_PER_CHANNEL) // 8
    if len(blob) > capacity_bytes:
        raise ValueError(f"Sigil payload too large: {len(blob)} > {capacity_bytes}")

    bits: list[int] = []
    for byte in blob:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    mask = (1 << BITS_PER_CHANNEL) - 1
    out_pixels: list[tuple[int, int, int]] = []
    cursor = 0
    for pixel in pixels:
        channels = list(pixel)
        for i in range(3):
            value = 0
            for _ in range(BITS_PER_CHANNEL):
                value <<= 1
                if cursor < len(bits):
                    value |= bits[cursor]
                    cursor += 1
            channels[i] = (channels[i] & ~mask) | value
        out_pixels.append(tuple(channels))
    image.putdata(out_pixels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, format="PNG", optimize=False)
    return {
        "image": str(out_path),
        "width": size,
        "height": size,
        "payload_bytes": len(payload),
        "embedded_blob_bytes": len(blob),
        "capacity_bytes": capacity_bytes,
        "payload_sha256": sha256_hex(payload),
    }


def _extract_embedded_bytes(image: Image.Image) -> bytes:
    bits: list[int] = []
    mask = (1 << BITS_PER_CHANNEL) - 1
    for pixel in image.convert("RGB").getdata():
        for channel in pixel:
            value = channel & mask
            for shift in range(BITS_PER_CHANNEL - 1, -1, -1):
                bits.append((value >> shift) & 1)
    output = bytearray()
    for i in range(0, len(bits) - 7, 8):
        byte = 0
        for bit in bits[i:i + 8]:
            byte = (byte << 1) | bit
        output.append(byte)
    return bytes(output)


def decode_sigil(path: Path) -> dict[str, Any]:
    data = _extract_embedded_bytes(Image.open(path))
    if data[:len(MAGIC)] != MAGIC:
        raise ValueError("KSIG magic mismatch")
    off = len(MAGIC)
    expected_hash = data[off:off + 32].hex()
    off += 32
    comp_len = struct.unpack(">Q", data[off:off + 8])[0]
    off += 8
    expected_crc = struct.unpack(">I", data[off:off + 4])[0]
    off += 4
    compressed = data[off:off + comp_len]
    if len(compressed) != comp_len:
        raise ValueError("KSIG payload truncated")
    if (zlib.crc32(compressed) & 0xFFFFFFFF) != expected_crc:
        raise ValueError("KSIG compressed payload CRC mismatch")
    payload = zlib.decompress(compressed)
    if sha256_hex(payload) != expected_hash:
        raise ValueError("KSIG payload hash mismatch")
    graph = json.loads(payload.decode("utf-8"))
    if graph.get("possible_directed_edges") != POSSIBLE_DIRECTED_EDGES:
        raise ValueError("KSIG 333x332 invariant mismatch")
    if any(edge["source"] == edge["target"] for edge in graph.get("active_edges", [])):
        raise ValueError("KSIG self expansion detected")
    expected_graph_hash = graph.get("graph_sha256")
    graph_without_hash = dict(graph)
    graph_without_hash.pop("graph_sha256", None)
    if expected_graph_hash != sha256_hex(canonical_json_bytes(graph_without_hash)):
        raise ValueError("KSIG semantic graph hash mismatch")
    return graph


def activate_cache(graph: dict[str, Any], observed_features: Iterable[str], top_k: int = 24) -> dict[str, Any]:
    seed_glyphs = {assign_glyph(feature) for feature in observed_features}
    weights: Counter[int] = Counter()
    for edge in graph.get("active_edges", []):
        if int(edge["source"]) in seed_glyphs:
            weights[int(edge["target"])] += int(edge["weight"])
    for glyph in seed_glyphs:
        weights[glyph] += 10**9
    selected = [glyph for glyph, _ in weights.most_common(top_k)]
    node_by_id = {int(node["glyph"]): node for node in graph.get("nodes", [])}
    return {
        "seed_glyphs": sorted(seed_glyphs),
        "active_glyphs": selected,
        "nodes": [node_by_id[glyph] for glyph in selected if glyph in node_by_id],
        "edges": [edge for edge in graph.get("active_edges", []) if int(edge["source"]) in selected and int(edge["target"]) in selected],
    }


def build_public25(games_root: Path, out_dir: Path, slugs: Iterable[str] = PUBLIC_GAMES) -> dict[str, Any]:
    games = [extract_game(find_game_source(games_root, slug), slug) for slug in slugs]
    graph = build_graph(games)
    loo = leave_one_out(games)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "public25.ksig.json").write_bytes(canonical_json_bytes(graph))
    image_path = out_dir / "public25.ksig.png"
    image_stats = render_sigil(graph, image_path)
    decoded = decode_sigil(image_path)
    if canonical_json_bytes(decoded) != canonical_json_bytes(graph):
        raise AssertionError("Knowledge Sigil round-trip changed the graph")
    (out_dir / "leave_one_out.json").write_text(json.dumps(loo, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    score = {
        "arc_gameplay_score": None,
        "arc_gameplay_score_note": "Private ARC score is only produced by the official Kaggle evaluation.",
        "ksig_structural_transfer_score": loo["mean_ksig_structural_transfer_score"],
        "round_trip_exact": True,
        "graph_sha256": graph["graph_sha256"],
        "image": image_stats,
        "public_game_count": len(games),
    }
    (out_dir / "score.json").write_text(json.dumps(score, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the KSIG-333 Public-25 ARC-AGI-3 knowledge sigil")
    parser.add_argument("--games-root", type=Path, help="Path to ARC3.Games environment_files")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/ksig333"))
    parser.add_argument("--decode", type=Path, default=None)
    args = parser.parse_args()
    if args.decode is not None:
        graph = decode_sigil(args.decode)
        print(json.dumps({
            "graph_sha256": graph["graph_sha256"],
            "games": len(graph["games"]),
            "active_edges": len(graph["active_edges"]),
        }, indent=2))
        return
    if args.games_root is None:
        parser.error("--games-root is required unless --decode is used")
    print(json.dumps(build_public25(args.games_root, args.out_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
