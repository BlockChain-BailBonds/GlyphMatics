from contextlib import redirect_stdout
import io

import pytest

from glyphmatics.programming_syntax import (
    EXECUTABLE_SYSTEMS,
    LANGUAGE_GLYPHS,
    PROGRAM_EXAMPLES,
    SYNTAX_GLYPHS,
    all_fixed_program_glyphs,
    canonicalize_program,
    decode_program_lossless,
    encode_program_lossless,
    iter_executable_system_rows,
    iter_program_training_rows,
    normalize_language,
    tokenize_program,
)


@pytest.mark.parametrize(
    ("language", "source"),
    [
        ("python", "def add(a, b):\n    return a + b"),
        ("javascript", "const total = left + right;"),
        ("typescript", "const total: number = left + right;"),
        ("rust", "fn add(a: i32, b: i32) -> i32 { a + b }"),
        ("go", "func add(a int, b int) int { return a + b }"),
        ("java", "int add(int a, int b) { return a + b; }"),
        ("c", "int main(void) { return 0; }"),
        ("cpp", "auto total = left + right;"),
        ("csharp", "var total = left + right;"),
        ("ruby", "def add(a, b)\n  a + b\nend"),
        ("php", "$total = $left + $right;"),
        ("swift", "let total = left + right"),
        ("kotlin", "val total = left + right"),
        ("bash", "if (( score > 80 )); then echo yes; fi"),
        ("sql", "SELECT left + right AS total;"),
    ],
)
def test_lossless_program_glyph_roundtrip(language, source):
    encoded = encode_program_lossless(source, language)
    assert encoded[0] == LANGUAGE_GLYPHS[language]
    assert decode_program_lossless(encoded) == source
    assert "".join(tokenize_program(source, language)) == source


def test_every_canonical_syntax_meaning_has_one_unique_character():
    glyphs = list(SYNTAX_GLYPHS.values())
    assert all(len(glyph) == 1 for glyph in glyphs)
    assert len(glyphs) == len(set(glyphs))


def test_fixed_program_glyph_inventory_has_no_collisions():
    rows = list(all_fixed_program_glyphs())
    glyphs = [glyph for _, glyph, _ in rows]
    tokens = [token for token, _, _ in rows]
    assert len(glyphs) == len(set(glyphs))
    assert len(tokens) == len(set(tokens))


def test_canonical_form_normalizes_surface_identifiers():
    first = canonicalize_program("left = 3", "python")
    second = canonicalize_program("right = 7", "python")
    assert first == second
    assert "≔" in first
    assert "№" in first


def test_aligned_training_rows_cover_every_language_and_intent():
    rows = list(iter_program_training_rows(repeats=1))
    expected = len(LANGUAGE_GLYPHS) * len(PROGRAM_EXAMPLES)
    assert len(rows) == expected
    assert {row["programming_language"] for row in rows} == set(LANGUAGE_GLYPHS)
    for row in rows:
        assert row["tokens"]
        assert row["canonical_glyphs"].startswith("⌘")


def test_program_training_rows_are_unique_across_repeat_variants():
    rows = list(iter_program_training_rows(repeats=2))
    assert len(rows) == len({row["text"] for row in rows})


def test_executable_system_inventory_has_fifty_unique_glyphs():
    assert len(EXECUTABLE_SYSTEMS) == 50
    glyphs = [system.glyph for system in EXECUTABLE_SYSTEMS]
    assert len(glyphs) == len(set(glyphs))
    assert all(len(glyph) == 1 for glyph in glyphs)


@pytest.mark.parametrize("system", EXECUTABLE_SYSTEMS, ids=lambda system: system.system_id.lower())
def test_executable_systems_run_to_expected_stdout(system):
    namespace = {"__name__": "__main__"}
    output = io.StringIO()
    with redirect_stdout(output):
        exec(compile(system.source, f"<{system.system_id}>", "exec"), namespace)
    assert output.getvalue() == system.expected_stdout


def test_executable_system_rows_cover_every_system():
    rows = list(iter_executable_system_rows(repeats=1))
    assert len(rows) == len(EXECUTABLE_SYSTEMS)
    assert {row["system_id"] for row in rows} == {system.system_id for system in EXECUTABLE_SYSTEMS}
    for row in rows:
        assert row["language"] == "code:python"
        assert row["tokens"]
        assert row["canonical_glyphs"].startswith("⌘")


def test_executable_system_rows_are_unique_across_repeat_variants():
    rows = list(iter_executable_system_rows(repeats=2))
    assert len(rows) == len({row["text"] for row in rows})


def test_language_aliases_are_deterministic():
    assert normalize_language("c++") == "cpp"
    assert normalize_language("C#") == "csharp"
    assert normalize_language("py") == "python"
    with pytest.raises(ValueError, match="unsupported"):
        normalize_language("braincode")
