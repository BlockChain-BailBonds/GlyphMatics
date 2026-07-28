"""Canonical one-character glyphs for programming-language syntax.

``encode_program_lossless`` preserves exact source: known unambiguous syntax
becomes one glyph and all other lexemes use a UTF-8/Braille literal escape.
``canonicalize_program`` instead collapses identifiers and literals into
semantic classes for cross-language model training.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from textwrap import dedent
from typing import Iterable, Iterator

from .semantic_codec import BRAILLE_BASE, ESCAPE, ESCAPE_END


PROGRAM_SOURCE_TOKEN = "⟪PROGRAM:SOURCE⟫"
PROGRAM_IR_TOKEN = "⟪PROGRAM:IR⟫"

SYNTAX_GLYPHS: dict[str, str] = {
    "PROGRAM": "⌘",
    "IDENTIFIER": "ι",
    "NUMBER": "№",
    "STRING": "§",
    "TRUE": "⊤",
    "FALSE": "⊥",
    "NULL": "∅",
    "DECLARE": "ℓ",
    "CONST": "κ",
    "TYPE": "τ",
    "ASSIGN": "≔",
    "ADD": "⊕",
    "SUBTRACT": "⊖",
    "MULTIPLY": "⊗",
    "DIVIDE": "⊘",
    "MODULO": "％",
    "EQUAL": "≡",
    "NOT_EQUAL": "≠",
    "LESS": "≺",
    "LESS_EQUAL": "≤",
    "GREATER": "≻",
    "GREATER_EQUAL": "≥",
    "AND": "∧",
    "OR": "∨",
    "NOT": "¬",
    "IF": "◇",
    "ELSE": "◆",
    "FOR": "↻",
    "WHILE": "⟳",
    "FUNCTION": "ƒ",
    "RETURN": "↩",
    "CALL": "☏",
    "IMPORT": "⇥",
    "CLASS": "♙",
    "NEW": "✦",
    "TRY": "⚑",
    "CATCH": "⚠",
    "THROW": "↯",
    "ASYNC": "⧖",
    "AWAIT": "⏳",
    "MATCH": "⌑",
    "BREAK": "⏹",
    "CONTINUE": "⏩",
    "IN": "∈",
    "RANGE": "⋯",
    "PAREN_OPEN": "❨",
    "PAREN_CLOSE": "❩",
    "BLOCK_OPEN": "❴",
    "BLOCK_CLOSE": "❵",
    "BRACKET_OPEN": "❲",
    "BRACKET_CLOSE": "❳",
    "COMMA": "‚",
    "STATEMENT_END": "⁏",
    "DOT": "·",
    "COLON": "∶",
    "ARROW": "⇒",
    "COMMENT": "※",
}

LANGUAGE_GLYPHS: dict[str, str] = {
    "python": "Ⓟ",
    "javascript": "Ⓙ",
    "typescript": "Ⓣ",
    "rust": "Ⓡ",
    "go": "Ⓖ",
    "java": "♨",
    "c": "Ⓒ",
    "cpp": "⊞",
    "csharp": "♯",
    "ruby": "♦",
    "php": "ⓗ",
    "swift": "≋",
    "kotlin": "Ⓚ",
    "bash": "Ⓑ",
    "sql": "Ⓢ",
}

COMMENT_STYLES: dict[str, tuple[str, str]] = {
    "python": ("# ", ""),
    "javascript": ("// ", ""),
    "typescript": ("// ", ""),
    "rust": ("// ", ""),
    "go": ("// ", ""),
    "java": ("// ", ""),
    "c": ("// ", ""),
    "cpp": ("// ", ""),
    "csharp": ("// ", ""),
    "ruby": ("# ", ""),
    "php": ("// ", ""),
    "swift": ("// ", ""),
    "kotlin": ("// ", ""),
    "bash": ("# ", ""),
    "sql": ("/* ", " */"),
}

COMMON_SYNTAX: dict[str, str] = {
    "===": "EQUAL", "!==": "NOT_EQUAL", "==": "EQUAL", "!=": "NOT_EQUAL",
    "<=": "LESS_EQUAL", ">=": "GREATER_EQUAL", "&&": "AND", "||": "OR",
    "=>": "ARROW", "->": "ARROW", ":=": "ASSIGN", "=": "ASSIGN",
    "+": "ADD", "-": "SUBTRACT", "*": "MULTIPLY", "/": "DIVIDE",
    "%": "MODULO", "<": "LESS", ">": "GREATER", "!": "NOT",
    "(": "PAREN_OPEN", ")": "PAREN_CLOSE", "{": "BLOCK_OPEN",
    "}": "BLOCK_CLOSE", "[": "BRACKET_OPEN", "]": "BRACKET_CLOSE",
    ",": "COMMA", ";": "STATEMENT_END", ".": "DOT", ":": "COLON",
}

LANGUAGE_KEYWORDS: dict[str, dict[str, str]] = {
    "python": {
        "def": "FUNCTION", "return": "RETURN", "if": "IF", "elif": "ELSE",
        "else": "ELSE", "for": "FOR", "while": "WHILE", "in": "IN",
        "range": "RANGE", "and": "AND", "or": "OR", "not": "NOT",
        "True": "TRUE", "False": "FALSE", "None": "NULL", "class": "CLASS",
        "import": "IMPORT", "from": "IMPORT", "try": "TRY", "except": "CATCH",
        "raise": "THROW", "async": "ASYNC", "await": "AWAIT",
        "break": "BREAK", "continue": "CONTINUE",
    },
    "javascript": {
        "function": "FUNCTION", "return": "RETURN", "if": "IF", "else": "ELSE",
        "for": "FOR", "while": "WHILE", "let": "DECLARE", "var": "DECLARE",
        "const": "CONST", "true": "TRUE", "false": "FALSE", "null": "NULL",
        "class": "CLASS", "import": "IMPORT", "new": "NEW", "try": "TRY",
        "catch": "CATCH", "throw": "THROW", "async": "ASYNC", "await": "AWAIT",
        "break": "BREAK", "continue": "CONTINUE",
    },
    "typescript": {
        "function": "FUNCTION", "return": "RETURN", "if": "IF", "else": "ELSE",
        "for": "FOR", "while": "WHILE", "let": "DECLARE", "var": "DECLARE",
        "const": "CONST", "true": "TRUE", "false": "FALSE", "null": "NULL",
        "class": "CLASS", "import": "IMPORT", "new": "NEW", "try": "TRY",
        "catch": "CATCH", "throw": "THROW", "async": "ASYNC", "await": "AWAIT",
        "interface": "TYPE", "type": "TYPE", "number": "TYPE",
        "string": "TYPE", "boolean": "TYPE", "break": "BREAK",
        "continue": "CONTINUE",
    },
    "rust": {
        "fn": "FUNCTION", "return": "RETURN", "if": "IF", "else": "ELSE",
        "for": "FOR", "while": "WHILE", "loop": "WHILE", "in": "IN",
        "let": "DECLARE", "const": "CONST", "mut": "DECLARE", "true": "TRUE",
        "false": "FALSE", "struct": "CLASS", "use": "IMPORT", "match": "MATCH",
        "async": "ASYNC", "await": "AWAIT", "break": "BREAK",
        "continue": "CONTINUE", "i32": "TYPE", "i64": "TYPE",
        "usize": "TYPE", "String": "TYPE", "bool": "TYPE",
    },
    "go": {
        "func": "FUNCTION", "return": "RETURN", "if": "IF", "else": "ELSE",
        "for": "FOR", "range": "RANGE", "var": "DECLARE", "const": "CONST",
        "true": "TRUE", "false": "FALSE", "nil": "NULL", "type": "TYPE",
        "struct": "CLASS", "import": "IMPORT", "go": "ASYNC",
        "break": "BREAK", "continue": "CONTINUE", "int": "TYPE",
        "string": "TYPE", "bool": "TYPE",
    },
    "java": {
        "return": "RETURN", "if": "IF", "else": "ELSE", "for": "FOR",
        "while": "WHILE", "final": "CONST", "true": "TRUE", "false": "FALSE",
        "null": "NULL", "class": "CLASS", "import": "IMPORT", "new": "NEW",
        "try": "TRY", "catch": "CATCH", "throw": "THROW", "break": "BREAK",
        "continue": "CONTINUE", "int": "TYPE", "long": "TYPE",
        "double": "TYPE", "boolean": "TYPE", "String": "TYPE", "void": "TYPE",
    },
    "c": {
        "return": "RETURN", "if": "IF", "else": "ELSE", "for": "FOR",
        "while": "WHILE", "const": "CONST", "true": "TRUE", "false": "FALSE",
        "NULL": "NULL", "struct": "CLASS", "break": "BREAK",
        "continue": "CONTINUE", "int": "TYPE", "long": "TYPE",
        "double": "TYPE", "float": "TYPE", "char": "TYPE", "void": "TYPE",
    },
    "cpp": {
        "return": "RETURN", "if": "IF", "else": "ELSE", "for": "FOR",
        "while": "WHILE", "const": "CONST", "constexpr": "CONST",
        "true": "TRUE", "false": "FALSE", "nullptr": "NULL", "class": "CLASS",
        "struct": "CLASS", "import": "IMPORT", "new": "NEW", "try": "TRY",
        "catch": "CATCH", "throw": "THROW", "break": "BREAK",
        "continue": "CONTINUE", "int": "TYPE", "long": "TYPE",
        "double": "TYPE", "bool": "TYPE", "void": "TYPE", "auto": "TYPE",
    },
    "csharp": {
        "return": "RETURN", "if": "IF", "else": "ELSE", "for": "FOR",
        "foreach": "FOR", "while": "WHILE", "in": "IN", "var": "DECLARE",
        "const": "CONST", "true": "TRUE", "false": "FALSE", "null": "NULL",
        "class": "CLASS", "using": "IMPORT", "new": "NEW", "try": "TRY",
        "catch": "CATCH", "throw": "THROW", "async": "ASYNC", "await": "AWAIT",
        "break": "BREAK", "continue": "CONTINUE", "int": "TYPE",
        "string": "TYPE", "bool": "TYPE", "void": "TYPE",
    },
    "ruby": {
        "def": "FUNCTION", "end": "BLOCK_CLOSE", "return": "RETURN",
        "if": "IF", "elsif": "ELSE", "else": "ELSE", "for": "FOR",
        "while": "WHILE", "in": "IN", "true": "TRUE", "false": "FALSE",
        "nil": "NULL", "class": "CLASS", "require": "IMPORT", "begin": "TRY",
        "rescue": "CATCH", "raise": "THROW", "break": "BREAK",
        "next": "CONTINUE", "and": "AND", "or": "OR", "not": "NOT",
    },
    "php": {
        "function": "FUNCTION", "return": "RETURN", "if": "IF",
        "elseif": "ELSE", "else": "ELSE", "for": "FOR", "foreach": "FOR",
        "while": "WHILE", "as": "IN", "true": "TRUE", "false": "FALSE",
        "null": "NULL", "class": "CLASS", "require": "IMPORT",
        "include": "IMPORT", "new": "NEW", "try": "TRY", "catch": "CATCH",
        "throw": "THROW", "break": "BREAK", "continue": "CONTINUE",
    },
    "swift": {
        "func": "FUNCTION", "return": "RETURN", "if": "IF", "else": "ELSE",
        "for": "FOR", "while": "WHILE", "in": "IN", "var": "DECLARE",
        "let": "CONST", "true": "TRUE", "false": "FALSE", "nil": "NULL",
        "class": "CLASS", "struct": "CLASS", "import": "IMPORT", "try": "TRY",
        "catch": "CATCH", "throw": "THROW", "async": "ASYNC", "await": "AWAIT",
        "break": "BREAK", "continue": "CONTINUE", "Int": "TYPE",
        "String": "TYPE", "Bool": "TYPE",
    },
    "kotlin": {
        "fun": "FUNCTION", "return": "RETURN", "if": "IF", "else": "ELSE",
        "for": "FOR", "while": "WHILE", "in": "IN", "var": "DECLARE",
        "val": "CONST", "true": "TRUE", "false": "FALSE", "null": "NULL",
        "class": "CLASS", "import": "IMPORT", "try": "TRY", "catch": "CATCH",
        "throw": "THROW", "suspend": "ASYNC", "break": "BREAK",
        "continue": "CONTINUE", "Int": "TYPE", "String": "TYPE",
        "Boolean": "TYPE", "Unit": "TYPE",
    },
    "bash": {
        "if": "IF", "then": "BLOCK_OPEN", "elif": "ELSE", "else": "ELSE",
        "fi": "BLOCK_CLOSE", "for": "FOR", "while": "WHILE", "in": "IN",
        "do": "BLOCK_OPEN", "done": "BLOCK_CLOSE", "function": "FUNCTION",
        "return": "RETURN", "true": "TRUE", "false": "FALSE",
        "source": "IMPORT", "break": "BREAK", "continue": "CONTINUE",
    },
    "sql": {
        "SELECT": "RETURN", "FROM": "IN", "WHERE": "IF", "JOIN": "CALL",
        "INSERT": "DECLARE", "UPDATE": "ASSIGN", "DELETE": "THROW",
        "CREATE": "NEW", "TABLE": "CLASS", "AS": "ASSIGN", "AND": "AND",
        "OR": "OR", "NOT": "NOT", "NULL": "NULL", "TRUE": "TRUE",
        "FALSE": "FALSE", "CASE": "MATCH", "WHEN": "IF", "ELSE": "ELSE",
        "END": "BLOCK_CLOSE",
    },
}


def syntax_token(concept: str) -> str:
    if concept not in SYNTAX_GLYPHS:
        raise KeyError(f"unknown syntax concept: {concept}")
    return f"⟪SYN:{concept}⟫"


def normalize_language(language: str) -> str:
    aliases = {
        "py": "python", "js": "javascript", "ts": "typescript", "rs": "rust",
        "golang": "go", "c++": "cpp", "cc": "cpp", "cs": "csharp",
        "c#": "csharp", "rb": "ruby", "sh": "bash", "shell": "bash",
    }
    normalized = aliases.get(language.strip().lower(), language.strip().lower())
    if normalized not in LANGUAGE_KEYWORDS:
        supported = ", ".join(sorted(LANGUAGE_KEYWORDS))
        raise ValueError(f"unsupported programming language {language!r}; choose: {supported}")
    return normalized


def programming_language_token(language: str) -> str:
    return f"⟪CODE:{normalize_language(language)}⟫"


def annotate_program_source(source: str, language: str, *, variant: int, label: str) -> str:
    normalized = normalize_language(language)
    prefix, suffix = COMMENT_STYLES[normalized]
    note = f"{prefix}glyphmatics {normalized} {label} variant {variant:03d}{suffix}"
    separator = "" if note.endswith("\n") else "\n"
    return f"{note}{separator}{source}"


def language_lexicon(language: str) -> dict[str, str]:
    normalized = normalize_language(language)
    lexicon = dict(COMMON_SYNTAX)
    lexicon.update(LANGUAGE_KEYWORDS[normalized])
    if normalized == "sql":
        lexicon.update({surface.lower(): concept for surface, concept in LANGUAGE_KEYWORDS["sql"].items()})
    return lexicon


def tokenize_program(source: str, language: str) -> list[str]:
    """Tokenize source exactly, including comments, strings, and whitespace."""

    normalize_language(language)
    tokens: list[str] = []
    index = 0
    while index < len(source):
        char = source[index]
        if char.isspace():
            end = index + 1
            while end < len(source) and source[end].isspace():
                end += 1
            tokens.append(source[index:end])
            index = end
            continue
        if source.startswith("//", index) or source.startswith("#", index):
            end = source.find("\n", index)
            end = len(source) if end < 0 else end
            tokens.append(source[index:end])
            index = end
            continue
        if source.startswith("/*", index):
            end = source.find("*/", index + 2)
            end = len(source) if end < 0 else end + 2
            tokens.append(source[index:end])
            index = end
            continue
        if char in {"'", '"', "`"}:
            quote = char
            end = index + 1
            while end < len(source):
                if source[end] == "\\":
                    end += 2
                    continue
                end += 1
                if source[end - 1] == quote:
                    break
            tokens.append(source[index:end])
            index = end
            continue
        if char.isalpha() or char == "_" or ord(char) >= 128:
            end = index + 1
            while end < len(source) and (
                source[end].isalnum() or source[end] == "_" or ord(source[end]) >= 128
            ):
                end += 1
            tokens.append(source[index:end])
            index = end
            continue
        if char.isdigit():
            end = index + 1
            while end < len(source) and (source[end].isalnum() or source[end] in {"_", "."}):
                end += 1
            tokens.append(source[index:end])
            index = end
            continue
        matched = next(
            (
                operator
                for operator in sorted(COMMON_SYNTAX, key=len, reverse=True)
                if source.startswith(operator, index)
            ),
            None,
        )
        if matched is not None:
            tokens.append(matched)
            index += len(matched)
            continue
        tokens.append(char)
        index += 1
    assert "".join(tokens) == source
    return tokens


def _literal_glyphs(token: str) -> str:
    return ESCAPE + "".join(chr(BRAILLE_BASE + byte) for byte in token.encode("utf-8")) + ESCAPE_END


def _decode_literal(glyphs: str, index: int) -> tuple[str, int]:
    raw = bytearray()
    index += 1
    while index < len(glyphs) and glyphs[index] != ESCAPE_END:
        codepoint = ord(glyphs[index])
        if not BRAILLE_BASE <= codepoint <= BRAILLE_BASE + 255:
            raise ValueError("program literal escape contains a non-Braille byte")
        raw.append(codepoint - BRAILLE_BASE)
        index += 1
    if index >= len(glyphs):
        raise ValueError("unterminated program literal escape")
    return bytes(raw).decode("utf-8"), index + 1


def _unambiguous_surfaces(lexicon: dict[str, str]) -> set[str]:
    counts = Counter(lexicon.values())
    return {surface for surface, concept in lexicon.items() if counts[concept] == 1}


def encode_program_lossless(source: str, language: str) -> str:
    normalized = normalize_language(language)
    lexicon = language_lexicon(normalized)
    safe = _unambiguous_surfaces(lexicon)
    output = [LANGUAGE_GLYPHS[normalized]]
    for token in tokenize_program(source, normalized):
        concept = lexicon.get(token)
        output.append(SYNTAX_GLYPHS[concept] if concept and token in safe else _literal_glyphs(token))
    return "".join(output)


def decode_program_lossless(glyphs: str, language: str | None = None) -> str:
    if not glyphs:
        raise ValueError("program glyph stream is empty")
    glyph_to_language = {glyph: name for name, glyph in LANGUAGE_GLYPHS.items()}
    embedded_language = glyph_to_language.get(glyphs[0])
    if embedded_language is None:
        raise ValueError("program glyph stream has no valid language sigil")
    if language is not None and normalize_language(language) != embedded_language:
        raise ValueError("program language sigil does not match requested language")
    lexicon = language_lexicon(embedded_language)
    safe = _unambiguous_surfaces(lexicon)
    reverse = {
        SYNTAX_GLYPHS[concept]: surface
        for surface, concept in lexicon.items()
        if surface in safe
    }
    output: list[str] = []
    index = 1
    while index < len(glyphs):
        glyph = glyphs[index]
        if glyph == ESCAPE:
            literal, index = _decode_literal(glyphs, index)
            output.append(literal)
            continue
        surface = reverse.get(glyph)
        if surface is None:
            raise ValueError(f"glyph U+{ord(glyph):04X} is invalid for {embedded_language}")
        output.append(surface)
        index += 1
    return "".join(output)


def _lexeme_concept(token: str, lexicon: dict[str, str]) -> str | None:
    known = lexicon.get(token)
    if known:
        return known
    if not token or token.isspace():
        return None
    if token.startswith(("//", "#", "/*")):
        return "COMMENT"
    if token[0] in {"'", '"', "`"}:
        return "STRING"
    if token[0].isdigit():
        return "NUMBER"
    if token[0].isalpha() or token[0] in {"_", "$"} or ord(token[0]) >= 128:
        return "IDENTIFIER"
    return None


def canonical_program_tokens(source: str, language: str) -> list[str]:
    normalized = normalize_language(language)
    lexicon = language_lexicon(normalized)
    concepts = [
        concept
        for token in tokenize_program(source, normalized)
        if (concept := _lexeme_concept(token, lexicon)) is not None
    ]
    return [syntax_token("PROGRAM"), *(syntax_token(item) for item in concepts)]


def canonicalize_program(source: str, language: str) -> str:
    prefix = "⟪SYN:"
    return "".join(
        SYNTAX_GLYPHS[token[len(prefix):-1]]
        for token in canonical_program_tokens(source, language)
    )


@dataclass(frozen=True)
class ProgramExample:
    intent: str
    concepts: tuple[str, ...]
    sources: dict[str, str]

    @property
    def canonical_tokens(self) -> list[str]:
        return [syntax_token("PROGRAM"), *(syntax_token(item) for item in self.concepts)]

    @property
    def canonical_glyphs(self) -> str:
        return "".join(SYNTAX_GLYPHS[item] for item in ("PROGRAM", *self.concepts))


def system_token(system_id: str) -> str:
    return f"⟪SYS:{system_id}⟫"


def _script(source: str) -> str:
    return dedent(source).strip() + "\n"


@dataclass(frozen=True)
class ExecutableSystem:
    system_id: str
    glyph: str
    title: str
    description: str
    source: str
    expected_stdout: str

    @property
    def token(self) -> str:
        return system_token(self.system_id)

    @property
    def canonical_tokens(self) -> list[str]:
        return canonical_program_tokens(self.source, "python")

    @property
    def canonical_glyphs(self) -> str:
        return canonicalize_program(self.source, "python")


PROGRAM_EXAMPLES: tuple[ProgramExample, ...] = (
    ProgramExample(
        "assign-number",
        ("DECLARE", "IDENTIFIER", "ASSIGN", "NUMBER"),
        {
            "python": "count = 3", "javascript": "let count = 3;",
            "typescript": "let count: number = 3;", "rust": "let count = 3;",
            "go": "count := 3", "java": "int count = 3;", "c": "int count = 3;",
            "cpp": "int count = 3;", "csharp": "int count = 3;",
            "ruby": "count = 3", "php": "$count = 3;", "swift": "var count = 3",
            "kotlin": "var count = 3", "bash": "count=3",
            "sql": "UPDATE counters SET count = 3;",
        },
    ),
    ProgramExample(
        "add-values",
        ("DECLARE", "IDENTIFIER", "ASSIGN", "IDENTIFIER", "ADD", "IDENTIFIER"),
        {
            "python": "total = left + right", "javascript": "const total = left + right;",
            "typescript": "const total: number = left + right;",
            "rust": "let total = left + right;", "go": "total := left + right",
            "java": "int total = left + right;", "c": "int total = left + right;",
            "cpp": "auto total = left + right;", "csharp": "var total = left + right;",
            "ruby": "total = left + right", "php": "$total = $left + $right;",
            "swift": "let total = left + right", "kotlin": "val total = left + right",
            "bash": "total=$((left + right))", "sql": "SELECT left + right AS total;",
        },
    ),
    ProgramExample(
        "greater-conditional-call",
        ("IF", "IDENTIFIER", "GREATER", "NUMBER", "CALL", "IDENTIFIER"),
        {
            "python": "if score > 80:\n    celebrate()",
            "javascript": "if (score > 80) { celebrate(); }",
            "typescript": "if (score > 80) { celebrate(); }",
            "rust": "if score > 80 { celebrate(); }",
            "go": "if score > 80 { celebrate() }",
            "java": "if (score > 80) { celebrate(); }",
            "c": "if (score > 80) { celebrate(); }",
            "cpp": "if (score > 80) { celebrate(); }",
            "csharp": "if (score > 80) { Celebrate(); }",
            "ruby": "if score > 80\n  celebrate\nend",
            "php": "if ($score > 80) { celebrate(); }",
            "swift": "if score > 80 { celebrate() }",
            "kotlin": "if (score > 80) { celebrate() }",
            "bash": "if (( score > 80 )); then celebrate; fi",
            "sql": "SELECT CASE WHEN score > 80 THEN celebrate END;",
        },
    ),
    ProgramExample(
        "while-increment",
        ("WHILE", "IDENTIFIER", "LESS", "NUMBER", "ASSIGN", "IDENTIFIER", "ADD", "NUMBER"),
        {
            "python": "while index < 10:\n    index = index + 1",
            "javascript": "while (index < 10) { index = index + 1; }",
            "typescript": "while (index < 10) { index = index + 1; }",
            "rust": "while index < 10 { index = index + 1; }",
            "go": "for index < 10 { index = index + 1 }",
            "java": "while (index < 10) { index = index + 1; }",
            "c": "while (index < 10) { index = index + 1; }",
            "cpp": "while (index < 10) { index = index + 1; }",
            "csharp": "while (index < 10) { index = index + 1; }",
            "ruby": "while index < 10\n  index = index + 1\nend",
            "php": "while ($index < 10) { $index = $index + 1; }",
            "swift": "while index < 10 { index = index + 1 }",
            "kotlin": "while (index < 10) { index = index + 1 }",
            "bash": "while (( index < 10 )); do index=$((index + 1)); done",
            "sql": "WHILE index < 10 UPDATE state SET index = index + 1;",
        },
    ),
    ProgramExample(
        "return-sum-function",
        ("FUNCTION", "IDENTIFIER", "IDENTIFIER", "IDENTIFIER", "RETURN", "IDENTIFIER", "ADD", "IDENTIFIER"),
        {
            "python": "def add(left, right):\n    return left + right",
            "javascript": "function add(left, right) { return left + right; }",
            "typescript": "function add(left: number, right: number): number { return left + right; }",
            "rust": "fn add(left: i32, right: i32) -> i32 { return left + right; }",
            "go": "func add(left int, right int) int { return left + right }",
            "java": "int add(int left, int right) { return left + right; }",
            "c": "int add(int left, int right) { return left + right; }",
            "cpp": "int add(int left, int right) { return left + right; }",
            "csharp": "int Add(int left, int right) { return left + right; }",
            "ruby": "def add(left, right)\n  return left + right\nend",
            "php": "function add($left, $right) { return $left + $right; }",
            "swift": "func add(_ left: Int, _ right: Int) -> Int { return left + right }",
            "kotlin": "fun add(left: Int, right: Int): Int { return left + right }",
            "bash": "add() { echo $(($1 + $2)); }",
            "sql": "CREATE FUNCTION add(left INT, right INT) RETURNS INT RETURN left + right;",
        },
    ),
    ProgramExample(
        "boolean-conjunction",
        ("DECLARE", "IDENTIFIER", "ASSIGN", "IDENTIFIER", "AND", "NOT", "IDENTIFIER"),
        {
            "python": "ready = online and not busy",
            "javascript": "const ready = online && !busy;",
            "typescript": "const ready: boolean = online && !busy;",
            "rust": "let ready = online && !busy;", "go": "ready := online && !busy",
            "java": "boolean ready = online && !busy;",
            "c": "bool ready = online && !busy;",
            "cpp": "bool ready = online && !busy;",
            "csharp": "bool ready = online && !busy;",
            "ruby": "ready = online && !busy", "php": "$ready = $online && !$busy;",
            "swift": "let ready = online && !busy",
            "kotlin": "val ready = online && !busy",
            "bash": "ready=$(( online && !busy ))",
            "sql": "SELECT online AND NOT busy AS ready;",
        },
    ),
)


EXECUTABLE_SYSTEMS: tuple[ExecutableSystem, ...] = (
    ExecutableSystem(
        "HELLO_WORLD",
        "①",
        "Hello World",
        "Print a fixed startup greeting.",
        _script(
            """
            def main():
                print("hello glyph")


            if __name__ == "__main__":
                main()
            """
        ),
        "hello glyph\n",
    ),
    ExecutableSystem(
        "ECHO_PIPELINE",
        "②",
        "Echo Pipeline",
        "Normalize words and join them into one output string.",
        _script(
            """
            def main():
                words = ["sigil", "agi"]
                print("-".join(word.upper() for word in words))


            if __name__ == "__main__":
                main()
            """
        ),
        "SIGIL-AGI\n",
    ),
    ExecutableSystem(
        "SUM_REDUCER",
        "③",
        "Sum Reducer",
        "Aggregate a sequence of integers.",
        _script(
            """
            def main():
                values = [2, 4, 6, 8]
                print(sum(values))


            if __name__ == "__main__":
                main()
            """
        ),
        "20\n",
    ),
    ExecutableSystem(
        "TEMP_CONVERTER",
        "④",
        "Temperature Converter",
        "Convert celsius into fahrenheit.",
        _script(
            """
            def main():
                celsius = 25
                fahrenheit = celsius * 9 / 5 + 32
                print(fahrenheit)


            if __name__ == "__main__":
                main()
            """
        ),
        "77.0\n",
    ),
    ExecutableSystem(
        "FIBONACCI_SERIES",
        "⑤",
        "Fibonacci Series",
        "Generate the first seven Fibonacci numbers.",
        _script(
            """
            def main():
                sequence = [0, 1]
                while len(sequence) < 7:
                    sequence.append(sequence[-1] + sequence[-2])
                print(",".join(str(item) for item in sequence))


            if __name__ == "__main__":
                main()
            """
        ),
        "0,1,1,2,3,5,8\n",
    ),
    ExecutableSystem(
        "FACTORIAL_ENGINE",
        "⑥",
        "Factorial Engine",
        "Compute a factorial iteratively.",
        _script(
            """
            def factorial(number):
                total = 1
                for value in range(2, number + 1):
                    total *= value
                return total


            def main():
                print(factorial(6))


            if __name__ == "__main__":
                main()
            """
        ),
        "720\n",
    ),
    ExecutableSystem(
        "PRIME_CHECKER",
        "⑦",
        "Prime Checker",
        "Determine whether a value is prime.",
        _script(
            """
            def is_prime(number):
                if number < 2:
                    return False
                for value in range(2, int(number ** 0.5) + 1):
                    if number % value == 0:
                        return False
                return True


            def main():
                print(is_prime(29))


            if __name__ == "__main__":
                main()
            """
        ),
        "True\n",
    ),
    ExecutableSystem(
        "PALINDROME_CHECKER",
        "⑧",
        "Palindrome Checker",
        "Validate mirrored text.",
        _script(
            """
            def main():
                text = "racecar"
                print(text == text[::-1])


            if __name__ == "__main__":
                main()
            """
        ),
        "True\n",
    ),
    ExecutableSystem(
        "WORD_FREQUENCY",
        "⑨",
        "Word Frequency",
        "Count repeated tokens in a sentence.",
        _script(
            """
            from collections import Counter


            def main():
                counts = Counter("glyph glyph sigil agi glyph".split())
                ordered = [f"{word}={counts[word]}" for word in sorted(counts)]
                print(";".join(ordered))


            if __name__ == "__main__":
                main()
            """
        ),
        "agi=1;glyph=3;sigil=1\n",
    ),
    ExecutableSystem(
        "LINE_COUNTER",
        "⑩",
        "Line Counter",
        "Count newline-delimited records.",
        _script(
            """
            def main():
                payload = "alpha\\nbeta\\ngamma\\n"
                print(len(payload.splitlines()))


            if __name__ == "__main__":
                main()
            """
        ),
        "3\n",
    ),
    ExecutableSystem(
        "CSV_TOTALER",
        "⑪",
        "CSV Totaler",
        "Parse CSV rows and total a numeric column.",
        _script(
            """
            import csv
            import io


            def main():
                handle = io.StringIO("item,qty\\nalpha,3\\nbeta,4\\ngamma,5\\n")
                rows = csv.DictReader(handle)
                print(sum(int(row["qty"]) for row in rows))


            if __name__ == "__main__":
                main()
            """
        ),
        "12\n",
    ),
    ExecutableSystem(
        "JSON_CONFIG_LOADER",
        "⑫",
        "JSON Config Loader",
        "Parse a JSON blob and read one configuration value.",
        _script(
            """
            import json


            def main():
                config = json.loads('{"port": 918, "mode": "local"}')
                print(config["port"])


            if __name__ == "__main__":
                main()
            """
        ),
        "918\n",
    ),
    ExecutableSystem(
        "ENV_RESOLVER",
        "⑬",
        "Environment Resolver",
        "Resolve a setting from an environment-like mapping.",
        _script(
            """
            def main():
                env = {"MODE": "offline", "DEBUG": "0"}
                print(env.get("MODE", "online"))


            if __name__ == "__main__":
                main()
            """
        ),
        "offline\n",
    ),
    ExecutableSystem(
        "FILE_COPY_WORKFLOW",
        "⑭",
        "File Copy Workflow",
        "Write, copy, and read a file through a temporary workspace.",
        _script(
            """
            import pathlib
            import shutil
            import tempfile


            def main():
                with tempfile.TemporaryDirectory() as directory:
                    root = pathlib.Path(directory)
                    source = root / "source.txt"
                    target = root / "target.txt"
                    source.write_text("copy-ok", encoding="utf-8")
                    shutil.copy2(source, target)
                    print(target.read_text(encoding="utf-8"))


            if __name__ == "__main__":
                main()
            """
        ),
        "copy-ok\n",
    ),
    ExecutableSystem(
        "DIRECTORY_LISTER",
        "⑮",
        "Directory Lister",
        "List files from a generated directory in sorted order.",
        _script(
            """
            import pathlib
            import tempfile


            def main():
                with tempfile.TemporaryDirectory() as directory:
                    root = pathlib.Path(directory)
                    for name in ("b.txt", "a.txt"):
                        (root / name).write_text(name, encoding="utf-8")
                    print(",".join(sorted(path.name for path in root.iterdir())))


            if __name__ == "__main__":
                main()
            """
        ),
        "a.txt,b.txt\n",
    ),
    ExecutableSystem(
        "LOG_FILTER",
        "⑯",
        "Log Filter",
        "Count error-level log lines.",
        _script(
            """
            def main():
                lines = [
                    "INFO boot",
                    "ERROR disk",
                    "WARN retry",
                    "ERROR timeout",
                ]
                print(sum(line.startswith("ERROR") for line in lines))


            if __name__ == "__main__":
                main()
            """
        ),
        "2\n",
    ),
    ExecutableSystem(
        "REGEX_EXTRACTOR",
        "⑰",
        "Regex Extractor",
        "Extract all numeric identifiers from text.",
        _script(
            """
            import re


            def main():
                text = "jobs 18, 42, and 105 completed"
                print(",".join(re.findall(r"\\d+", text)))


            if __name__ == "__main__":
                main()
            """
        ),
        "18,42,105\n",
    ),
    ExecutableSystem(
        "TEMPLATE_RENDERER",
        "⑱",
        "Template Renderer",
        "Render a greeting from named placeholders.",
        _script(
            """
            def main():
                template = "Hello, {name}"
                print(template.format(name="GlyphMatics"))


            if __name__ == "__main__":
                main()
            """
        ),
        "Hello, GlyphMatics\n",
    ),
    ExecutableSystem(
        "QUERYSTRING_BUILDER",
        "⑲",
        "Querystring Builder",
        "Encode URL query parameters.",
        _script(
            """
            from urllib.parse import urlencode


            def main():
                print(urlencode({"mode": "test", "user": "nine"}))


            if __name__ == "__main__":
                main()
            """
        ),
        "mode=test&user=nine\n",
    ),
    ExecutableSystem(
        "STOPWATCH_DELTA",
        "⑳",
        "Stopwatch Delta",
        "Compute an elapsed duration from fixed timestamps.",
        _script(
            """
            def main():
                started = 10.5
                ended = 13.25
                print(round(ended - started, 2))


            if __name__ == "__main__":
                main()
            """
        ),
        "2.75\n",
    ),
    ExecutableSystem(
        "RETRY_BACKOFF",
        "㉑",
        "Retry Backoff",
        "Retry until a simulated task succeeds.",
        _script(
            """
            def main():
                attempts = 0
                while True:
                    attempts += 1
                    if attempts >= 3:
                        print(attempts)
                        return


            if __name__ == "__main__":
                main()
            """
        ),
        "3\n",
    ),
    ExecutableSystem(
        "QUEUE_WORKER",
        "㉒",
        "Queue Worker",
        "Drain a FIFO queue.",
        _script(
            """
            from collections import deque


            def main():
                queue = deque([1, 2, 3])
                output = []
                while queue:
                    output.append(str(queue.popleft()))
                print(",".join(output))


            if __name__ == "__main__":
                main()
            """
        ),
        "1,2,3\n",
    ),
    ExecutableSystem(
        "SCHEDULER_TICKS",
        "㉓",
        "Scheduler Ticks",
        "Run scheduled jobs in time order.",
        _script(
            """
            import heapq


            def main():
                jobs = [(3, "ship"), (1, "build"), (2, "test")]
                heapq.heapify(jobs)
                output = []
                while jobs:
                    _, name = heapq.heappop(jobs)
                    output.append(name)
                print(",".join(output))


            if __name__ == "__main__":
                main()
            """
        ),
        "build,test,ship\n",
    ),
    ExecutableSystem(
        "TTL_CACHE",
        "㉔",
        "TTL Cache",
        "Serve a cached value until it expires.",
        _script(
            """
            def main():
                cache = {"answer": ("warm", 12)}
                checkpoints = [10, 13]
                output = []
                for current in checkpoints:
                    value, expires = cache["answer"]
                    output.append("hit" if current < expires else "miss")
                print(",".join(output))


            if __name__ == "__main__":
                main()
            """
        ),
        "hit,miss\n",
    ),
    ExecutableSystem(
        "MEMOIZED_FIB",
        "㉕",
        "Memoized Fibonacci",
        "Use memoization to speed up Fibonacci recursion.",
        _script(
            """
            def fib(number, memo):
                if number in memo:
                    return memo[number]
                memo[number] = fib(number - 1, memo) + fib(number - 2, memo)
                return memo[number]


            def main():
                print(fib(8, {0: 0, 1: 1}))


            if __name__ == "__main__":
                main()
            """
        ),
        "21\n",
    ),
    ExecutableSystem(
        "STACK_MACHINE",
        "㉖",
        "Stack Machine",
        "Evaluate a postfix arithmetic expression.",
        _script(
            """
            def main():
                stack = []
                for token in ["3", "4", "5", "*", "+"]:
                    if token.isdigit():
                        stack.append(int(token))
                    else:
                        right = stack.pop()
                        left = stack.pop()
                        stack.append(left + right if token == "+" else left * right)
                print(stack[-1])


            if __name__ == "__main__":
                main()
            """
        ),
        "23\n",
    ),
    ExecutableSystem(
        "QUEUE_CLASS",
        "㉗",
        "Queue Class",
        "Implement a minimal queue abstraction.",
        _script(
            """
            class Queue:
                def __init__(self):
                    self.items = []

                def push(self, value):
                    self.items.append(value)

                def pop(self):
                    return self.items.pop(0)


            def main():
                queue = Queue()
                queue.push("alpha")
                queue.push("beta")
                print(",".join([queue.pop(), queue.pop()]))


            if __name__ == "__main__":
                main()
            """
        ),
        "alpha,beta\n",
    ),
    ExecutableSystem(
        "BINARY_SEARCH",
        "㉘",
        "Binary Search",
        "Locate a target inside a sorted array.",
        _script(
            """
            def search(values, target):
                low = 0
                high = len(values) - 1
                while low <= high:
                    mid = (low + high) // 2
                    if values[mid] == target:
                        return mid
                    if values[mid] < target:
                        low = mid + 1
                    else:
                        high = mid - 1
                return -1


            def main():
                print(search([1, 3, 5, 7, 9, 11], 7))


            if __name__ == "__main__":
                main()
            """
        ),
        "3\n",
    ),
    ExecutableSystem(
        "QUICKSORT",
        "㉙",
        "Quicksort",
        "Sort an unsorted list using recursive partitioning.",
        _script(
            """
            def quicksort(values):
                if len(values) < 2:
                    return values
                pivot = values[0]
                lower = [value for value in values[1:] if value <= pivot]
                higher = [value for value in values[1:] if value > pivot]
                return quicksort(lower) + [pivot] + quicksort(higher)


            def main():
                print(",".join(str(item) for item in quicksort([5, 1, 4, 2, 3])))


            if __name__ == "__main__":
                main()
            """
        ),
        "1,2,3,4,5\n",
    ),
    ExecutableSystem(
        "MERGE_DEDUP",
        "㉚",
        "Merge Dedup",
        "Merge two lists and remove duplicates.",
        _script(
            """
            def main():
                values = sorted(set([1, 2, 5] + [2, 3, 8]))
                print(",".join(str(item) for item in values))


            if __name__ == "__main__":
                main()
            """
        ),
        "1,2,3,5,8\n",
    ),
    ExecutableSystem(
        "BFS_PATH",
        "㉛",
        "BFS Path",
        "Compute the shortest path in an unweighted graph.",
        _script(
            """
            from collections import deque


            def main():
                graph = {
                    "A": ["B", "C"],
                    "B": ["D"],
                    "C": ["D"],
                    "D": [],
                }
                queue = deque([("A", ["A"])])
                seen = {"A"}
                while queue:
                    node, path = queue.popleft()
                    if node == "D":
                        print("->".join(path))
                        return
                    for neighbor in graph[node]:
                        if neighbor not in seen:
                            seen.add(neighbor)
                            queue.append((neighbor, [*path, neighbor]))


            if __name__ == "__main__":
                main()
            """
        ),
        "A->B->D\n",
    ),
    ExecutableSystem(
        "DFS_ORDER",
        "㉜",
        "DFS Order",
        "Traverse a graph depth first.",
        _script(
            """
            def dfs(graph, node, seen, order):
                seen.add(node)
                order.append(node)
                for neighbor in graph[node]:
                    if neighbor not in seen:
                        dfs(graph, neighbor, seen, order)


            def main():
                graph = {
                    "A": ["B", "C"],
                    "B": ["D"],
                    "C": [],
                    "D": [],
                }
                order = []
                dfs(graph, "A", set(), order)
                print(",".join(order))


            if __name__ == "__main__":
                main()
            """
        ),
        "A,B,D,C\n",
    ),
    ExecutableSystem(
        "DIJKSTRA_ROUTE",
        "㉝",
        "Dijkstra Route",
        "Compute a weighted shortest path distance.",
        _script(
            """
            import heapq


            def main():
                graph = {
                    "A": [("B", 2), ("C", 5)],
                    "B": [("D", 2)],
                    "C": [("D", 1)],
                    "D": [],
                }
                heap = [(0, "A")]
                seen = {}
                while heap:
                    cost, node = heapq.heappop(heap)
                    if node in seen:
                        continue
                    seen[node] = cost
                    if node == "D":
                        print(cost)
                        return
                    for neighbor, weight in graph[node]:
                        heapq.heappush(heap, (cost + weight, neighbor))


            if __name__ == "__main__":
                main()
            """
        ),
        "4\n",
    ),
    ExecutableSystem(
        "TOPOLOGICAL_SORT",
        "㉞",
        "Topological Sort",
        "Order tasks by dependency.",
        _script(
            """
            from collections import defaultdict, deque


            def main():
                edges = [("plan", "code"), ("code", "test"), ("test", "ship")]
                graph = defaultdict(list)
                indegree = defaultdict(int)
                for start, end in edges:
                    graph[start].append(end)
                    indegree[end] += 1
                    indegree.setdefault(start, 0)
                queue = deque(node for node, degree in indegree.items() if degree == 0)
                order = []
                while queue:
                    node = queue.popleft()
                    order.append(node)
                    for neighbor in graph[node]:
                        indegree[neighbor] -= 1
                        if indegree[neighbor] == 0:
                            queue.append(neighbor)
                print(",".join(order))


            if __name__ == "__main__":
                main()
            """
        ),
        "plan,code,test,ship\n",
    ),
    ExecutableSystem(
        "LRU_CACHE",
        "㉟",
        "LRU Cache",
        "Evict the least recently used key when capacity is exceeded.",
        _script(
            """
            from collections import OrderedDict


            def main():
                cache = OrderedDict()
                for key, value in [("a", 1), ("b", 2)]:
                    cache[key] = value
                cache.move_to_end("a")
                cache["c"] = 3
                cache.popitem(last=False)
                print(",".join(cache.keys()))


            if __name__ == "__main__":
                main()
            """
        ),
        "a,c\n",
    ),
    ExecutableSystem(
        "EVENT_BUS",
        "㊱",
        "Event Bus",
        "Dispatch an event to multiple subscribers.",
        _script(
            """
            def main():
                handlers = []
                count = {"value": 0}

                def subscribe(handler):
                    handlers.append(handler)

                def publish(payload):
                    for handler in handlers:
                        handler(payload)

                subscribe(lambda payload: count.__setitem__("value", count["value"] + payload))
                subscribe(lambda payload: count.__setitem__("value", count["value"] + payload))
                publish(1)
                print(count["value"])


            if __name__ == "__main__":
                main()
            """
        ),
        "2\n",
    ),
    ExecutableSystem(
        "STATE_MACHINE",
        "㊲",
        "State Machine",
        "Advance through a simple transition table.",
        _script(
            """
            def main():
                state = "idle"
                transitions = {
                    ("idle", "start"): "running",
                    ("running", "stop"): "stopped",
                }
                for event in ("start", "stop"):
                    state = transitions[(state, event)]
                print(state)


            if __name__ == "__main__":
                main()
            """
        ),
        "stopped\n",
    ),
    ExecutableSystem(
        "PLUGIN_REGISTRY",
        "㊳",
        "Plugin Registry",
        "Register and execute a named plugin.",
        _script(
            """
            def main():
                registry = {}
                registry["double"] = lambda value: value * 2
                print(registry["double"](21))


            if __name__ == "__main__":
                main()
            """
        ),
        "42\n",
    ),
    ExecutableSystem(
        "TOKEN_VALIDATOR",
        "㊴",
        "Token Validator",
        "Validate a token signature with HMAC.",
        _script(
            """
            import hmac
            import hashlib


            def main():
                secret = b"glyph-secret"
                message = b"token"
                digest = hmac.new(secret, message, hashlib.sha256).hexdigest()
                expected = hmac.new(secret, message, hashlib.sha256).hexdigest()
                print(hmac.compare_digest(digest, expected))


            if __name__ == "__main__":
                main()
            """
        ),
        "True\n",
    ),
    ExecutableSystem(
        "BYTE_CHECKSUM",
        "㊵",
        "Byte Checksum",
        "Compute a simple byte-sum checksum.",
        _script(
            """
            def main():
                payload = b"glyph"
                print(sum(payload))


            if __name__ == "__main__":
                main()
            """
        ),
        "548\n",
    ),
    ExecutableSystem(
        "BASE64_CODEC",
        "㊶",
        "Base64 Codec",
        "Encode and decode a payload with Base64.",
        _script(
            """
            import base64


            def main():
                encoded = base64.b64encode(b"glyph")
                print(base64.b64decode(encoded).decode("utf-8"))


            if __name__ == "__main__":
                main()
            """
        ),
        "glyph\n",
    ),
    ExecutableSystem(
        "GZIP_ROUNDTRIP",
        "㊷",
        "Gzip Roundtrip",
        "Compress and restore a text payload.",
        _script(
            """
            import gzip


            def main():
                compressed = gzip.compress(b"sigil")
                print(gzip.decompress(compressed).decode("utf-8"))


            if __name__ == "__main__":
                main()
            """
        ),
        "sigil\n",
    ),
    ExecutableSystem(
        "SQLITE_TODO",
        "㊸",
        "SQLite Todo",
        "Use an in-memory SQLite table to count open tasks.",
        _script(
            """
            import sqlite3


            def main():
                connection = sqlite3.connect(":memory:")
                connection.execute("CREATE TABLE todo (title TEXT, done INTEGER)")
                connection.executemany(
                    "INSERT INTO todo VALUES (?, ?)",
                    [("plan", 0), ("build", 1), ("test", 0)],
                )
                row = connection.execute("SELECT COUNT(*) FROM todo WHERE done = 0").fetchone()
                print(row[0])


            if __name__ == "__main__":
                main()
            """
        ),
        "2\n",
    ),
    ExecutableSystem(
        "BANK_LEDGER",
        "㊹",
        "Bank Ledger",
        "Track account balance through deposits and withdrawals.",
        _script(
            """
            class Account:
                def __init__(self, balance):
                    self.balance = balance

                def deposit(self, amount):
                    self.balance += amount

                def withdraw(self, amount):
                    self.balance -= amount


            def main():
                account = Account(50)
                account.deposit(25)
                account.withdraw(10)
                print(account.balance)


            if __name__ == "__main__":
                main()
            """
        ),
        "65\n",
    ),
    ExecutableSystem(
        "INVENTORY_RESTOCK",
        "㊺",
        "Inventory Restock",
        "Report items below the reorder threshold.",
        _script(
            """
            def main():
                stock = {"mouse": 2, "keyboard": 6, "cable": 1}
                low = sorted(name for name, count in stock.items() if count < 3)
                print(",".join(low))


            if __name__ == "__main__":
                main()
            """
        ),
        "cable,mouse\n",
    ),
    ExecutableSystem(
        "GRADEBOOK_AVERAGE",
        "㊻",
        "Gradebook Average",
        "Compute the average of numeric scores.",
        _script(
            """
            def main():
                grades = [88, 92, 90]
                print(sum(grades) / len(grades))


            if __name__ == "__main__":
                main()
            """
        ),
        "90.0\n",
    ),
    ExecutableSystem(
        "MARKDOWNISH_RENDERER",
        "㊼",
        "Markdownish Renderer",
        "Convert a tiny markdown subset into HTML.",
        _script(
            """
            def render_line(line):
                if line.startswith("# "):
                    return f"<h1>{line[2:]}</h1>"
                return f"<p>{line}</p>"


            def main():
                lines = ["# Glyph", "Ready"]
                print("".join(render_line(line) for line in lines))


            if __name__ == "__main__":
                main()
            """
        ),
        "<h1>Glyph</h1><p>Ready</p>\n",
    ),
    ExecutableSystem(
        "TRIE_PREFIX_SEARCH",
        "㊽",
        "Trie Prefix Search",
        "Return words that match a prefix from a trie-like index.",
        _script(
            """
            def main():
                words = ["glyph", "glide", "signal"]
                prefix = "gl"
                matches = sorted(word for word in words if word.startswith(prefix))
                print(",".join(matches))


            if __name__ == "__main__":
                main()
            """
        ),
        "glide,glyph\n",
    ),
    ExecutableSystem(
        "GRAPH_CYCLE_DETECTOR",
        "㊾",
        "Graph Cycle Detector",
        "Detect a cycle in a directed graph.",
        _script(
            """
            def has_cycle(graph):
                visiting = set()
                visited = set()

                def visit(node):
                    if node in visiting:
                        return True
                    if node in visited:
                        return False
                    visiting.add(node)
                    for neighbor in graph[node]:
                        if visit(neighbor):
                            return True
                    visiting.remove(node)
                    visited.add(node)
                    return False

                return any(visit(node) for node in graph)


            def main():
                graph = {"A": ["B"], "B": ["C"], "C": ["A"]}
                print(has_cycle(graph))


            if __name__ == "__main__":
                main()
            """
        ),
        "True\n",
    ),
    ExecutableSystem(
        "INI_CONFIG_PARSER",
        "㊿",
        "INI Config Parser",
        "Load a setting from an INI document.",
        _script(
            """
            import configparser


            def main():
                config = configparser.ConfigParser()
                config.read_string("[app]\\nport = 918\\n")
                print(config["app"]["port"])


            if __name__ == "__main__":
                main()
            """
        ),
        "918\n",
    ),
)


def iter_program_training_rows(repeats: int = 16) -> Iterator[dict[str, object]]:
    if repeats < 1:
        raise ValueError("program training repeats must be positive")
    for repeat in range(repeats):
        for example in PROGRAM_EXAMPLES:
            for language, source in sorted(example.sources.items()):
                annotated_source = annotate_program_source(
                    source,
                    language,
                    variant=repeat,
                    label=example.intent,
                )
                yield {
                    "text": annotated_source,
                    "language": f"code:{language}",
                    "source": "glyphmatics/programming_syntax",
                    "programming_language": language,
                    "intent": example.intent,
                    "tokens": [
                        programming_language_token(language),
                        PROGRAM_SOURCE_TOKEN,
                        *tokenize_program(annotated_source, language),
                        PROGRAM_IR_TOKEN,
                        *example.canonical_tokens,
                    ],
                    "canonical_tokens": example.canonical_tokens,
                    "canonical_glyphs": example.canonical_glyphs,
                    "repeat": repeat,
                }


def iter_executable_system_rows(repeats: int = 16) -> Iterator[dict[str, object]]:
    if repeats < 1:
        raise ValueError("executable system repeats must be positive")
    for repeat in range(repeats):
        for system in EXECUTABLE_SYSTEMS:
            annotated_source = annotate_program_source(
                system.source,
                "python",
                variant=repeat,
                label=system.system_id.lower(),
            )
            yield {
                "text": annotated_source,
                "language": "code:python",
                "source": "glyphmatics/executable_systems",
                "programming_language": "python",
                "system_id": system.system_id,
                "system_title": system.title,
                "system_glyph": system.glyph,
                "tokens": [
                    programming_language_token("python"),
                    PROGRAM_SOURCE_TOKEN,
                    *tokenize_program(annotated_source, "python"),
                    PROGRAM_IR_TOKEN,
                    system.token,
                    *system.canonical_tokens,
                ],
                "canonical_tokens": system.canonical_tokens,
                "canonical_glyphs": system.canonical_glyphs,
                "repeat": repeat,
            }


def all_fixed_program_glyphs() -> Iterable[tuple[str, str, dict[str, object]]]:
    yield PROGRAM_SOURCE_TOKEN, "◁", {"kind": "program-boundary", "role": "source"}
    yield PROGRAM_IR_TOKEN, "▷", {"kind": "program-boundary", "role": "canonical-ir"}
    for language, glyph in LANGUAGE_GLYPHS.items():
        yield programming_language_token(language), glyph, {
            "kind": "programming-language",
            "language": language,
        }
    for concept, glyph in SYNTAX_GLYPHS.items():
        yield syntax_token(concept), glyph, {
            "kind": "programming-syntax",
            "concept_id": f"SYNTAX.{concept}",
            "language": None,
        }
    for system in EXECUTABLE_SYSTEMS:
        yield system.token, system.glyph, {
            "kind": "executable-system",
            "system_id": system.system_id,
            "title": system.title,
            "description": system.description,
            "language": "python",
            "expected_stdout": system.expected_stdout.rstrip("\n"),
        }
