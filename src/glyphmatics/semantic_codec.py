"""Deterministic multilingual word-to-glyph codec.

The visible form uses one Unicode private-use code point for every vocabulary
token. Unknown text is escaped as UTF-8 bytes represented by Braille code
points, so every input can be decoded exactly.

The binary form uses frequency-ordered variable-length integer IDs and is the
storage-oriented representation. The visible glyph stream is optimized for
model token count and human transport, not for UTF-8 byte size.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import unicodedata
from typing import Any


FORMAT = "glyphmatics-semantic-v1"
BINARY_MAGIC = b"VILC1"
ESCAPE = "\ue000"
ESCAPE_END = "\ue001"
BRAILLE_BASE = 0x2800
SPECIAL_TOKEN_COUNT = 4
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

# The first 256 BMP private-use characters are reserved for framing. Continue
# into supplementary private-use space when the BMP range is exhausted.
GLYPH_RANGES = (
    (0xE100, 0xF8FF),
    (0xF0000, 0xFFFFD),
    (0x100000, 0x10FFFD),
)


def _is_word_character(char: str) -> bool:
    return char == "_" or unicodedata.category(char)[0] in {"L", "M", "N"}


def tokenize_lossless(text: str) -> list[str]:
    """Split text without normalization and preserve every code point.

    Letters, marks, and numbers stay together, including apostrophes inside a
    word. Whitespace runs are tokens, and punctuation/symbols are single tokens.
    Concatenating the returned tokens always recreates *text* exactly.
    """

    tokens: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char.isspace():
            end = index + 1
            while end < len(text) and text[end].isspace():
                end += 1
            tokens.append(text[index:end])
            index = end
            continue
        if _is_word_character(char):
            end = index + 1
            while end < len(text):
                candidate = text[end]
                if _is_word_character(candidate):
                    end += 1
                    continue
                if (
                    candidate in {"'", "’"}
                    and end + 1 < len(text)
                    and _is_word_character(text[end - 1])
                    and _is_word_character(text[end + 1])
                ):
                    end += 1
                    continue
                break
            tokens.append(text[index:end])
            index = end
            continue
        tokens.append(char)
        index += 1
    assert "".join(tokens) == text
    return tokens


def iter_glyphs() -> Iterator[str]:
    for start, end in GLYPH_RANGES:
        for codepoint in range(start, end + 1):
            yield chr(codepoint)


def _uvarint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varint cannot encode a negative value")
    output = bytearray()
    while value >= 0x80:
        output.append((value & 0x7F) | 0x80)
        value >>= 7
    output.append(value)
    return bytes(output)


def _read_uvarint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(data):
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
        if shift > 63:
            raise ValueError("varint is too large")
    raise ValueError("truncated varint")


@dataclass(frozen=True)
class CompressionStats:
    source_characters: int
    source_tokens: int
    known_tokens: int
    glyph_characters: int
    source_utf8_bytes: int
    glyph_utf8_bytes: int
    binary_bytes: int
    binary_header_bytes: int

    def as_dict(self) -> dict[str, int | float]:
        known_ratio = self.known_tokens / max(1, self.source_tokens)
        return {
            **self.__dict__,
            "binary_payload_bytes": self.binary_bytes - self.binary_header_bytes,
            "known_token_ratio": round(known_ratio, 6),
            "visual_character_ratio": round(
                self.source_characters / max(1, self.glyph_characters), 6
            ),
            "glyph_utf8_ratio": round(
                self.source_utf8_bytes / max(1, self.glyph_utf8_bytes), 6
            ),
            "binary_ratio": round(
                self.source_utf8_bytes / max(1, self.binary_bytes), 6
            ),
            "binary_payload_ratio": round(
                self.source_utf8_bytes
                / max(1, self.binary_bytes - self.binary_header_bytes),
                6,
            ),
        }


class SemanticVocabulary:
    """Shared vocabulary connecting exact surface forms to semantic metadata."""

    def __init__(self, records: Iterable[dict[str, Any]], *, metadata: dict[str, Any] | None = None):
        normalized: list[dict[str, Any]] = []
        seen_tokens: set[str] = set()
        seen_glyphs: set[str] = set()
        for expected_id, raw in enumerate(records):
            token = str(raw["token"])
            glyph = str(raw["glyph"])
            token_id = int(raw.get("id", expected_id))
            if token_id != expected_id:
                raise ValueError("vocabulary IDs must be contiguous and frequency ordered")
            if not token or token in seen_tokens:
                raise ValueError(f"duplicate or empty token: {token!r}")
            if len(glyph) != 1 or glyph in seen_glyphs or glyph in {ESCAPE, ESCAPE_END}:
                raise ValueError(f"invalid or duplicate glyph: {glyph!r}")
            seen_tokens.add(token)
            seen_glyphs.add(glyph)
            record = dict(raw)
            record["id"] = token_id
            record["token"] = token
            record["glyph"] = glyph
            normalized.append(record)
        self.records = normalized
        self.metadata = dict(metadata or {})
        self.token_to_id = {record["token"]: record["id"] for record in normalized}
        self.id_to_token = [record["token"] for record in normalized]
        self.token_to_glyph = {record["token"]: record["glyph"] for record in normalized}
        self.glyph_to_token = {record["glyph"]: record["token"] for record in normalized}
        self._record_by_token = {record["token"]: record for record in normalized}
        self.digest = self._calculate_digest()

    def _canonical_payload(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "metadata": self.metadata,
            "records": self.records,
        }

    def _calculate_digest(self) -> str:
        payload = json.dumps(
            self._canonical_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def model_vocab_size(self) -> int:
        return len(self.records) + SPECIAL_TOKEN_COUNT

    def model_id(self, token: str) -> int:
        token_id = self.token_to_id.get(token)
        return UNK_ID if token_id is None else token_id + SPECIAL_TOKEN_COUNT

    def token_for_model_id(self, model_id: int) -> str:
        if model_id < SPECIAL_TOKEN_COUNT:
            return {PAD_ID: "<pad>", BOS_ID: "<bos>", EOS_ID: "<eos>", UNK_ID: "<unk>"}[model_id]
        return self.id_to_token[model_id - SPECIAL_TOKEN_COUNT]

    def encode_model(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [self.model_id(token) for token in tokenize_lossless(text)]
        if add_bos:
            ids.insert(0, BOS_ID)
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def decode_model(self, ids: Iterable[int], *, skip_special: bool = True) -> str:
        output: list[str] = []
        for raw_id in ids:
            model_id = int(raw_id)
            if model_id < SPECIAL_TOKEN_COUNT:
                if skip_special:
                    continue
                output.append(self.token_for_model_id(model_id))
            else:
                token_index = model_id - SPECIAL_TOKEN_COUNT
                if token_index >= len(self.id_to_token):
                    raise ValueError(f"model token ID is outside vocabulary: {model_id}")
                output.append(self.id_to_token[token_index])
        return "".join(output)

    def glyphs_for_model_ids(
        self,
        ids: Iterable[int],
        *,
        skip_special: bool = True,
    ) -> str:
        """Render model IDs directly without re-tokenizing structured tokens."""

        output: list[str] = []
        for raw_id in ids:
            model_id = int(raw_id)
            if model_id < SPECIAL_TOKEN_COUNT:
                if skip_special:
                    continue
                output.append(self.token_for_model_id(model_id))
                continue
            token_index = model_id - SPECIAL_TOKEN_COUNT
            if token_index >= len(self.records):
                raise ValueError(f"model token ID is outside vocabulary: {model_id}")
            output.append(self.records[token_index]["glyph"])
        return "".join(output)

    def encode_glyphs(self, text: str) -> str:
        output: list[str] = []
        for token in tokenize_lossless(text):
            glyph = self.token_to_glyph.get(token)
            if glyph is not None:
                output.append(glyph)
                continue
            raw = token.encode("utf-8")
            output.append(ESCAPE)
            output.extend(chr(BRAILLE_BASE + byte) for byte in raw)
            output.append(ESCAPE_END)
        return "".join(output)

    def decode_glyphs(self, glyphs: str) -> str:
        output: list[str] = []
        index = 0
        while index < len(glyphs):
            glyph = glyphs[index]
            if glyph == ESCAPE:
                index += 1
                raw = bytearray()
                while index < len(glyphs) and glyphs[index] != ESCAPE_END:
                    codepoint = ord(glyphs[index])
                    if not BRAILLE_BASE <= codepoint <= BRAILLE_BASE + 255:
                        raise ValueError("literal escape contains a non-Braille byte")
                    raw.append(codepoint - BRAILLE_BASE)
                    index += 1
                if index >= len(glyphs):
                    raise ValueError("unterminated literal escape")
                output.append(bytes(raw).decode("utf-8"))
                index += 1
                continue
            token = self.glyph_to_token.get(glyph)
            if token is None:
                raise ValueError(f"unknown vocabulary glyph: U+{ord(glyph):04X}")
            output.append(token)
            index += 1
        return "".join(output)

    def encode_binary(self, text: str) -> bytes:
        output = bytearray(BINARY_MAGIC)
        output.extend(bytes.fromhex(self.digest))
        for token in tokenize_lossless(text):
            token_id = self.token_to_id.get(token)
            if token_id is not None:
                output.extend(_uvarint(((token_id + 1) << 1) | 1))
                continue
            raw = token.encode("utf-8")
            output.extend(_uvarint(len(raw) << 1))
            output.extend(raw)
        return bytes(output)

    def decode_binary(self, data: bytes) -> str:
        header_size = len(BINARY_MAGIC) + 32
        if len(data) < header_size or data[: len(BINARY_MAGIC)] != BINARY_MAGIC:
            raise ValueError("invalid VIL codec header")
        digest = data[len(BINARY_MAGIC):header_size].hex()
        if digest != self.digest:
            raise ValueError("vocabulary digest mismatch")
        output: list[str] = []
        offset = header_size
        while offset < len(data):
            code, offset = _read_uvarint(data, offset)
            if code & 1:
                token_id = (code >> 1) - 1
                if not 0 <= token_id < len(self.id_to_token):
                    raise ValueError(f"binary token ID is outside vocabulary: {token_id}")
                output.append(self.id_to_token[token_id])
                continue
            length = code >> 1
            if length <= 0 or offset + length > len(data):
                raise ValueError("invalid or truncated binary literal")
            output.append(data[offset:offset + length].decode("utf-8"))
            offset += length
        return "".join(output)

    def semantics(self, token: str, *, language: str | None = None) -> list[dict[str, Any]]:
        meanings = list(self._record_by_token.get(token, {}).get("semantics", []))
        if language is None:
            return meanings
        return [item for item in meanings if item.get("language") in {None, language}]

    def compression_stats(self, text: str) -> CompressionStats:
        tokens = tokenize_lossless(text)
        glyphs = self.encode_glyphs(text)
        binary = self.encode_binary(text)
        return CompressionStats(
            source_characters=len(text),
            source_tokens=len(tokens),
            known_tokens=sum(token in self.token_to_id for token in tokens),
            glyph_characters=len(glyphs),
            source_utf8_bytes=len(text.encode("utf-8")),
            glyph_utf8_bytes=len(glyphs.encode("utf-8")),
            binary_bytes=len(binary),
            binary_header_bytes=len(BINARY_MAGIC) + 32,
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **self._canonical_payload(),
            "sha256": self.digest,
            "model_special_ids": {
                "pad": PAD_ID,
                "bos": BOS_ID,
                "eos": EOS_ID,
                "unk": UNK_ID,
            },
        }
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "SemanticVocabulary":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != FORMAT:
            raise ValueError(f"unsupported semantic vocabulary format: {payload.get('format')!r}")
        vocabulary = cls(payload["records"], metadata=payload.get("metadata"))
        expected = payload.get("sha256")
        if expected and expected != vocabulary.digest:
            raise ValueError("semantic vocabulary checksum mismatch")
        return vocabulary
