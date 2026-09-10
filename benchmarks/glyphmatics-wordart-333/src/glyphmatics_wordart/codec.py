# Copyright 2026 918 Technologies
# SPDX-License-Identifier: Apache-2.0
"""Reversible visual transport for canonical glyph IDs."""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Iterable

from .canon333 import CANON333, Canon333
from .models import GlyphProgram


BRAILLE_BASE = 0x2800
BRAILLE_MAX = 0x28FF


@dataclass(frozen=True, slots=True)
class GlyphTransportCodec:
    """Encode each 1..333 glyph ID as two Braille cells.

    The first cell stores the low eight bits. The second stores the remaining
    two bits plus a six-bit CRC. This is compact, contains no ASCII letters,
    survives the Word Art character policy, and rejects substitution errors.
    """

    canon: Canon333 = CANON333

    @staticmethod
    def _crc6(glyph_id: int) -> int:
        return zlib.crc32(int(glyph_id).to_bytes(2, "big")) & 0x3F

    def encode_id(self, glyph_id: int) -> str:
        definition = self.canon.by_id(glyph_id)
        value = definition.glyph_id
        low = value & 0xFF
        high_and_crc = ((self._crc6(value) & 0x3F) << 2) | ((value >> 8) & 0x03)
        return chr(BRAILLE_BASE + low) + chr(BRAILLE_BASE + high_and_crc)

    def decode_id(self, pair: str) -> int:
        if not isinstance(pair, str) or len(pair) != 2:
            raise ValueError("a glyph transport unit must contain exactly two Braille cells")
        values = [ord(char) - BRAILLE_BASE for char in pair]
        if any(value < 0 or value > 255 for value in values):
            raise ValueError("glyph transport contains a non-Braille character")
        low, high_and_crc = values
        glyph_id = low | ((high_and_crc & 0x03) << 8)
        expected_crc = (high_and_crc >> 2) & 0x3F
        if expected_crc != self._crc6(glyph_id):
            raise ValueError("glyph transport checksum mismatch")
        self.canon.by_id(glyph_id)
        return glyph_id

    def encode_ids(self, glyph_ids: Iterable[int]) -> str:
        return "".join(self.encode_id(glyph_id) for glyph_id in glyph_ids)

    def decode_ids(self, transport: str) -> tuple[int, ...]:
        if not isinstance(transport, str) or len(transport) % 2:
            raise ValueError("transport length must be an even number of Braille cells")
        return tuple(self.decode_id(transport[index : index + 2]) for index in range(0, len(transport), 2))

    def encode_program(self, program: GlyphProgram) -> str:
        return self.encode_ids(program.glyph_ids)

    def decode_tokens(self, transport: str) -> tuple[str, ...]:
        return tuple(self.canon.by_id(glyph_id).token for glyph_id in self.decode_ids(transport))

    def verify_program(self, program: GlyphProgram, transport: str | None = None) -> bool:
        payload = transport if transport is not None else self.encode_program(program)
        return self.decode_tokens(payload) == program.tokens

    def render_grid(self, program: GlyphProgram, cells_per_row: int = 8) -> str:
        if cells_per_row < 1:
            raise ValueError("cells_per_row must be positive")
        units = [self.encode_id(glyph_id) for glyph_id in program.glyph_ids]
        rows = [" ".join(units[i : i + cells_per_row]) for i in range(0, len(units), cells_per_row)]
        return "\n".join(rows)

