(function () {
  const ESCAPE = "\uE000";
  const ESCAPE_END = "\uE001";
  const BRAILLE_BASE = 0x2800;
  const BINARY_MAGIC = [86, 73, 76, 67, 49]; // VILC1

  const state = {
    benchmark: null,
    vocab: null,
    semanticSamples: [],
    systems: [],
  };

  function isWordCharacter(char) {
    return char === "_" || /[\p{L}\p{M}\p{N}]/u.test(char);
  }

  function tokenizeLossless(text) {
    const chars = Array.from(text);
    const tokens = [];
    let index = 0;
    while (index < chars.length) {
      const char = chars[index];
      if (/\s/u.test(char)) {
        let end = index + 1;
        while (end < chars.length && /\s/u.test(chars[end])) {
          end += 1;
        }
        tokens.push(chars.slice(index, end).join(""));
        index = end;
        continue;
      }
      if (isWordCharacter(char)) {
        let end = index + 1;
        while (end < chars.length) {
          const candidate = chars[end];
          if (isWordCharacter(candidate)) {
            end += 1;
            continue;
          }
          if (
            (candidate === "'" || candidate === "’") &&
            end + 1 < chars.length &&
            isWordCharacter(chars[end - 1]) &&
            isWordCharacter(chars[end + 1])
          ) {
            end += 1;
            continue;
          }
          break;
        }
        tokens.push(chars.slice(index, end).join(""));
        index = end;
        continue;
      }
      tokens.push(char);
      index += 1;
    }
    return tokens;
  }

  function createVocabulary(payload) {
    const tokenToGlyph = new Map();
    const glyphToToken = new Map();
    const tokenToId = new Map();
    for (const record of payload.records) {
      tokenToGlyph.set(record.token, record.glyph);
      glyphToToken.set(record.glyph, record.token);
      tokenToId.set(record.token, record.id);
    }
    return {
      tokenToGlyph,
      glyphToToken,
      tokenToId,
      sha256: payload.sha256,
    };
  }

  function encodeGlyphs(vocab, text) {
    const output = [];
    const encoder = new TextEncoder();
    for (const token of tokenizeLossless(text)) {
      const glyph = vocab.tokenToGlyph.get(token);
      if (glyph) {
        output.push(glyph);
        continue;
      }
      output.push(ESCAPE);
      for (const byte of encoder.encode(token)) {
        output.push(String.fromCodePoint(BRAILLE_BASE + byte));
      }
      output.push(ESCAPE_END);
    }
    return output.join("");
  }

  function decodeGlyphs(vocab, glyphs) {
    const output = [];
    const decoder = new TextDecoder();
    const chars = Array.from(glyphs);
    let index = 0;
    while (index < chars.length) {
      const glyph = chars[index];
      if (glyph === ESCAPE) {
        index += 1;
        const bytes = [];
        while (index < chars.length && chars[index] !== ESCAPE_END) {
          bytes.push(chars[index].codePointAt(0) - BRAILLE_BASE);
          index += 1;
        }
        if (index >= chars.length) {
          throw new Error("unterminated literal escape");
        }
        output.push(decoder.decode(new Uint8Array(bytes)));
        index += 1;
        continue;
      }
      const token = vocab.glyphToToken.get(glyph);
      if (token === undefined) {
        throw new Error(`unknown vocabulary glyph: ${glyph}`);
      }
      output.push(token);
      index += 1;
    }
    return output.join("");
  }

  function encodeUvarint(value) {
    if (value < 0) {
      throw new Error("varint cannot encode a negative value");
    }
    const output = [];
    let current = value;
    while (current >= 0x80) {
      output.push((current & 0x7f) | 0x80);
      current >>= 7;
    }
    output.push(current);
    return output;
  }

  function hexToBytes(hex) {
    const output = [];
    for (let index = 0; index < hex.length; index += 2) {
      output.push(parseInt(hex.slice(index, index + 2), 16));
    }
    return output;
  }

  function encodeBinary(vocab, text) {
    const encoder = new TextEncoder();
    const output = [...BINARY_MAGIC, ...hexToBytes(vocab.sha256)];
    for (const token of tokenizeLossless(text)) {
      const tokenIndex = vocab.tokenToId.get(token);
      if (tokenIndex !== undefined) {
        output.push(...encodeUvarint(((tokenIndex + 1) << 1) | 1));
        continue;
      }
      const raw = Array.from(encoder.encode(token));
      output.push(...encodeUvarint(raw.length << 1));
      output.push(...raw);
    }
    return new Uint8Array(output);
  }

  function compressionStats(vocab, text) {
    const tokens = tokenizeLossless(text);
    const glyphs = encodeGlyphs(vocab, text);
    const binary = encodeBinary(vocab, text);
    const sourceBytes = new TextEncoder().encode(text).length;
    const glyphBytes = new TextEncoder().encode(glyphs).length;
    const headerBytes = BINARY_MAGIC.length + 32;
    const knownTokens = tokens.filter((token) => vocab.tokenToGlyph.has(token)).length;
    return {
      sourceCharacters: Array.from(text).length,
      sourceTokens: tokens.length,
      knownTokens,
      knownTokenRatio: (knownTokens / Math.max(1, tokens.length)).toFixed(6),
      glyphCharacters: Array.from(glyphs).length,
      sourceUtf8Bytes: sourceBytes,
      glyphUtf8Bytes: glyphBytes,
      binaryBytes: binary.length,
      binaryHeaderBytes: headerBytes,
      binaryPayloadBytes: binary.length - headerBytes,
      visualCharacterRatio: (Array.from(text).length / Math.max(1, Array.from(glyphs).length)).toFixed(6),
      glyphUtf8Ratio: (sourceBytes / Math.max(1, glyphBytes)).toFixed(6),
      binaryRatio: (sourceBytes / Math.max(1, binary.length)).toFixed(6),
      binaryPayloadRatio: (sourceBytes / Math.max(1, binary.length - headerBytes)).toFixed(6),
      glyphs,
      binary,
    };
  }

  function renderHeroSummary(summary) {
    const container = document.getElementById("hero-summary");
    const metrics = [
      ["Semantic samples", summary.semantic_sample_count],
      ["Executable systems", summary.executable_system_count],
      ["Mean semantic binary ratio", summary.mean_semantic_binary_ratio],
      ["Mean system char ratio", summary.mean_system_char_ratio],
    ];
    container.innerHTML = metrics
      .map(
        ([label, value]) => `
          <div class="summary-card">
            <strong>${label}</strong>
            <div class="summary-value">${value}</div>
          </div>
        `,
      )
      .join("");
  }

  function renderSemanticSamples(rows) {
    const body = document.getElementById("semantic-samples-body");
    body.innerHTML = rows
      .map(
        (row) => `
          <tr>
            <td><strong>${row.label}</strong><div class="small">${row.language}</div></td>
            <td>${escapeHtml(row.text)}</td>
            <td>${row.stats.known_token_ratio}</td>
            <td>${row.stats.visual_character_ratio}</td>
            <td>${row.stats.binary_ratio}</td>
          </tr>
        `,
      )
      .join("");
  }

  function renderSystems(rows) {
    const filter = document.getElementById("system-filter").value.trim().toLowerCase();
    const filtered = rows.filter((row) => {
      if (!filter) {
        return true;
      }
      return (
        row.system_id.toLowerCase().includes(filter) ||
        row.title.toLowerCase().includes(filter)
      );
    });
    const body = document.getElementById("systems-body");
    body.innerHTML = filtered
      .map(
        (row) => `
          <tr>
            <td class="glyph-cell">${row.glyph}</td>
            <td>
              <strong>${row.title}</strong>
              <div class="small">${row.system_id}</div>
              <div class="small">${escapeHtml(row.description)}</div>
            </td>
            <td>${row.source_characters}</td>
            <td>${row.lossless_program_glyph_characters}</td>
            <td>${row.system_char_ratio}</td>
            <td>${row.semantic_stats_v3.binary_ratio}</td>
          </tr>
        `,
      )
      .join("");
  }

  function renderLiveMetrics(result) {
    const metrics = [
      ["Source chars", result.sourceCharacters],
      ["Source tokens", result.sourceTokens],
      ["Known token ratio", result.knownTokenRatio],
      ["Glyph chars", result.glyphCharacters],
      ["Source bytes", result.sourceUtf8Bytes],
      ["Glyph bytes", result.glyphUtf8Bytes],
      ["Binary bytes", result.binaryBytes],
      ["Binary ratio", result.binaryRatio],
    ];
    document.getElementById("live-metrics").innerHTML = metrics
      .map(
        ([label, value]) => `
          <article class="metric-card">
            <strong>${label}</strong>
            <div class="metric-value">${value}</div>
          </article>
        `,
      )
      .join("");
    document.getElementById("glyph-preview").textContent = result.glyphs;
    document.getElementById("binary-preview").textContent = Array.from(result.binary.slice(0, 64))
      .map((byte) => byte.toString(16).padStart(2, "0"))
      .join(" ");
  }

  function escapeHtml(text) {
    return text
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");
  }

  async function loadBenchmarkData() {
    const response = await fetch("./assets/benchmark-data.json");
    state.benchmark = await response.json();
    state.semanticSamples = state.benchmark.semantic_samples;
    state.systems = state.benchmark.executable_systems;
    renderHeroSummary(state.benchmark.summary);
    renderSemanticSamples(state.semanticSamples);
    renderSystems(state.systems);
    const select = document.getElementById("sample-select");
    select.innerHTML = state.semanticSamples
      .map((row) => `<option value="${row.id}">${row.label}</option>`)
      .join("");
    document.getElementById("input-text").value = state.semanticSamples[0].text;
  }

  async function loadVocabulary() {
    const response = await fetch("./assets/semantic-vocab-v2.json");
    const payload = await response.json();
    state.vocab = createVocabulary(payload);
    document.getElementById("live-status").textContent =
      `Loaded semantic v2 vocabulary: ${payload.records.length} records, digest ${payload.sha256.slice(0, 12)}…`;
  }

  function runLiveBenchmark() {
    if (!state.vocab) {
      document.getElementById("live-status").textContent = "Vocabulary is still loading.";
      return;
    }
    const text = document.getElementById("input-text").value;
    const result = compressionStats(state.vocab, text);
    const roundTrip = decodeGlyphs(state.vocab, result.glyphs) === text;
    document.getElementById("live-status").textContent =
      `Round-trip ${roundTrip ? "passed" : "failed"} using the browser-side semantic codec.`;
    renderLiveMetrics(result);
  }

  function attachEvents() {
    document.getElementById("run-benchmark").addEventListener("click", runLiveBenchmark);
    document.getElementById("load-sample").addEventListener("click", () => {
      const current = document.getElementById("sample-select").value;
      const row = state.semanticSamples.find((item) => item.id === current);
      if (row) {
        document.getElementById("input-text").value = row.text;
        runLiveBenchmark();
      }
    });
    document.getElementById("system-filter").addEventListener("input", () => {
      renderSystems(state.systems);
    });
  }

  async function boot() {
    attachEvents();
    await loadBenchmarkData();
    loadVocabulary().then(runLiveBenchmark).catch((error) => {
      document.getElementById("live-status").textContent = `Vocabulary load failed: ${String(error)}`;
    });
  }

  boot().catch((error) => {
    document.getElementById("hero-summary").innerHTML = `
      <div class="summary-card">
        <strong>Benchmark load failed</strong>
        <div class="summary-value">error</div>
        <div class="small">${escapeHtml(String(error))}</div>
      </div>
    `;
  });
})();
