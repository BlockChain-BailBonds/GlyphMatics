# GlyphMatics Word-Art Bridge

This package turns Kaggle Word Art into a focused evaluation layer for the
GlyphMatics 333 canon. It preserves the exact `GlyphCanon333` token ordering
from the production RDL/ADL/GhostBridge artifact and adds:

- strict canonical program parsing and exact-333 startup verification;
- a reversible, word-filter-safe Braille transport for all 333 glyph IDs;
- Word Art artist/guesser prompts and `<art>...</art>` parsing;
- local drawing disqualification checks with automatic delegation to Kaggle's
  official `check_art` when `kaggle-environments` is installed;
- semantic, graph, robustness, round-trip, compression, execution, and
  collision scoring;
- a hash-chained, append-only ADL/GhostBridge failure ledger;
- a Kaggle Benchmarks task using isolated artist and guesser conversations;
- a self-contained Kaggle notebook and deterministic local tests.

## What it measures

The bridge evaluates the path:

`concept -> canonical graph -> 333-glyph program -> constrained art -> guess -> recovered concept`

It does **not** claim that Word Art alone proves arbitrary program execution or
lossless general-purpose compression. Those are reported as separate score
components rather than being folded invisibly into guess accuracy.

The composite score is:

```text
100 * (0.30 semantic + 0.20 graph + 0.15 robustness
     + 0.15 round_trip + 0.10 compression + 0.10 execution
     - collision_penalty)
```

Word Art points are also retained exactly: first guess `2.0`, second `1.5`,
third `1.0`, otherwise `0.0`.

## Local validation

```bash
python -m pip install -e .
python -m pytest -q
python -m glyphmatics_wordart.selftest
python -m glyphmatics_wordart.cli demo --output GLYPHMATICS_WORDART_REPORT.json
```

The core package is standard-library only. `pytest` is optional, and the demo
does not call an external model.

## Kaggle task

Open a new Kaggle Benchmark Task notebook and run
`GLYPHMATICS_WORDART_KAGGLE.ipynb`, or push the generated task script:

```bash
kaggle b init -y
kaggle b t push glyphmatics-wordart-333 -f GLYPHMATICS_WORDART_TASK.py --wait
kaggle b t run glyphmatics-wordart-333 -m <model-slug> --wait
```

`kaggle_task.py` intentionally ends with
`glyphmatics_wordart_333.run(kbench.llm)` because a decorated task that is not
run produces no benchmark result.

## Output contracts

Artist output:

```text
<glyphmatics>{"version":"glyphmatics-333-v1","tokens":["OBJECT","SYMMETRY"],"edges":[[0,1,"CONTAIN"]]}</glyphmatics>
<art>
 /\_/\\
( o.o )
 > ^ <
</art>
```

Guesser output:

```json
{"guess":"cat"}
```

Every token and every edge relation must exist in the exact 333 canon. The
drawing is separately checked for target-word leakage, textual labels, maximum
length, and empty content.

## Files

- `src/glyphmatics_wordart/`: reusable bridge implementation.
- `GLYPHMATICS_WORDART_TASK.py`: self-contained Kaggle Benchmarks task.
- `kaggle_task.py`: readable development form of the task.
- `GLYPHMATICS_WORDART_KAGGLE.ipynb`: self-contained notebook.
- `tests/`: unit and integration coverage.
- `data/cases.json`: deterministic public evaluation cases.
