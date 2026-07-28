import json

import pytest

torch = pytest.importorskip("torch")

from glyphmatics.semantic_codec import SemanticVocabulary, iter_glyphs
from glyphmatics.semantic_lm import load_checkpoint
from glyphmatics.semantic_train import (
    evaluate_checkpoint,
    load_corpus,
    pack_corpus,
    resolve_model_config,
    train,
)


def _vocabulary(path):
    glyphs = iter_glyphs()
    records = []
    for token_id, token in enumerate([" ", ".", "Hello", "world", "glyph", "train", "code"]):
        records.append(
            {
                "id": token_id,
                "token": token,
                "glyph": next(glyphs),
                "frequency": 100 - token_id,
                "semantics": [],
            }
        )
    vocabulary = SemanticVocabulary(records, metadata={"languages": ["en"]})
    vocabulary.save(path)
    return vocabulary


def _corpus(path):
    rows = [
        {"text": "Hello world.", "language": "en", "ids": [6, 4, 7]},
        {"text": "glyph train.", "language": "en", "ids": [8, 4, 9]},
        {"text": "Hello glyph.", "language": "en", "ids": [6, 4, 8]},
        {"text": "world code.", "language": "en", "ids": [7, 4, 10]},
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_resolve_model_config_preset_uses_expected_defaults(tmp_path):
    vocab_path = tmp_path / "vocab.json"
    vocabulary = _vocabulary(vocab_path)
    config = resolve_model_config(vocabulary=vocabulary, preset="small")
    assert config.dimension == 256
    assert config.layers == 6
    assert config.heads == 8
    assert config.context_length == 256


def test_train_resume_and_eval_checkpoint(tmp_path):
    vocab_path = tmp_path / "vocab.json"
    corpus_path = tmp_path / "corpus.jsonl"
    checkpoint_path = tmp_path / "model.pt"
    _vocabulary(vocab_path)
    _corpus(corpus_path)

    first = train(
        vocabulary_path=vocab_path,
        corpus_path=corpus_path,
        checkpoint_path=checkpoint_path,
        steps=2,
        batch_size=2,
        context_length=8,
        dimension=24,
        layers=2,
        heads=4,
        learning_rate=1e-3,
        warmup_steps=1,
        gradient_accumulation_steps=2,
        device_name="cpu",
        validation_fraction=0.25,
        preset="reference",
    )
    assert first["completed_steps"] == 2
    assert first["optimizer_steps"] == 2

    resumed = train(
        vocabulary_path=vocab_path,
        corpus_path=corpus_path,
        checkpoint_path=checkpoint_path,
        steps=4,
        batch_size=2,
        context_length=8,
        dimension=24,
        layers=2,
        heads=4,
        learning_rate=1e-3,
        warmup_steps=1,
        gradient_accumulation_steps=2,
        device_name="cpu",
        validation_fraction=0.25,
        preset="reference",
        resume_from=checkpoint_path,
    )
    assert resumed["completed_steps"] == 4
    assert resumed["optimizer_steps"] == 4
    assert resumed["parameter_count"] > 0

    model, payload = load_checkpoint(checkpoint_path, device="cpu")
    assert payload["training_state"]["step"] == 4
    assert payload["training_state"]["optimizer_steps"] == 4
    assert model.config.context_length == 8

    evaluation = evaluate_checkpoint(
        vocabulary_path=vocab_path,
        corpus_path=corpus_path,
        checkpoint_path=checkpoint_path,
        batch_size=2,
        validation_fraction=0.25,
        device_name="cpu",
    )
    assert evaluation["validation_blocks"] >= 1
    assert evaluation["parameter_count"] == resumed["parameter_count"]
    assert torch.isfinite(torch.tensor(evaluation["validation_loss"]))


def test_pack_corpus_and_train_from_packed_directory(tmp_path):
    vocab_path = tmp_path / "vocab.json"
    corpus_path = tmp_path / "corpus.jsonl"
    packed_dir = tmp_path / "packed"
    checkpoint_path = tmp_path / "packed-model.pt"
    _vocabulary(vocab_path)
    _corpus(corpus_path)

    packed = pack_corpus(corpus_path=corpus_path, output_dir=packed_dir, seed=918)
    assert (packed_dir / "manifest.json").exists()
    assert (packed_dir / "tokens.u32").exists()
    assert packed["record_count"] == 4

    loaded = load_corpus(packed_dir)
    assert loaded.source_kind == "packed"
    assert loaded.record_count == 4
    assert loaded.token_count == packed["token_count"]

    result = train(
        vocabulary_path=vocab_path,
        corpus_path=packed_dir,
        checkpoint_path=checkpoint_path,
        steps=2,
        batch_size=2,
        context_length=8,
        dimension=24,
        layers=2,
        heads=4,
        learning_rate=1e-3,
        warmup_steps=1,
        device_name="cpu",
        validation_fraction=0.25,
        preset="reference",
    )
    assert result["corpus_source_kind"] == "packed"
    assert result["record_count"] == 4

    evaluation = evaluate_checkpoint(
        vocabulary_path=vocab_path,
        corpus_path=packed_dir,
        checkpoint_path=checkpoint_path,
        batch_size=2,
        validation_fraction=0.25,
        device_name="cpu",
    )
    assert evaluation["corpus_source_kind"] == "packed"


def test_train_uses_preset_defaults_when_dimensions_are_omitted(tmp_path):
    vocab_path = tmp_path / "vocab.json"
    corpus_path = tmp_path / "corpus.jsonl"
    checkpoint_path = tmp_path / "preset-model.pt"
    _vocabulary(vocab_path)
    _corpus(corpus_path)

    result = train(
        vocabulary_path=vocab_path,
        corpus_path=corpus_path,
        checkpoint_path=checkpoint_path,
        steps=1,
        batch_size=2,
        learning_rate=1e-3,
        warmup_steps=0,
        device_name="cpu",
        validation_fraction=0.25,
        preset="small",
    )
    assert result["model_config"]["dimension"] == 256
    assert result["model_config"]["layers"] == 6
    assert result["model_config"]["heads"] == 8
    assert result["model_config"]["context_length"] == 256
