import pytest

torch = pytest.importorskip("torch")

from glyphmatics.semantic_lm import ModelConfig, SemanticGlyphLM


def test_causal_model_shapes_and_loss():
    config = ModelConfig(
        vocab_size=32,
        context_length=8,
        dimension=24,
        layers=2,
        heads=4,
        dropout=0.0,
    )
    model = SemanticGlyphLM(config)
    inputs = torch.randint(0, config.vocab_size, (2, 8))
    logits, loss = model(inputs, targets=inputs)
    assert logits.shape == (2, 8, config.vocab_size)
    assert loss is not None and torch.isfinite(loss)


def test_future_tokens_do_not_change_earlier_logits():
    config = ModelConfig(
        vocab_size=24,
        context_length=6,
        dimension=24,
        layers=2,
        heads=4,
        dropout=0.0,
    )
    model = SemanticGlyphLM(config).eval()
    first = torch.tensor([[1, 5, 6, 7, 8, 9]])
    second = torch.tensor([[1, 5, 6, 20, 21, 22]])
    with torch.no_grad():
        first_logits, _ = model(first)
        second_logits, _ = model(second)
    assert torch.allclose(first_logits[:, :3], second_logits[:, :3], atol=1e-6)
