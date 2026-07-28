"""Command-line tools for the VIL multilingual semantic glyph model."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys

from .glyph_container import pack_payload, unpack_payload
from .semantic_codec import SemanticVocabulary
from .semantic_data import build_artifacts
from .semantic_data import language_token


def _json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def command_build(args: argparse.Namespace) -> int:
    _json(
        build_artifacts(
            args.semantic_root,
            args.output,
            max_tokens=args.max_tokens,
            min_frequency=args.min_frequency,
        )
    )
    return 0


def command_encode(args: argparse.Namespace) -> int:
    vocabulary = SemanticVocabulary.load(args.vocabulary)
    if args.binary:
        encoded = vocabulary.encode_binary(args.text)
        print(base64.b64encode(encoded).decode("ascii"))
    else:
        print(vocabulary.encode_glyphs(args.text))
    return 0


def command_decode(args: argparse.Namespace) -> int:
    vocabulary = SemanticVocabulary.load(args.vocabulary)
    if args.binary:
        data = base64.b64decode(args.data, validate=True)
        print(vocabulary.decode_binary(data))
    else:
        print(vocabulary.decode_glyphs(args.data))
    return 0


def command_inspect(args: argparse.Namespace) -> int:
    vocabulary = SemanticVocabulary.load(args.vocabulary)
    _json(
        {
            "token": args.token,
            "glyph": vocabulary.token_to_glyph.get(args.token),
            "model_id": vocabulary.model_id(args.token),
            "semantics": vocabulary.semantics(args.token, language=args.language),
        }
    )
    return 0


def command_stats(args: argparse.Namespace) -> int:
    vocabulary = SemanticVocabulary.load(args.vocabulary)
    stats = vocabulary.compression_stats(args.text)
    glyphs = vocabulary.encode_glyphs(args.text)
    binary = vocabulary.encode_binary(args.text)
    if vocabulary.decode_glyphs(glyphs) != args.text:
        raise RuntimeError("visible glyph round-trip failed")
    if vocabulary.decode_binary(binary) != args.text:
        raise RuntimeError("binary round-trip failed")
    _json({"lossless_roundtrip": True, **stats.as_dict()})
    return 0


def command_pack(args: argparse.Namespace) -> int:
    payload = Path(args.input).read_bytes()
    Path(args.output).write_text(
        pack_payload(args.title, args.name, payload),
        encoding="utf-8",
    )
    _json({"ok": True, "input_bytes": len(payload), "output": args.output})
    return 0


def command_unpack(args: argparse.Namespace) -> int:
    payload = unpack_payload(
        Path(args.input).read_text(encoding="utf-8"),
        expected_name=args.name,
    )
    Path(args.output).write_bytes(payload)
    _json({"ok": True, "output_bytes": len(payload), "output": args.output})
    return 0


def command_train(args: argparse.Namespace) -> int:
    from .semantic_train import train

    _json(
        train(
            vocabulary_path=args.vocabulary,
            corpus_path=args.corpus,
            checkpoint_path=args.checkpoint,
            steps=args.steps,
            batch_size=args.batch_size,
            context_length=args.context,
            dimension=args.dimension,
            layers=args.layers,
            heads=args.heads,
            learning_rate=args.learning_rate,
            device_name=args.device,
            validation_fraction=args.validation_fraction,
        )
    )
    return 0


def command_generate(args: argparse.Namespace) -> int:
    import torch

    from .semantic_codec import BOS_ID
    from .semantic_lm import load_checkpoint

    vocabulary = SemanticVocabulary.load(args.vocabulary)
    device_name = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else args.device
    device = torch.device(device_name)
    model, payload = load_checkpoint(
        args.checkpoint,
        device=device,
        expected_vocabulary_sha256=vocabulary.digest,
    )
    prompt_ids = [BOS_ID]
    if args.language:
        prompt_ids.append(vocabulary.model_id(language_token(args.language)))
    prompt_ids.extend(vocabulary.encode_model(args.prompt))
    torch.manual_seed(args.seed)
    inputs = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = model.generate(
        inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )[0].tolist()
    continuation = generated[len(prompt_ids):]
    text = vocabulary.decode_model(continuation)
    _json(
        {
            "text": text,
            "glyphs": vocabulary.encode_glyphs(text),
            "model_ids": continuation,
            "checkpoint_training": payload.get("training", {}),
        }
    )
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="glyphmatics-semantic")
    commands = root.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="build vocabulary and corpus")
    build.add_argument("--semantic-root", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--max-tokens", type=int, default=60000)
    build.add_argument("--min-frequency", type=int, default=1)
    build.set_defaults(func=command_build)

    encode = commands.add_parser("encode", help="encode exact text")
    encode.add_argument("--vocabulary", required=True)
    encode.add_argument("--binary", action="store_true")
    encode.add_argument("text")
    encode.set_defaults(func=command_encode)

    decode = commands.add_parser("decode", help="decode glyphs or base64 binary")
    decode.add_argument("--vocabulary", required=True)
    decode.add_argument("--binary", action="store_true")
    decode.add_argument("data")
    decode.set_defaults(func=command_decode)

    inspect = commands.add_parser("inspect", help="inspect meanings for one exact token")
    inspect.add_argument("--vocabulary", required=True)
    inspect.add_argument("--language")
    inspect.add_argument("token")
    inspect.set_defaults(func=command_inspect)

    stats = commands.add_parser("stats", help="verify round-trip and measure compression")
    stats.add_argument("--vocabulary", required=True)
    stats.add_argument("text")
    stats.set_defaults(func=command_stats)

    pack = commands.add_parser("pack", help="pack a file as a Glyph-LLM3 Braille page")
    pack.add_argument("--title", required=True)
    pack.add_argument("--name", default="semantic_model")
    pack.add_argument("--input", required=True)
    pack.add_argument("--output", required=True)
    pack.set_defaults(func=command_pack)

    unpack = commands.add_parser("unpack", help="unpack a Glyph-LLM3 Braille page")
    unpack.add_argument("--name", default="semantic_model")
    unpack.add_argument("--input", required=True)
    unpack.add_argument("--output", required=True)
    unpack.set_defaults(func=command_unpack)

    train_parser = commands.add_parser("train", help="train the causal semantic glyph LM")
    train_parser.add_argument("--vocabulary", required=True)
    train_parser.add_argument("--corpus", required=True)
    train_parser.add_argument("--checkpoint", required=True)
    train_parser.add_argument("--steps", type=int, default=500)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--context", type=int, default=128)
    train_parser.add_argument("--dimension", type=int, default=192)
    train_parser.add_argument("--layers", type=int, default=4)
    train_parser.add_argument("--heads", type=int, default=6)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--validation-fraction", type=float, default=0.05)
    train_parser.set_defaults(func=command_train)

    generate = commands.add_parser("generate", help="generate with a trained semantic glyph LM")
    generate.add_argument("--vocabulary", required=True)
    generate.add_argument("--checkpoint", required=True)
    generate.add_argument("--language")
    generate.add_argument("--device", default="auto")
    generate.add_argument("--max-new-tokens", type=int, default=64)
    generate.add_argument("--temperature", type=float, default=0.8)
    generate.add_argument("--top-k", type=int, default=50)
    generate.add_argument("--seed", type=int, default=918)
    generate.add_argument("prompt")
    generate.set_defaults(func=command_generate)
    return root


def main(argv: list[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
        return int(args.func(args))
    except (ValueError, RuntimeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
