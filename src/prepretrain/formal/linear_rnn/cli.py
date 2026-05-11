from __future__ import annotations

import argparse
import sys

from .dataset import (
    iter_sequences,
    load_tokenizer,
    resolve_device,
    save_jsonl,
    save_txt,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Random RNN generator")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate sequences")
    gen.add_argument("--out", required=True, help="Output file path")
    gen.add_argument("--format", choices=["jsonl", "txt"], default="jsonl")
    gen.add_argument("--vocab-size", type=int, default=None)
    gen.add_argument("--tokenizer-name", default="EleutherAI/pythia-160m")
    gen.add_argument(
        "--detokenize",
        action="store_true",
        help="Decode token ids to text output",
    )
    gen.add_argument("--hidden-size", type=int, default=64)
    gen.add_argument("--seq-length", type=int, default=1000)
    gen.add_argument("--num-models", type=int, default=1)
    gen.add_argument("--num-seqs", type=int, default=10)
    gen.add_argument("--batch-size", type=int, default=1)
    gen.add_argument("--temperature", type=float, default=1.0)
    gen.add_argument("--spectral-radius", type=float, default=0.9)
    gen.add_argument("--burn-in", type=int, default=0)
    gen.add_argument("--approx-k", type=int, default=None)
    gen.add_argument(
        "--model-type", choices=["linear", "tanh", "relu"], default="linear"
    )
    gen.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    gen.add_argument("--device", default="auto")
    gen.add_argument("--model-seed", type=int, default=None)
    gen.add_argument("--sequence-seed", type=int, default=None)
    gen.add_argument("--seed", type=int, default=0)

    return parser


def run_generate(args: argparse.Namespace) -> int:
    tokenizer = None
    vocab_size = args.vocab_size
    if args.detokenize:
        tokenizer = load_tokenizer(args.tokenizer_name)
        vocab_size = len(tokenizer)
        if args.vocab_size is not None and args.vocab_size != vocab_size:
            raise ValueError("vocab_size must match tokenizer length when detokenizing")
    device = resolve_device(args.device)
    records = iter_sequences(
        num_models=args.num_models,
        num_sequences_per_model=args.num_seqs,
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        seq_length=args.seq_length,
        temperature=args.temperature,
        spectral_radius=args.spectral_radius,
        burn_in=args.burn_in,
        model_type=args.model_type,
        tokenizer_name=args.tokenizer_name,
        seed=args.seed,
        model_seed=args.model_seed,
        sequence_seed=args.sequence_seed,
        dtype=args.dtype,
        batch_size=args.batch_size,
        approx_k=args.approx_k, # lagacy feature, not used
        device=device,
    )
    if args.format == "jsonl":
        save_jsonl(records, args.out, tokenizer=tokenizer, detokenize=args.detokenize)
    else:
        save_txt(records, args.out, tokenizer=tokenizer, detokenize=args.detokenize)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate":
        return run_generate(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
