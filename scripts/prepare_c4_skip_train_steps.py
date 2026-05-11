#!/usr/bin/env python3
import argparse
import json
import shutil
import time
from pathlib import Path

import datasets
from datasets import Dataset, Features, Sequence, Value
from datasets.arrow_writer import ArrowWriter
from transformers import AutoTokenizer


def token_cap_from_steps(steps: int, bsz: int, max_length: int) -> int:
    return int(steps) * int(bsz) * int(max_length)


def maybe_remove(path: Path, overwrite: bool):
    if not path.exists():
        return
    if not overwrite:
        raise RuntimeError(f"Path exists (set --overwrite to replace): {path}")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare C4 train dataset from a skipped prefix with token budget by steps, "
            "for example skip1m + cap_steps=20400."
        )
    )
    parser.add_argument("--repo-id", default="allenai/c4")
    parser.add_argument("--config", default="en")
    parser.add_argument("--split", default="train")
    parser.add_argument("--revision", default="")

    parser.add_argument("--output-root", default="")
    parser.add_argument("--output-name", default="c4_train_skip1m_ep1_20k")

    parser.add_argument("--skip-docs", type=int, default=1000000)
    parser.add_argument("--target-steps", type=int, default=20000)
    parser.add_argument("--cap-steps", type=int, default=20400)

    parser.add_argument("--tokenizer", default="EleutherAI/pythia-160m")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    if args.skip_docs < 0:
        raise RuntimeError("--skip-docs must be >= 0")
    if args.target_steps <= 0 or args.cap_steps <= 0:
        raise RuntimeError("--target-steps and --cap-steps must be > 0")
    if args.cap_steps <= args.target_steps:
        raise RuntimeError("--cap-steps must be > --target-steps to guarantee <1 epoch")
    if args.max_length <= 0 or args.bsz <= 0 or args.batch_size <= 0:
        raise RuntimeError("--max-length, --bsz, --batch-size must be > 0")

    root_dir = Path(__file__).resolve().parents[1]
    if not args.output_root:
        args.output_root = str(root_dir / "data" / "tokenized")

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    out_dir = output_root / args.output_name
    meta_path = output_root / f"{args.output_name}.meta.json"

    maybe_remove(out_dir, args.overwrite)
    maybe_remove(meta_path, args.overwrite)

    tokens_per_step = int(args.bsz * args.max_length)
    target_tokens = token_cap_from_steps(args.target_steps, args.bsz, args.max_length)
    cap_tokens = token_cap_from_steps(args.cap_steps, args.bsz, args.max_length)

    print(f"repo/config/split: {args.repo_id}/{args.config}/{args.split}")
    print(f"skip_docs={args.skip_docs}")
    print(f"tokens_per_step={tokens_per_step}")
    print(f"target_steps={args.target_steps}, target_tokens={target_tokens}")
    print(f"cap_steps={args.cap_steps}, cap_tokens={cap_tokens}")
    print(f"output={out_dir}")

    ds = datasets.load_dataset(
        args.repo_id,
        args.config,
        split=args.split,
        streaming=True,
        revision=args.revision or None,
    )
    if args.skip_docs > 0:
        ds = ds.skip(args.skip_docs)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

    tmp_dir = output_root / f".tmp_c4_build_{args.output_name}_{int(time.time())}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    arrow_path = tmp_dir / "train.arrow"

    features = Features(
        {
            "input_ids": Sequence(Value("int32")),
            "attention_mask": Sequence(Value("int8")),
        }
    )
    writer = ArrowWriter(path=str(arrow_path), features=features)

    docs_seen_after_skip = 0
    docs_written = 0
    tokens_written = 0
    batch_texts: list[str] = []

    def flush_batch():
        nonlocal batch_texts, docs_written, tokens_written
        if not batch_texts:
            return
        tok = tokenizer(
            batch_texts,
            truncation=True,
            max_length=args.max_length,
            add_special_tokens=True,
        )
        for input_ids, attention_mask in zip(tok["input_ids"], tok["attention_mask"]):
            if not input_ids:
                continue
            n_tokens = len(input_ids)
            writer.write(
                {
                    "input_ids": [int(x) for x in input_ids],
                    "attention_mask": [int(x) for x in attention_mask],
                }
            )
            docs_written += 1
            tokens_written += n_tokens
            if tokens_written >= cap_tokens:
                break
        batch_texts = []

    for ex in ds:
        docs_seen_after_skip += 1
        text = ex.get("text", "")
        if not isinstance(text, str) or not text:
            continue
        batch_texts.append(text)
        if len(batch_texts) >= args.batch_size:
            flush_batch()
            if tokens_written >= cap_tokens:
                break

        if docs_seen_after_skip % 100000 == 0:
            print(
                f"progress docs_seen={docs_seen_after_skip} docs_written={docs_written} "
                f"tokens_written={tokens_written}/{cap_tokens}"
            )

    if tokens_written < cap_tokens and batch_texts:
        flush_batch()

    writer.finalize()

    if tokens_written < cap_tokens:
        raise RuntimeError(
            f"Insufficient tokens: wrote {tokens_written}, need at least {cap_tokens}"
        )

    train_ds = Dataset.from_file(str(arrow_path))
    train_ds.save_to_disk(str(out_dir))

    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repo_id": args.repo_id,
        "config": args.config,
        "split": args.split,
        "revision": args.revision or None,
        "tokenizer": args.tokenizer,
        "max_length": args.max_length,
        "bsz": args.bsz,
        "skip_docs": args.skip_docs,
        "target_steps": args.target_steps,
        "cap_steps": args.cap_steps,
        "tokens_per_step": tokens_per_step,
        "target_tokens": target_tokens,
        "cap_tokens": cap_tokens,
        "docs_seen_after_skip": docs_seen_after_skip,
        "docs_written": docs_written,
        "tokens_written": tokens_written,
        "output_dir": str(out_dir),
        "notes": "Built from C4 stream after skipping prefix docs to reduce overlap with PPT.",
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with (out_dir / "build_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if not args.keep_temp:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("Done")
    print(f"- dataset: {out_dir}")
    print(f"- meta:    {meta_path}")
    print(f"- max_steps_floor={tokens_written // tokens_per_step}")


if __name__ == "__main__":
    main()
