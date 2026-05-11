#!/usr/bin/env python3
import argparse
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_from_disk
from transformers import AutoTokenizer


def as_train_dataset(ds, name: str) -> Dataset:
    if isinstance(ds, DatasetDict):
        if "train" in ds:
            return ds["train"]
        keys = list(ds.keys())
        if len(keys) == 1:
            return ds[keys[0]]
        raise RuntimeError(f"DatasetDict for {name} has no train split: {keys}")
    return ds


def token_total(ds: Dataset) -> int:
    col = ds._data.column("input_ids")
    total = 0
    for c in col.chunks:
        lens = c.value_lengths().to_numpy(zero_copy_only=False)
        total += int(lens.sum())
    return total


def build_unigram_probs(source_ds: Dataset, vocab_size: int, token_budget: int) -> tuple[np.ndarray, int]:
    counts = np.zeros(vocab_size, dtype=np.int64)
    consumed = 0
    col = source_ds._data.column("input_ids")
    for chunk in col.chunks:
        vals = chunk.values.to_numpy(zero_copy_only=False)
        if vals.size == 0:
            continue
        vals = vals.astype(np.int64, copy=False)
        if token_budget > 0 and consumed + vals.size > token_budget:
            vals = vals[: token_budget - consumed]
        vals = vals[(vals >= 0) & (vals < vocab_size)]
        if vals.size > 0:
            counts += np.bincount(vals, minlength=vocab_size)
            consumed += int(vals.size)
        if token_budget > 0 and consumed >= token_budget:
            break

    total = int(counts.sum())
    if total <= 0:
        raise RuntimeError("Failed to build unigram distribution: token count is zero")
    probs = counts.astype(np.float64) / float(total)
    return probs, total


def save_uniform_dataset(
    out_dir: Path,
    rows: int,
    seq_length: int,
    vocab_size: int,
    seed: int,
    overwrite: bool,
):
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    if (out_dir / "state.json").exists():
        print(f"[skip] uniform exists: {out_dir}")
        return

    features = Features(
        {
            "input_ids": Sequence(Value("int32")),
            "attention_mask": Sequence(Value("int8")),
        }
    )
    rng = np.random.default_rng(seed)

    def gen():
        batch_rows = 512
        full_batches = rows // batch_rows
        rem = rows % batch_rows
        for _ in range(full_batches):
            arr = rng.integers(0, vocab_size, size=(batch_rows, seq_length), dtype=np.int32)
            for row in arr:
                yield {
                    "input_ids": row.tolist(),
                    "attention_mask": [1] * seq_length,
                }
        if rem > 0:
            arr = rng.integers(0, vocab_size, size=(rem, seq_length), dtype=np.int32)
            for row in arr:
                yield {
                    "input_ids": row.tolist(),
                    "attention_mask": [1] * seq_length,
                }

    ds = Dataset.from_generator(gen, features=features)
    ds.save_to_disk(str(out_dir))
    print(f"Saved uniform noise dataset: {out_dir} rows={len(ds)}")


def save_unigram_dataset(
    out_dir: Path,
    rows: int,
    seq_length: int,
    vocab_size: int,
    probs: np.ndarray,
    seed: int,
    overwrite: bool,
):
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    if (out_dir / "state.json").exists():
        print(f"[skip] unigram exists: {out_dir}")
        return

    features = Features(
        {
            "input_ids": Sequence(Value("int32")),
            "attention_mask": Sequence(Value("int8")),
        }
    )
    rng = np.random.default_rng(seed)

    def gen():
        batch_rows = 256
        full_batches = rows // batch_rows
        rem = rows % batch_rows
        for _ in range(full_batches):
            arr = rng.choice(vocab_size, size=(batch_rows, seq_length), p=probs).astype(np.int32)
            for row in arr:
                yield {
                    "input_ids": row.tolist(),
                    "attention_mask": [1] * seq_length,
                }
        if rem > 0:
            arr = rng.choice(vocab_size, size=(rem, seq_length), p=probs).astype(np.int32)
            for row in arr:
                yield {
                    "input_ids": row.tolist(),
                    "attention_mask": [1] * seq_length,
                }

    ds = Dataset.from_generator(gen, features=features)
    ds.save_to_disk(str(out_dir))
    print(f"Saved unigram noise dataset: {out_dir} rows={len(ds)}")


def save_shuffle_dataset(
    out_dir: Path,
    source_ds: Dataset,
    start_row: int,
    rows: int,
    seed: int,
    overwrite: bool,
):
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    if (out_dir / "state.json").exists():
        print(f"[skip] shuffle exists: {out_dir}")
        return

    n = len(source_ds)
    if start_row < 0 or start_row >= n:
        raise RuntimeError(f"Invalid --shuffle-start-row {start_row} for n={n}")
    if rows <= 0:
        raise RuntimeError("--shuffle-rows must be > 0")
    if start_row + rows > n:
        raise RuntimeError(
            "Shuffle row range exceeds source dataset size. "
            f"start={start_row}, rows={rows}, n={n}"
        )

    subset = source_ds.select(range(start_row, start_row + rows))
    features = Features(
        {
            "input_ids": Sequence(Value("int32")),
            "attention_mask": Sequence(Value("int8")),
        }
    )

    def gen():
        rng = np.random.default_rng(seed)
        for i in range(len(subset)):
            row = subset[i]
            ids = row["input_ids"]
            arr = np.asarray(ids, dtype=np.int32)
            if arr.size <= 1:
                shuffled = arr
            else:
                perm = rng.permutation(arr.size)
                shuffled = arr[perm]
            yield {
                "input_ids": shuffled.tolist(),
                "attention_mask": [1] * int(shuffled.size),
            }

    ds = Dataset.from_generator(gen, features=features)
    ds.save_to_disk(str(out_dir))
    print(f"Saved shuffle noise dataset: {out_dir} rows={len(ds)}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build reusable PT noise-family datasets: uniform vocab, c4-unigram, c4-shuffled rows."
        )
    )
    parser.add_argument(
        "--source-data",
        default="",
        help="Source C4 dataset for unigram/shuffle construction. Defaults to pre-pretraining/data/tokenized/c4_train",
    )
    parser.add_argument(
        "--output-root",
        default="",
        help="Output directory for tokenized noise datasets",
    )
    parser.add_argument("--tokenizer", default="EleutherAI/pythia-160m")
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--uniform-rows", type=int, default=260000)
    parser.add_argument("--unigram-rows", type=int, default=260000)
    parser.add_argument("--unigram-token-budget", type=int, default=150_000_000)
    parser.add_argument("--shuffle-start-row", type=int, default=200000)
    parser.add_argument("--shuffle-rows", type=int, default=800000)
    parser.add_argument("--seed", type=int, default=20260225)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    ppt_dir = script_path.parent.parent

    if not args.source_data:
        args.source_data = str(ppt_dir / "data" / "tokenized" / "c4_train")
    if not args.output_root:
        args.output_root = str(ppt_dir / "data" / "tokenized" / "noise_families")

    source_data = Path(args.source_data).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not source_data.exists():
        raise RuntimeError(f"Source dataset not found: {source_data}")
    if args.seq_length <= 0:
        raise RuntimeError("--seq-length must be > 0")

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = len(tok)
    print(f"Tokenizer vocab size ({args.tokenizer}) = {vocab_size}")

    source_ds = as_train_dataset(load_from_disk(str(source_data)), name="source")
    source_tokens = token_total(source_ds)
    print(f"Source rows={len(source_ds)}, tokens={source_tokens}")

    uniform_name = f"noise_uniform_vocab_int_{args.uniform_rows // 1000}k_seq{args.seq_length}"
    unigram_name = f"noise_c4unigram_int_{args.unigram_rows // 1000}k_seq{args.seq_length}"
    shuffle_name = f"noise_c4shuffle_rows_{args.shuffle_rows // 1000}k"

    uniform_dir = output_root / uniform_name
    unigram_dir = output_root / unigram_name
    shuffle_dir = output_root / shuffle_name

    save_uniform_dataset(
        out_dir=uniform_dir,
        rows=args.uniform_rows,
        seq_length=args.seq_length,
        vocab_size=vocab_size,
        seed=args.seed + 1,
        overwrite=args.overwrite,
    )

    probs, sampled_tokens = build_unigram_probs(
        source_ds=source_ds,
        vocab_size=vocab_size,
        token_budget=args.unigram_token_budget,
    )
    save_unigram_dataset(
        out_dir=unigram_dir,
        rows=args.unigram_rows,
        seq_length=args.seq_length,
        vocab_size=vocab_size,
        probs=probs,
        seed=args.seed + 2,
        overwrite=args.overwrite,
    )

    save_shuffle_dataset(
        out_dir=shuffle_dir,
        source_ds=source_ds,
        start_row=args.shuffle_start_row,
        rows=args.shuffle_rows,
        seed=args.seed + 3,
        overwrite=args.overwrite,
    )

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_data": str(source_data),
        "source_rows": int(len(source_ds)),
        "source_tokens": int(source_tokens),
        "tokenizer": args.tokenizer,
        "vocab_size": int(vocab_size),
        "seq_length": int(args.seq_length),
        "uniform": {
            "name": uniform_name,
            "rows": int(args.uniform_rows),
            "tokens": int(args.uniform_rows * args.seq_length),
            "path": str(uniform_dir),
        },
        "unigram": {
            "name": unigram_name,
            "rows": int(args.unigram_rows),
            "tokens": int(args.unigram_rows * args.seq_length),
            "unigram_token_budget": int(args.unigram_token_budget),
            "unigram_tokens_sampled": int(sampled_tokens),
            "path": str(unigram_dir),
        },
        "shuffle": {
            "name": shuffle_name,
            "rows": int(args.shuffle_rows),
            "shuffle_start_row": int(args.shuffle_start_row),
            "path": str(shuffle_dir),
        },
    }
    meta_path = output_root / "noise_family_build_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
