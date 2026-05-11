#!/usr/bin/env python3
import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from datasets import Dataset, Features, Sequence, Value
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Build uniform random-token noise data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", default="EleutherAI/pythia-160m")
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260225)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output).resolve()
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    if (out_dir / "state.json").exists():
        print(f"[skip] exists: {out_dir}")
        return
    if args.rows <= 0 or args.seq_length <= 0:
        raise RuntimeError("--rows and --seq-length must be > 0")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = len(tokenizer)
    features = Features(
        {
            "input_ids": Sequence(Value("int32")),
            "attention_mask": Sequence(Value("int8")),
        }
    )
    rng = np.random.default_rng(args.seed)

    def gen():
        batch_rows = 512
        full_batches, rem = divmod(args.rows, batch_rows)
        for n_rows in [batch_rows] * full_batches + ([rem] if rem else []):
            arr = rng.integers(0, vocab_size, size=(n_rows, args.seq_length), dtype=np.int32)
            for row in arr:
                yield {"input_ids": row.tolist(), "attention_mask": [1] * args.seq_length}

    ds = Dataset.from_generator(gen, features=features)
    ds.save_to_disk(str(out_dir))
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "kind": "uniform_random_tokens",
        "tokenizer": args.tokenizer,
        "vocab_size": int(vocab_size),
        "rows": int(args.rows),
        "seq_length": int(args.seq_length),
        "total_tokens": int(args.rows * args.seq_length),
        "seed": int(args.seed),
    }
    with (out_dir / "build_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved noise dataset: {out_dir}")


if __name__ == "__main__":
    main()
