#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

import numpy as np
from datasets import Dataset, Features, Sequence, Value
from transformers import AutoTokenizer


def save_random_dataset(path: Path, rows: int, seq_length: int, vocab_size: int, seed: int, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    if (path / "state.json").exists():
        print(f"[skip] {path}")
        return

    rng = np.random.default_rng(seed)
    features = Features({"input_ids": Sequence(Value("int32")), "attention_mask": Sequence(Value("int8"))})

    def gen():
        arr = rng.integers(0, vocab_size, size=(rows, seq_length), dtype=np.int32)
        for row in arr:
            yield {"input_ids": row.tolist(), "attention_mask": [1] * seq_length}

    Dataset.from_generator(gen, features=features).save_to_disk(str(path))
    print(f"saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create tiny local datasets for smoke tests")
    parser.add_argument("--output-root", default="data/tokenized")
    parser.add_argument("--tokenizer", default="EleutherAI/pythia-70m")
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--eval-rows", type=int, default=8)
    parser.add_argument("--seq-length", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    vocab_size = len(AutoTokenizer.from_pretrained(args.tokenizer))

    save_random_dataset(out / "c4_val", args.eval_rows, args.seq_length, vocab_size, 1, args.overwrite)
    save_random_dataset(out / "ptmix_vocabfull_p000", args.rows, args.seq_length, vocab_size, 2, args.overwrite)
    save_random_dataset(out / "ptmix_vocabfull_p050", args.rows + args.rows // 2, args.seq_length, vocab_size, 3, args.overwrite)
    save_random_dataset(out / "rnn_m1000", args.rows, args.seq_length, vocab_size, 4, args.overwrite)
    save_random_dataset(out / "shuffdyck_d8", args.rows, args.seq_length, vocab_size, 5, args.overwrite)


if __name__ == "__main__":
    main()
