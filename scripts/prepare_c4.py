#!/usr/bin/env python3
"""
Download and prepare C4 dataset for pre-pretraining experiments.

Usage:
    python scripts/prepare_c4.py --output_dir data/tokenized --num_examples 100000 --val_size 10000
"""
import argparse
import os

import datasets
from datasets import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm


def prepare_c4(
    output_dir: str,
    num_examples: int = 100000,
    val_size: int = 10000,
    tokenizer_name: str = "EleutherAI/pythia-160m",
    max_length: int = 2048,
    train_split: str = "train",
    val_split: str = "validation",
):
    """Download C4, tokenize, and save train/val from official splits."""
    print("Downloading C4 train split (streaming)...")
    ds_train = datasets.load_dataset("allenai/c4", "en", split=train_split, streaming=True)

    print(f"Tokenizing {num_examples} train examples...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

    train_data = []
    for i, example in enumerate(tqdm(ds_train, total=num_examples)):
        if i >= num_examples:
            break
        tokens = tokenizer(example["text"], truncation=True, max_length=max_length)
        train_data.append(tokens)

    train_dataset = Dataset.from_list(train_data)

    os.makedirs(output_dir, exist_ok=True)
    c4_train_path = os.path.join(output_dir, "c4_train")
    train_dataset.save_to_disk(c4_train_path)
    print(f"Saved {len(train_dataset)} train examples to {c4_train_path}")

    print("Downloading C4 validation split (streaming)...")
    ds_val = datasets.load_dataset("allenai/c4", "en", split=val_split, streaming=True)
    print(f"Tokenizing {val_size} validation examples...")

    val_data = []
    for i, example in enumerate(tqdm(ds_val, total=val_size)):
        if i >= val_size:
            break
        tokens = tokenizer(example["text"], truncation=True, max_length=max_length)
        val_data.append(tokens)

    val_dataset = Dataset.from_list(val_data)
    c4_val_path = os.path.join(output_dir, "c4_val")
    val_dataset.save_to_disk(c4_val_path)
    print(f"Saved {len(val_dataset)} val examples to {c4_val_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare C4 dataset")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--num_examples", type=int, default=100000, help="Total examples")
    parser.add_argument("--val_size", type=int, default=10000, help="Validation set size")
    parser.add_argument("--tokenizer", default="EleutherAI/pythia-160m", help="Tokenizer name")
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--train_split", default="train", help="C4 train split")
    parser.add_argument("--val_split", default="validation", help="C4 validation split")
    args = parser.parse_args()

    prepare_c4(
        output_dir=args.output_dir,
        num_examples=args.num_examples,
        val_size=args.val_size,
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
        train_split=args.train_split,
        val_split=args.val_split,
    )


if __name__ == "__main__":
    main()
