#!/usr/bin/env python3
"""
Convert raw text file to HuggingFace datasets format.

Supported input formats:
- int: space-separated token IDs per line
- text: raw text per line, tokenized with a HuggingFace tokenizer

Usage:
    python scripts/convert_to_dataset.py --input data/raw/rnn_1model.txt --output data/tokenized/rnn_1model
    python scripts/convert_to_dataset.py --input data/raw/random.txt --output data/tokenized/random --input_format text
"""
import argparse
import os

from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def convert(
    input_path: str,
    output_path: str,
    input_format: str = "int",
    tokenizer_name: str = "EleutherAI/pythia-160m",
    max_length: int = 2048,
):
    """Convert txt file to datasets format."""
    data = []

    if input_format == "text":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

        with open(input_path, "r") as f:
            for line in tqdm(f, desc=f"Tokenizing {os.path.basename(input_path)}"):
                line = line.strip()
                if line:
                    tokens = tokenizer(
                        line,
                        truncation=True,
                        max_length=max_length,
                        padding="max_length",
                    )
                    data.append(tokens)
    else:
        with open(input_path, "r") as f:
            for line in tqdm(f, desc=f"Loading {os.path.basename(input_path)}"):
                line = line.strip()
                if line:
                    tokens = [int(x) for x in line.split()]
                    data.append({"input_ids": tokens, "attention_mask": [1] * len(tokens)})

    dataset = Dataset.from_list(data)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    dataset.save_to_disk(output_path)
    print(f"Saved {len(dataset)} examples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert raw txt to datasets format")
    parser.add_argument("--input", required=True, help="Input txt file path")
    parser.add_argument("--output", required=True, help="Output datasets directory")
    parser.add_argument(
        "--input_format",
        default="int",
        choices=["int", "text"],
        help="Input format: int (token IDs) or text",
    )
    parser.add_argument(
        "--tokenizer",
        default="EleutherAI/pythia-160m",
        help="Tokenizer name (for text input)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Max sequence length (for text input)",
    )
    args = parser.parse_args()

    convert(
        args.input,
        args.output,
        input_format=args.input_format,
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
