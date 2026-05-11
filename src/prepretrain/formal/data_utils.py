"""
Data utilities for pre-pretraining experiments.
Includes functions for caching/tokenizing data and generating formal languages.
"""
import os
import random

import datasets
import fire
import numpy as np
from datasets import Dataset, load_dataset
from tqdm import tqdm, trange
from transformers import AutoTokenizer


def cache_data(
    dataset_path: str,
    out_dir: str,
    tokenizer_name: str = "EleutherAI/pythia-160m",
    input_format: str = "text",
):
    """
    Cache and tokenize dataset.

    Args:
        dataset_path: Path to the dataset file (text file, one sequence per line)
        out_dir: Output directory for cached data
        tokenizer_name: Name of tokenizer to use
        input_format: "text" for space-separated tokens, "int" for token IDs
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

    if input_format == "int":
        # Each line is space-separated token IDs
        data = []
        with open(dataset_path, "r") as f:
            for line in tqdm(f, desc="Loading int data"):
                line = line.strip()
                if line:
                    tokens = [int(x) for x in line.split()]
                    data.append({"input_ids": tokens, "attention_mask": [1] * len(tokens)})
        dataset = Dataset.from_list(data)
    else:
        # Text format - tokenize with tokenizer
        dataset = load_dataset("text", data_files=dataset_path, split="train")
        dataset = dataset.map(
            lambda x: tokenizer(
                x["text"],
                truncation=True,
                max_length=2048,
            ),
        ).remove_columns(["text"])

    os.makedirs(out_dir, exist_ok=True)
    dataset.save_to_disk(out_dir)
    print(f"Saved tokenized dataset to {out_dir}")
    print(f"Dataset size: {len(dataset)} examples")
    return dataset


def generate_shuff_dyck(k, max_length=2048, p_open=0.5, min_depth=1, max_depth=8):
    """
    Generate a k-shuffle Dyck sequence (cross-serial dependencies).

    Args:
        k: Number of different types of brackets
        max_length: Target maximum length of the sequence
        p_open: Probability of opening a new bracket
        min_depth: Minimum required depth
        max_depth: Maximum nesting depth allowed

    Returns:
        list: Generated sequence where i represents opening bracket i
             and i+k represents closing bracket i
    """
    sequence = []
    counts = [0] * k

    if min_depth < 1:
        raise ValueError("min_depth must be at least 1.")

    # Initialize with minimum depth
    for _ in range(min_depth):
        bracket = random.randint(0, k - 1)
        sequence.append(bracket)
        counts[bracket] += 1

    while len(sequence) < max_length:
        depth = sum(counts)

        if depth == 0:
            bracket = random.randint(0, k - 1)
            sequence.append(bracket)
            counts[bracket] += 1
            continue

        if depth >= max_depth:
            open_brackets = [i for i, count in enumerate(counts) if count > 0]
            bracket = random.choice(open_brackets)
            sequence.append(bracket + k)
            counts[bracket] -= 1
            continue

        if random.random() < p_open and depth < max_depth:
            bracket = random.randint(0, k - 1)
            sequence.append(bracket)
            counts[bracket] += 1
        else:
            open_brackets = [i for i, count in enumerate(counts) if count > 0]
            bracket = random.choice(open_brackets)
            sequence.append(bracket + k)
            counts[bracket] -= 1

    return sequence


def generate_shuff_dyck_dataset(
    out_path: str,
    num_symbols: int = 64,
    n: int = 100000,
    target_length: int = 2048,
    p: float = 0.51,
    min_depth: int = 1,
    max_depth: int = 8,
):
    """
    Generate shuffle Dyck sequences and save to file.

    Args:
        out_path: Output file path
        num_symbols: Number of distinct symbol pairs
        n: Number of sequences to generate
        target_length: Target sequence length
        p: Probability of opening bracket
        min_depth: Minimum required depth
        max_depth: Maximum nesting depth allowed
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w") as f:
        for i in trange(n, desc="Generating shuffle Dyck"):
            result = generate_shuff_dyck(
                num_symbols,
                target_length,
                p,
                min_depth=min_depth,
                max_depth=max_depth,
            )
            dyck_str = " ".join([str(x) for x in result[:target_length]])
            f.write(f"{dyck_str}\n")

    print(f"Generated {n} sequences to {out_path}")


def generate_k_dyck(k, max_length=2048, p_open=0.5, min_depth=1, max_depth=16):
    """
    Generate a k-Dyck sequence (nested/well-balanced parentheses with k bracket types).
    Unlike shuffle Dyck, brackets must be properly nested.

    Args:
        k: Number of different types of brackets
        max_length: Target maximum length of the sequence
        p_open: Probability of opening a new bracket
        min_depth: Minimum required depth
        max_depth: Maximum nesting depth allowed

    Returns:
        list: Generated sequence where i represents opening bracket i
             and i+k represents closing bracket i
    """
    sequence = []
    stack = []  # Stack to track open brackets for proper nesting

    # Initialize with minimum depth
    for _ in range(min_depth):
        bracket = random.randint(0, k - 1)
        sequence.append(bracket)
        stack.append(bracket)

    while len(sequence) < max_length:
        depth = len(stack)

        if depth == 0:
            # Must open a bracket
            bracket = random.randint(0, k - 1)
            sequence.append(bracket)
            stack.append(bracket)
            continue

        if depth >= max_depth:
            # Must close the most recent bracket (proper nesting)
            bracket = stack.pop()
            sequence.append(bracket + k)
            continue

        if random.random() < p_open:
            # Open a new bracket
            bracket = random.randint(0, k - 1)
            sequence.append(bracket)
            stack.append(bracket)
        else:
            # Close the most recent bracket (proper nesting)
            bracket = stack.pop()
            sequence.append(bracket + k)

    # Close any remaining open brackets
    while stack:
        bracket = stack.pop()
        sequence.append(bracket + k)

    return sequence[:max_length]


def generate_k_dyck_dataset(
    out_path: str,
    num_symbols: int = 64,
    n: int = 100000,
    target_length: int = 2048,
    p: float = 0.5,
    min_depth: int = 1,
    max_depth: int = 16,
):
    """
    Generate k-Dyck sequences (properly nested) and save to file.

    Args:
        out_path: Output file path
        num_symbols: Number of distinct symbol pairs (k)
        n: Number of sequences to generate
        target_length: Target sequence length
        p: Probability of opening bracket
        min_depth: Minimum required depth
        max_depth: Maximum nesting depth allowed
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w") as f:
        for i in trange(n, desc="Generating k-Dyck"):
            result = generate_k_dyck(
                num_symbols,
                target_length,
                p,
                min_depth=min_depth,
                max_depth=max_depth,
            )
            dyck_str = " ".join([str(x) for x in result[:target_length]])
            f.write(f"{dyck_str}\n")

    print(f"Generated {n} k-Dyck sequences to {out_path}")


def generate_1_dyck_dataset(
    out_path: str,
    n: int = 100000,
    target_length: int = 2048,
    p: float = 0.5,
    min_depth: int = 1,
    max_depth: int = 16,
):
    """Generate 1-Dyck sequences (single bracket type) and save to file."""
    generate_k_dyck_dataset(
        out_path=out_path,
        num_symbols=1,
        n=n,
        target_length=target_length,
        p=p,
        min_depth=min_depth,
        max_depth=max_depth,
    )


def generate_random_dataset(
    out_path: str,
    vocab_size: int = 128,
    n: int = 100000,
    target_length: int = 2048,
):
    """
    Generate random token sequences (k-integer strings).

    Args:
        out_path: Output file path
        vocab_size: Number of unique tokens (k for k-integers)
        n: Number of sequences
        target_length: Sequence length
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w") as f:
        for _ in trange(n, desc=f"Generating random (vocab={vocab_size})"):
            tokens = np.random.randint(0, vocab_size, size=target_length)
            f.write(" ".join(map(str, tokens)) + "\n")

    print(f"Generated {n} random sequences to {out_path}")


def generate_random_text_dataset(
    out_path: str,
    vocab_size: int = 10,
    n: int = 100000,
    target_length: int = 2048,
):
    """
    Generate random integer strings as text (space-separated digits).

    Args:
        out_path: Output file path
        vocab_size: Number of unique symbols (default 10 for digits 0-9)
        n: Number of sequences
        target_length: Sequence length
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w") as f:
        for _ in trange(n, desc=f"Generating random text (vocab={vocab_size})"):
            tokens = np.random.randint(0, vocab_size, size=target_length)
            f.write(" ".join(map(str, tokens)) + "\n")

    print(f"Generated {n} random text sequences to {out_path}")


def make_copy_tokens(
    num_symbols: int = 64,
    min_w_length: int = 10,
    max_w_length: int = 510,
):
    """Generate a copy string ww by duplicating a random sequence."""
    if min_w_length > max_w_length:
        raise ValueError("min_w_length cannot be greater than max_w_length")

    length = random.randint(min_w_length, max_w_length)
    original_token_seq = np.random.randint(0, num_symbols, size=length).tolist()
    return original_token_seq + original_token_seq


def generate_ww_dataset(
    out_path: str,
    num_symbols: int = 64,
    n: int = 100000,
    seq_length: int = 2048,
    min_w_length: int = 10,
):
    """Generate ww sequences (copy task) and save to file."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for _ in trange(n, desc="Generating ww"):
            sequence = make_copy_tokens(num_symbols, min_w_length=min_w_length)
            while len(sequence) < seq_length:
                sequence.extend(make_copy_tokens(num_symbols, min_w_length=min_w_length))
            repeated_str = " ".join(map(str, sequence[:seq_length]))
            f.write(f"{repeated_str}\n")

    print(f"Generated {n} ww sequences to {out_path}")


if __name__ == "__main__":
    fire.Fire({
        "cache_data": cache_data,
        "generate_shuff_dyck": generate_shuff_dyck_dataset,
        "generate_k_dyck": generate_k_dyck_dataset,
        "generate_1_dyck": generate_1_dyck_dataset,
        "generate_random": generate_random_dataset, # random token ids
        "generate_random_text": generate_random_text_dataset, # random digits
        "generate_ww": generate_ww_dataset,
    })
