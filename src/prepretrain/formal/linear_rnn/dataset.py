from __future__ import annotations

import json
import os
from typing import Iterable, Iterator, Optional

import numpy as np
import torch
from numpy.typing import DTypeLike

from .linear_rnn import LinearRNN


def load_tokenizer(tokenizer_name: str, add_pad_token: bool = True):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers is required to load tokenizer") from exc
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if add_pad_token:
        tokenizer.add_special_tokens({"pad_token": "<|padding|>"})
    return tokenizer


def resolve_vocab_size(
    vocab_size: Optional[int],
    tokenizer_name: Optional[str],
    add_pad_token: bool = True,
) -> int:
    if vocab_size is not None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        return vocab_size
    if tokenizer_name is None:
        raise ValueError("Either vocab_size or tokenizer_name must be provided")
    tokenizer = load_tokenizer(tokenizer_name, add_pad_token=add_pad_token)
    return len(tokenizer)


def resolve_device(device: Optional[str]) -> str:
    if device is None or device.lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    device = device.lower()
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but not available")
    return device


def _spawn_torch_generators(
    seed: int, num_models: int, device: str
) -> list[torch.Generator]:
    if num_models == 1:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        return [gen]
    seq_seed = np.random.SeedSequence(seed)
    generators: list[torch.Generator] = []
    for child in seq_seed.spawn(num_models):
        child_seed = int(child.generate_state(1, dtype=np.uint64)[0] % (2**63 - 1))
        gen = torch.Generator(device=device)
        gen.manual_seed(child_seed)
        generators.append(gen)
    return generators


def iter_sequences(
    *,
    num_models: int,
    num_sequences_per_model: int,
    vocab_size: Optional[int],
    hidden_size: int,
    seq_length: int,
    temperature: float = 1.0,
    spectral_radius: Optional[float] = 0.9,
    burn_in: int = 0,
    model_type: str = "linear",
    tokenizer_name: Optional[str] = "EleutherAI/pythia-160m",
    seed: Optional[int] = None,
    model_seed: Optional[int] = None,
    sequence_seed: Optional[int] = None,
    dtype: Optional[DTypeLike] = None,
    batch_size: int = 1,
    approx_k: Optional[int] = None,
    device: Optional[str] = None,
) -> Iterator[dict]:
    if num_models <= 0:
        raise ValueError("num_models must be > 0")
    if num_sequences_per_model <= 0:
        raise ValueError("num_sequences_per_model must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    if model_type not in {"linear", "tanh", "relu"}:
        raise ValueError(f"Unsupported model_type: {model_type}")

    vocab_size = resolve_vocab_size(vocab_size, tokenizer_name)
    device = resolve_device(device)

    if model_seed is not None:
        if num_models == 1:
            model_seeds = np.array([model_seed], dtype=np.int64)
        else:
            model_rng = np.random.default_rng(model_seed)
            model_seeds = model_rng.integers(
                0, 2**31 - 1, size=num_models, dtype=np.int64
            )
    else:
        rng = np.random.default_rng(seed)
        model_seeds = rng.integers(0, 2**31 - 1, size=num_models, dtype=np.int64)

    sequence_rngs: list[torch.Generator] | None = None
    if sequence_seed is not None:
        sequence_rngs = _spawn_torch_generators(sequence_seed, num_models, device)

    for model_id, model_seed_value in enumerate(model_seeds):
        model_seed_int = int(model_seed_value)
        rnn = LinearRNN(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            spectral_radius=spectral_radius,
            model_type=model_type,
            seed=model_seed_int,
            dtype=dtype,
            device=device,
        )
        seq_rng: torch.Generator | None = None
        if sequence_rngs is not None:
            seq_rng = sequence_rngs[model_id]
        if batch_size == 1:
            for sequence_id in range(num_sequences_per_model):
                seq = rnn.generate(
                    seq_length=seq_length,
                    temperature=temperature,
                    burn_in=burn_in,
                    rng=seq_rng,
                    approx_k=approx_k,
                )
                yield {
                    "model_id": model_id,
                    "sequence_id": sequence_id,
                    "seed": model_seed_int,
                    "model_type": model_type,
                    "vocab_size": vocab_size,
                    "tokenizer_name": tokenizer_name,
                    "sequence": seq,
                }
        else:
            sequence_id = 0
            while sequence_id < num_sequences_per_model:
                chunk = min(batch_size, num_sequences_per_model - sequence_id)
                batch = rnn.generate_batch(
                    num_sequences=chunk,
                    seq_length=seq_length,
                    temperature=temperature,
                    burn_in=burn_in,
                    rng=seq_rng,
                    approx_k=approx_k,
                )
                for offset, seq in enumerate(batch):
                    yield {
                        "model_id": model_id,
                        "sequence_id": sequence_id + offset,
                        "seed": model_seed_int,
                        "model_type": model_type,
                        "vocab_size": vocab_size,
                        "tokenizer_name": tokenizer_name,
                        "sequence": seq,
                    }
                sequence_id += chunk


def save_jsonl(
    records: Iterable[dict],
    out_path: str,
    *,
    tokenizer=None,
    detokenize: bool = False,
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            if detokenize:
                if tokenizer is None:
                    raise ValueError("tokenizer is required for detokenize")
                record = dict(record)
                record["text"] = tokenizer.decode(record["sequence"])
            f.write(json.dumps(record))
            f.write("\n")


def save_txt(
    records: Iterable[dict],
    out_path: str,
    *,
    tokenizer=None,
    detokenize: bool = False,
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            seq = record["sequence"]
            if detokenize:
                if tokenizer is None:
                    raise ValueError("tokenizer is required for detokenize")
                f.write(tokenizer.decode(seq))
            else:
                f.write(" ".join(str(x) for x in seq))
            f.write("\n")
