from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import DTypeLike


def _torch_softmax(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    scaled = logits / temperature
    if scaled.dim() == 1:
        scaled = scaled - torch.max(scaled)
    else:
        scaled = scaled - torch.max(scaled, dim=1, keepdim=True).values
    return torch.softmax(scaled, dim=-1)


def _sample_from_logits_batch(
    logits: torch.Tensor, temperature: float, generator: torch.Generator
) -> torch.Tensor:
    probs = _torch_softmax(logits, temperature)
    return torch.multinomial(
        probs, num_samples=1, replacement=True, generator=generator
    ).squeeze(1)


def estimate_spectral_radius(matrix: np.ndarray, iters: int = 50) -> float:
    if matrix.shape[0] == 0:
        return 0.0
    rng = np.random.default_rng(0)
    v = rng.normal(size=(matrix.shape[0],))
    norm = np.linalg.norm(v)
    if norm == 0:
        return 0.0
    v = v / norm
    for _ in range(iters):
        v = matrix @ v
        norm = np.linalg.norm(v)
        if norm == 0:
            return 0.0
        v = v / norm
    return float(np.linalg.norm(matrix @ v))


def _resolve_torch_dtype(dtype: np.dtype) -> torch.dtype:
    if dtype == np.float32:
        return torch.float32
    if dtype == np.float64:
        return torch.float64
    raise ValueError(f"Unsupported dtype: {dtype}")


class LinearRNN:
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        out_size: Optional[int] = None,
        spectral_radius: Optional[float] = 0.9,
        model_type: str = "linear",
        seed: Optional[int] = None,
        dtype: Optional[DTypeLike] = None,
        device: Optional[str] = None,
    ) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be > 0")
        if spectral_radius is not None and spectral_radius <= 0:
            raise ValueError("spectral_radius must be > 0")
        if model_type not in {"linear", "tanh", "relu"}:
            raise ValueError(f"Unsupported model_type: {model_type}")
        if out_size is not None and out_size > vocab_size:
            raise ValueError("out_size cannot be greater than vocab_size")

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.out_size = out_size or vocab_size
        self.model_type = model_type
        self.rng = np.random.default_rng(seed)
        if dtype is None:
            dtype = np.float32
        self.dtype = np.dtype(dtype)
        self.torch_dtype = _resolve_torch_dtype(self.dtype)

        device = (device or ("cuda" if torch.cuda.is_available() else "cpu")).lower()
        torch_device = torch.device(device)
        if torch_device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but not available")
        self.device = torch_device

        # Initialize with numpy so spectral-radius scaling is straightforward.
        A = self.rng.normal(
            loc=0.0,
            scale=1.0 / math.sqrt(vocab_size),
            size=(hidden_size, vocab_size),
        ).astype(self.dtype)
        W = self.rng.normal(
            loc=0.0,
            scale=1.0 / math.sqrt(hidden_size),
            size=(hidden_size, hidden_size),
        ).astype(self.dtype)
        b = np.zeros((hidden_size,), dtype=self.dtype)
        C = self.rng.normal(
            loc=0.0,
            scale=1.0 / math.sqrt(hidden_size),
            size=(self.out_size, hidden_size),
        ).astype(self.dtype)
        d = np.zeros((self.out_size,), dtype=self.dtype)

        if spectral_radius is not None:
            rho = estimate_spectral_radius(W)
            if rho > 0:
                scale = float(spectral_radius) / rho
                W = (W * scale).astype(self.dtype, copy=False)

        # Move parameters to torch tensors.
        self.A = torch.as_tensor(A, dtype=self.torch_dtype, device=self.device)
        self.W = torch.as_tensor(W, dtype=self.torch_dtype, device=self.device)
        self.b = torch.as_tensor(b, dtype=self.torch_dtype, device=self.device)
        self.C = torch.as_tensor(C, dtype=self.torch_dtype, device=self.device)
        self.d = torch.as_tensor(d, dtype=self.torch_dtype, device=self.device)

        torch_seed = seed if seed is not None else int(self.rng.integers(0, 2**63 - 1))
        self.torch_generator = torch.Generator(device=self.device)
        self.torch_generator.manual_seed(int(torch_seed))

    def _activate(self, h: torch.Tensor) -> torch.Tensor:
        if self.model_type == "linear":
            return h
        if self.model_type == "tanh":
            return torch.tanh(h)
        return F.relu(h)

    def _resolve_generator(
        self, rng: Optional[np.random.Generator | torch.Generator]
    ) -> torch.Generator:
        if rng is None:
            return self.torch_generator
        if isinstance(rng, torch.Generator):
            return rng
        if isinstance(rng, np.random.Generator):
            seed = int(rng.integers(0, 2**63 - 1))
            gen = torch.Generator(device=self.device)
            gen.manual_seed(seed)
            return gen
        raise TypeError("rng must be a numpy or torch generator")

    def generate(
        self,
        seq_length: int,
        temperature: float = 1.0,
        seed_token: Optional[int] = None,
        burn_in: int = 0,
        init_state: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator | torch.Generator] = None,
        approx_k: Optional[int] = None,
    ) -> list[int]:
        if seq_length <= 0:
            raise ValueError("seq_length must be > 0")
        if burn_in < 0:
            raise ValueError("burn_in must be >= 0")

        gen = self._resolve_generator(rng)
        if seed_token is None:
            token = int(
                torch.randint(
                    0, self.vocab_size, (1,), device=self.device, generator=gen
                ).item()
            )
        else:
            token = seed_token
        if not (0 <= token < self.vocab_size):
            raise ValueError("seed_token out of range")

        if approx_k is not None:
            if approx_k <= 0:
                raise ValueError("approx_k must be > 0")
            if approx_k >= self.out_size:
                approx_k = None

        if init_state is None:
            h = torch.zeros(
                (self.hidden_size,), device=self.device, dtype=self.torch_dtype
            )
        else:
            if init_state.shape != (self.hidden_size,):
                raise ValueError("init_state shape mismatch")
            h = torch.as_tensor(
                init_state, device=self.device, dtype=self.torch_dtype
            ).clone()

        outputs: list[int] = []
        total_steps = seq_length + burn_in
        for step in range(total_steps):
            h = self._activate(self.A[:, token] + self.W @ h + self.b)
            if approx_k is None:
                logits = self.C @ h + self.d
                probs = _torch_softmax(logits, temperature)
                token = int(torch.multinomial(probs, 1, generator=gen).item())
            else:
                subset = torch.randperm(
                    self.out_size, device=self.device, generator=gen
                )[:approx_k]
                logits = self.C[subset] @ h + self.d[subset]
                probs = _torch_softmax(logits, temperature)
                choice = torch.multinomial(probs, 1, generator=gen).item()
                token = int(subset[choice].item())
            if step >= burn_in:
                outputs.append(token)

        return outputs

    def generate_batch(
        self,
        num_sequences: int,
        seq_length: int,
        temperature: float = 1.0,
        seed_token: Optional[int] = None,
        burn_in: int = 0,
        init_state: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator | torch.Generator] = None,
        approx_k: Optional[int] = None,
    ) -> list[list[int]]:
        if num_sequences <= 0:
            raise ValueError("num_sequences must be > 0")
        if seq_length <= 0:
            raise ValueError("seq_length must be > 0")
        if burn_in < 0:
            raise ValueError("burn_in must be >= 0")

        gen = self._resolve_generator(rng)
        if seed_token is None:
            tokens = torch.randint(
                0, self.vocab_size, (num_sequences,), device=self.device, generator=gen
            )
        else:
            if not (0 <= seed_token < self.vocab_size):
                raise ValueError("seed_token out of range")
            tokens = torch.full(
                (num_sequences,), seed_token, device=self.device, dtype=torch.int64
            )

        if init_state is None:
            h = torch.zeros(
                (num_sequences, self.hidden_size),
                device=self.device,
                dtype=self.torch_dtype,
            )
        else:
            if init_state.shape == (self.hidden_size,):
                init = torch.as_tensor(
                    init_state, device=self.device, dtype=self.torch_dtype
                )
                h = init.expand(num_sequences, -1).clone()
            elif init_state.shape == (num_sequences, self.hidden_size):
                h = torch.as_tensor(
                    init_state, device=self.device, dtype=self.torch_dtype
                ).clone()
            else:
                raise ValueError("init_state shape mismatch")

        if approx_k is not None:
            if approx_k <= 0:
                raise ValueError("approx_k must be > 0")
            if approx_k >= self.out_size:
                approx_k = None

        outputs = torch.empty(
            (num_sequences, seq_length), device=self.device, dtype=torch.int64
        )
        total_steps = seq_length + burn_in
        for step in range(total_steps):
            h = h @ self.W.T
            h += self.A[:, tokens].T
            h += self.b
            h = self._activate(h)

            if approx_k is None:
                logits = h @ self.C.T + self.d
                tokens = _sample_from_logits_batch(logits, temperature, gen)
            else:
                subset = torch.randperm(
                    self.out_size, device=self.device, generator=gen
                )[:approx_k]
                logits = h @ self.C[subset].T + self.d[subset]
                token_idx = _sample_from_logits_batch(logits, temperature, gen)
                tokens = subset[token_idx]

            if step >= burn_in:
                outputs[:, step - burn_in] = tokens

        return outputs.cpu().tolist()
