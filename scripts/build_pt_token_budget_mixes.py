#!/usr/bin/env python3
import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk


def parse_int_list(text: str) -> list[int]:
    out: list[int] = []
    for x in text.split(","):
        v = x.strip()
        if not v:
            continue
        out.append(int(v))
    if not out:
        raise RuntimeError(f"Empty int list: {text}")
    return out


def as_train_dataset(ds, name: str) -> Dataset:
    if isinstance(ds, DatasetDict):
        if "train" in ds:
            return ds["train"]
        keys = list(ds.keys())
        if len(keys) == 1:
            return ds[keys[0]]
        raise RuntimeError(f"DatasetDict for {name} has no train split: {keys}")
    return ds


def compute_lengths(ds: Dataset, name: str) -> np.ndarray:
    col = ds._data.column("input_ids")
    chunks = []
    for chunk in col.chunks:
        lens = chunk.value_lengths().to_numpy(zero_copy_only=False)
        if lens.size:
            chunks.append(lens.astype(np.int32, copy=False))
    if not chunks:
        raise RuntimeError(f"No input_ids rows found in dataset: {name}")
    out = np.concatenate(chunks)
    if out.shape[0] != len(ds):
        raise RuntimeError(
            f"Length array size mismatch for {name}: {out.shape[0]} vs {len(ds)}"
        )
    return out


def ensure_saved(path: Path) -> bool:
    return (path / "state.json").exists()


def select_without_replacement(lengths: np.ndarray, target_tokens: int, rng: np.random.Generator):
    if target_tokens <= 0:
        return np.empty(0, dtype=np.int64), 0
    perm = rng.permutation(lengths.shape[0])
    cum = np.cumsum(lengths[perm], dtype=np.int64)
    k = int(np.searchsorted(cum, target_tokens, side="left") + 1)
    if k > lengths.shape[0]:
        raise RuntimeError(
            f"Token target {target_tokens} exceeds dataset capacity {int(lengths.sum())}"
        )
    return perm[:k], int(cum[k - 1])


def select_to_token_budget(
    lengths: np.ndarray,
    target_tokens: int,
    rng: np.random.Generator,
    allow_replacement: bool,
):
    if target_tokens <= 0:
        return np.empty(0, dtype=np.int64), 0, False

    total_tokens = int(lengths.sum())
    if target_tokens <= total_tokens:
        idx, tok = select_without_replacement(lengths, target_tokens, rng)
        return idx, tok, False

    if not allow_replacement:
        raise RuntimeError(
            "Noise token target exceeds dataset size and --allow-noise-replacement is not set. "
            f"target={target_tokens}, available={total_tokens}"
        )

    cycle_count = int(target_tokens // total_tokens)
    rem = int(target_tokens % total_tokens)
    parts = []
    for _ in range(cycle_count):
        parts.append(rng.permutation(lengths.shape[0]))

    tokens = cycle_count * total_tokens
    if rem > 0:
        rem_idx, rem_tok = select_without_replacement(lengths, rem, rng)
        parts.append(rem_idx)
        tokens += rem_tok

    if not parts:
        return np.empty(0, dtype=np.int64), 0, True
    return np.concatenate(parts).astype(np.int64, copy=False), int(tokens), True


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build PT mixed datasets with fixed clean token budget and appended noise tokens."
        )
    )
    parser.add_argument("--clean-data", required=True, help="Tokenized clean PT dataset path")
    parser.add_argument("--noise-data", required=True, help="Tokenized noise dataset path")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory where ptmix_<mix-name>_pXXX dirs are written",
    )
    parser.add_argument(
        "--mix-name",
        required=True,
        help="Mix name used in output folder names: ptmix_<mix-name>_pXXX",
    )
    parser.add_argument(
        "--noise-pcts",
        default="20,50,80",
        help="Comma-separated appended-noise percentages relative to fixed clean token budget.",
    )
    parser.add_argument("--base-pt-steps", type=int, default=10000)
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260225)
    parser.add_argument("--allow-noise-replacement", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--clean-indices-in",
        default="",
        help="Optional .npy path of clean indices to reuse (forces same clean subset across runs)",
    )
    parser.add_argument(
        "--clean-indices-out",
        default="",
        help="Optional .npy path to save selected clean indices",
    )
    args = parser.parse_args()

    clean_path = Path(args.clean_data).resolve()
    noise_path = Path(args.noise_data).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not clean_path.exists():
        raise RuntimeError(f"Clean dataset not found: {clean_path}")
    if not noise_path.exists():
        raise RuntimeError(f"Noise dataset not found: {noise_path}")
    if args.base_pt_steps <= 0:
        raise RuntimeError(f"--base-pt-steps must be > 0, got {args.base_pt_steps}")
    if args.bsz <= 0 or args.max_seq_length <= 0:
        raise RuntimeError("--bsz and --max-seq-length must be > 0")

    noise_pcts = sorted(set(parse_int_list(args.noise_pcts)))
    for p in noise_pcts:
        if p < 0 or p > 100:
            raise RuntimeError(f"Noise pct must be in [0,100], got {p}")
    if 0 not in noise_pcts:
        noise_pcts = [0] + noise_pcts

    print(f"Loading clean dataset: {clean_path}")
    clean_ds = as_train_dataset(load_from_disk(str(clean_path)), name="clean")
    print(f"Loading noise dataset: {noise_path}")
    noise_ds = as_train_dataset(load_from_disk(str(noise_path)), name="noise")

    clean_lengths = compute_lengths(clean_ds, name="clean")
    noise_lengths = compute_lengths(noise_ds, name="noise")

    clean_total_tokens = int(clean_lengths.sum())
    noise_total_tokens = int(noise_lengths.sum())
    print(
        "Dataset stats: "
        f"clean_rows={len(clean_ds)}, clean_tokens={clean_total_tokens}; "
        f"noise_rows={len(noise_ds)}, noise_tokens={noise_total_tokens}"
    )

    tokens_per_step = int(args.bsz * args.max_seq_length)
    clean_target_tokens = int(args.base_pt_steps * tokens_per_step)
    if clean_target_tokens > clean_total_tokens:
        raise RuntimeError(
            "Clean token budget exceeds clean dataset capacity. "
            f"target={clean_target_tokens}, available={clean_total_tokens}"
        )

    clean_idx: np.ndarray
    if args.clean_indices_in:
        idx_path = Path(args.clean_indices_in).resolve()
        if not idx_path.exists():
            raise RuntimeError(f"--clean-indices-in not found: {idx_path}")
        clean_idx = np.load(idx_path)
        if clean_idx.ndim != 1:
            raise RuntimeError(f"clean indices must be 1-D array: {idx_path}")
        clean_tokens = int(clean_lengths[clean_idx].sum())
        if clean_tokens < clean_target_tokens:
            raise RuntimeError(
                "Provided clean indices do not satisfy clean token target. "
                f"target={clean_target_tokens}, actual={clean_tokens}"
            )
        print(
            f"Reusing clean indices from {idx_path}: rows={clean_idx.shape[0]}, tokens={clean_tokens}"
        )
    else:
        rng_clean = np.random.default_rng(args.seed)
        clean_idx, clean_tokens = select_without_replacement(clean_lengths, clean_target_tokens, rng_clean)
        print(f"Selected clean subset: rows={clean_idx.shape[0]}, tokens={clean_tokens}")

    if args.clean_indices_out:
        out_idx_path = Path(args.clean_indices_out).resolve()
        out_idx_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_idx_path, clean_idx)
        print(f"Saved clean indices: {out_idx_path}")

    clean_selected = clean_ds.select(clean_idx.tolist())

    summary_rows: list[dict] = []
    for pct in noise_pcts:
        out_dir = output_dir / f"ptmix_{args.mix_name}_p{pct:03d}"
        noise_target_tokens = int(round(clean_target_tokens * (pct / 100.0)))
        formula_steps = int(round(args.base_pt_steps * (1.0 + pct / 100.0)))

        if ensure_saved(out_dir) and not args.overwrite:
            print(f"[skip] Exists: {out_dir}")
            meta_path = out_dir / "build_meta.json"
            if meta_path.exists():
                with meta_path.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
            else:
                meta = {
                    "dataset_dir": str(out_dir),
                    "pt_noise_pct_target": pct,
                }
            meta["status"] = "skipped"
            summary_rows.append(meta)
            continue

        noise_rows_sel = 0
        noise_tokens = 0
        noise_replace = False
        parts = [clean_selected]

        if pct > 0:
            rng_noise = np.random.default_rng(args.seed + 1000 + pct)
            noise_idx, noise_tokens, noise_replace = select_to_token_budget(
                lengths=noise_lengths,
                target_tokens=noise_target_tokens,
                rng=rng_noise,
                allow_replacement=args.allow_noise_replacement,
            )
            noise_rows_sel = int(noise_idx.shape[0])
            parts.append(noise_ds.select(noise_idx.tolist()))

        total_tokens = int(clean_tokens + noise_tokens)
        recommended_steps = int(total_tokens // tokens_per_step)
        if recommended_steps <= 0:
            raise RuntimeError(f"recommended_steps <= 0 for pct={pct}")

        if out_dir.exists() and args.overwrite:
            shutil.rmtree(out_dir)

        print(
            f"[build] mix={args.mix_name} pct={pct} clean_rows={clean_idx.shape[0]} "
            f"noise_rows={noise_rows_sel} clean_tokens={clean_tokens} noise_tokens={noise_tokens} "
            f"formula_steps={formula_steps} recommended_steps={recommended_steps}"
        )
        mixed = concatenate_datasets(parts).shuffle(seed=args.seed + pct * 17 + 1)
        mixed.save_to_disk(str(out_dir))

        meta = {
            "mix_name": args.mix_name,
            "pt_noise_pct_target": int(pct),
            "pt_noise_pct_actual_total": round(
                100.0 * float(noise_tokens) / float(total_tokens) if total_tokens > 0 else 0.0,
                6,
            ),
            "noise_over_clean_pct_actual": round(
                100.0 * float(noise_tokens) / float(clean_tokens) if clean_tokens > 0 else 0.0,
                6,
            ),
            "base_pt_steps": int(args.base_pt_steps),
            "tokens_per_step": int(tokens_per_step),
            "clean_target_tokens": int(clean_target_tokens),
            "clean_selected_rows": int(clean_idx.shape[0]),
            "clean_selected_tokens": int(clean_tokens),
            "noise_target_tokens": int(noise_target_tokens),
            "noise_selected_rows": int(noise_rows_sel),
            "noise_selected_tokens": int(noise_tokens),
            "noise_replacement_used": bool(noise_replace),
            "total_selected_rows": int(clean_idx.shape[0] + noise_rows_sel),
            "total_selected_tokens": int(total_tokens),
            "pt_steps_formula": int(formula_steps),
            "pt_steps_recommended": int(recommended_steps),
            "epoch_ratio_at_formula": float(formula_steps * tokens_per_step / float(total_tokens)),
            "epoch_ratio_at_recommended": float(
                recommended_steps * tokens_per_step / float(total_tokens)
            ),
            "dataset_dir": str(out_dir),
            "clean_data_dir": str(clean_path),
            "noise_data_dir": str(noise_path),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": "done",
        }

        with (out_dir / "build_meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        summary_rows.append(meta)

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty and "pt_noise_pct_target" in summary_df.columns:
        summary_df = summary_df.sort_values("pt_noise_pct_target")
    summary_csv = output_dir / f"ptmix_{args.mix_name}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary: {summary_csv}")


if __name__ == "__main__":
    main()
