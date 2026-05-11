# Formal-Language Pre-Pretraining

Code for reproducing our pre-pretraining experiments: first expose a randomly initialized language model to structured formal-language data, then continue pretraining on clean or noisy natural-language data.

## Methods

| Method | Description | Display name |
|---|---|---|
| `baseline` | Random initialization, then C4 or noisy-C4 PT only. | Baseline |
| `rnn` / `rnn1000` | RNN-generated formal-language PPT, then PT. | RNN-PPT |
| `shuffdyck` | Bracket-language PPT, then PT. | Dyck |

## Reproduced Results

### Main C4 Slice

Model: `EleutherAI/pythia-160m`. Results are mean final C4 validation losses over 3 seeds. Lower is better.

| Noise | Baseline | RNN-PPT | Dyck |
|---:|---:|---:|---:|
| 0% | 3.6269 | **3.6012** | 3.6176 |
| 10% | 3.6554 | **3.6331** | 3.6452 |
| 30% | 3.7156 | **3.6881** | 3.7069 |
| 50% | 3.8111 | **3.7736** | 3.7961 |

## Repository Layout

```text
formal-language-prepretraining/
  configs/
    main_c4.env                    # 160M main reproduction defaults
    paperlike_1b.env               # tuned 1B compact reproduction defaults
    smoke.env                      # tiny code-path test defaults
  results_expected/
    main_ci_summary.csv            # 160M expected summary
    paperlike_1b_0_15_summary.csv  # 1B expected summary
  scripts/
    00_prepare_c4.sh               # C4 eval, C4-PPT, and skip PT split
    01_generate_formal_data.sh     # RNN and Dyck PPT data
    02_build_noise_mixes.sh        # noisy PT datasets
    03_run_main_c4.sh              # sequential 160M run
    04_run_smoke.sh                # tiny end-to-end check
    05_run_main_c4_parallel.sh     # parallel 160M run
    06_run_main_c4_10k_with_ntfy.sh
    07_run_paperlike_1b_0_15.py
    08_run_paperlike_1b_0_15_with_ntfy.sh
    analyze_main_results.py
  src/prepretrain/
    train.py                       # training entry point
    formal/                        # formal-language generators
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install flash-attn --no-build-isolation
```

If `flash-attn` is not available, set `ATTN_IMPLEMENTATION=eager` for smoke tests or small debugging runs. Full-scale runs are intended for CUDA GPUs.

## Quick Smoke Test

This creates tiny local random datasets and runs a few steps. It only verifies that the code path works.

```bash
bash scripts/04_run_smoke.sh
```

Output:

```text
outputs/smoke/
```

## Reproduce the 160M Main Slice

Load defaults:

```bash
set -a
source configs/main_c4.env
set +a
```

Prepare C4 and the canonical PT skip split:

```bash
bash scripts/00_prepare_c4.sh
```

Generate RNN and Dyck PPT datasets:

```bash
bash scripts/01_generate_formal_data.sh
```

Build noisy PT mixes:

```bash
bash scripts/02_build_noise_mixes.sh
```

Run the 3-seed `0/10/30/50%` experiment on 4 GPUs:

```bash
GPUS="0 1 2 3" bash scripts/06_run_main_c4_10k_with_ntfy.sh
```

Output:

```text
outputs/main_c4_10k/metrics/final_c4_by_experiment.csv
outputs/main_c4_10k/metrics/summary_by_method_noise.csv
```

The `ntfy` wrapper sends a notification when the job finishes. If `ntfy` is not installed, the script still prints the final status.

## Reproduce the Tuned 1B Slice

The 1B experiment intentionally uses the tuned paper-like hyperparameters rather than the 160M defaults.

| Setting | Value |
|---|---|
| Model | `EleutherAI/pythia-1b` |
| PPT step | `1000` |
| PT steps | `25000` |
| Batch | `bsz=16`, `gradient_accumulation_steps=2` |
| LR | `5e-4` |
| Warmup | `1000` |
| min_lr_rate | `0.1` |
| weight_decay | `0.1` |
| Noise | `0`, `15` |

Run:

```bash
set -a
source configs/paperlike_1b.env
set +a
GPUS="0 1 2 3" bash scripts/08_run_paperlike_1b_0_15_with_ntfy.sh
```

Output:

```text
outputs/paperlike_1b_0_15/metrics/results_1b_0_15.csv
outputs/paperlike_1b_0_15/metrics/summary_1b_0_15.csv
```

If you already have step-aligned PPT checkpoints, set `PPT_SOURCE_ROOT` to the directory containing runs like:

```text
rnn1000_1b_align_ppt1000_sd3407/checkpoint-1000
shuff_1b_align_ppt1000_sd3407/checkpoint-1000
c4_1b_align_ppt1000_sd3407/checkpoint-1000
```

If `PPT_SOURCE_ROOT` is unset, the script trains the required PPT sources first.

## Canonical C4 Split Rule

Use non-overlapping C4 data for PPT and PT.

## Practical Notes

Data and checkpoints are large. The `.gitignore` excludes `data/`, `outputs/`, `wandb/`, and local environments.

The training scripts do not keep GPU memory alive after completion. If your cluster uses a keep-alive process, it can take the GPUs back after training exits.

For Weights and Biases logging, set `REPORT_TO=wandb`; release defaults use `REPORT_TO=none`.

## Citation

Citation information will be added after the paper metadata is finalized.
