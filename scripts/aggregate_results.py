"""
Aggregate experiment results from multiple runs.

Supports two data sources:
- history.json: Custom training history file
- trainer_state.json: HuggingFace Trainer state file (in checkpoint directories)
"""
import json
import os
from pathlib import Path

import fire
import pandas as pd


def _load_from_history_json(subdir: Path) -> list:
    """Load results from history.json file."""
    history_file = subdir / "history.json"
    if not history_file.exists():
        return []

    with open(history_file) as f:
        history = json.load(f)

    results = []
    for i, step in enumerate(history.get("step", [])):
        results.append({
            "step": step,
            "train_loss": history.get("train_loss", [None])[i] if i < len(history.get("train_loss", [])) else None,
            "eval_loss": history.get("eval_loss", [None])[i] if i < len(history.get("eval_loss", [])) else None,
        })
    return results


def _load_from_trainer_state(subdir: Path) -> list:
    """Load results from trainer_state.json in checkpoint directories."""
    checkpoints = [d for d in subdir.iterdir() if d.is_dir() and d.name.startswith('checkpoint-')]
    if not checkpoints:
        return []

    # Get the latest checkpoint
    checkpoints.sort(key=lambda x: int(x.name.split('-')[1]))
    state_file = checkpoints[-1] / "trainer_state.json"

    if not state_file.exists():
        return []

    with open(state_file) as f:
        state = json.load(f)

    results = []
    for entry in state.get('log_history', []):
        if 'eval_loss' in entry:
            results.append({
                "step": entry['step'],
                "train_loss": entry.get('loss'),
                "eval_loss": entry['eval_loss']
            })
    return results


def aggregate_results(output_dir: str, out_csv: str = None):
    """
    Aggregate results from all experiment directories.

    Automatically detects and loads from history.json or trainer_state.json.

    Args:
        output_dir: Directory containing experiment outputs
        out_csv: Output CSV path
    """
    output_dir = Path(output_dir)

    all_results = []

    for subdir in output_dir.iterdir():
        if not subdir.is_dir():
            continue

        # Determine experiment type from directory name
        name = subdir.name
        if "_ppt" in name:
            exp_type = "ppt"
            exp_name = name.replace("_ppt", "")
        elif "_c4" in name:
            exp_type = "c4"
            exp_name = name.replace("_c4", "")
        else:
            exp_type = "unknown"
            exp_name = name

        # Try loading from different sources
        results = _load_from_history_json(subdir)
        if not results:
            results = _load_from_trainer_state(subdir)

        # Add experiment metadata to each result
        for r in results:
            all_results.append({
                "experiment": exp_name,
                "stage": exp_type,
                **r
            })

    if not all_results:
        print("No results found!")
        return

    df = pd.DataFrame(all_results)
    df = df.sort_values(["experiment", "stage", "step"])

    if out_csv:
        df.to_csv(out_csv, index=False)
        print(f"Saved {len(all_results)} records to {out_csv}")

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)

    # Get final results for each experiment/stage
    final_results = {}
    for exp_name in df["experiment"].unique():
        exp_df = df[df["experiment"] == exp_name]

        # PPT final
        ppt_df = exp_df[exp_df["stage"] == "ppt"]
        if not ppt_df.empty:
            final_results[(exp_name, "ppt")] = ppt_df.iloc[-1]

        # C4 final
        c4_df = exp_df[exp_df["stage"] == "c4"]
        if not c4_df.empty:
            final_results[(exp_name, "c4")] = c4_df.iloc[-1]

    # Print C4 results with comparison to baseline
    print("\nFinal C4 Validation Loss:")
    print("-" * 40)

    baseline_loss = None
    if ("baseline", "c4") in final_results:
        baseline_loss = final_results[("baseline", "c4")]["eval_loss"]

    c4_results = [(k, v) for k, v in final_results.items() if k[1] == "c4"]
    c4_results.sort(key=lambda x: x[1]["eval_loss"])

    for (exp_name, _), data in c4_results:
        loss = data["eval_loss"]
        if baseline_loss is not None:
            diff = loss - baseline_loss
            sign = "+" if diff > 0 else ""
            marker = "better" if diff < 0 else ("worse" if diff > 0 else "")
            print(f"{exp_name:20s}: {loss:.4f}  ({sign}{diff:.4f}) {marker}")
        else:
            print(f"{exp_name:20s}: {loss:.4f}")

    return df


if __name__ == "__main__":
    fire.Fire(aggregate_results)
