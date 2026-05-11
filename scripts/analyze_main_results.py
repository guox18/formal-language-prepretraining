#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import pandas as pd


RUN_RE = re.compile(r"^(?P<method>[a-z0-9_]+)_n(?P<noise>\d{3})_sd(?P<seed>\d+)_pt$")


def final_eval_loss(run_dir: Path) -> tuple[int | None, float | None]:
    history_path = run_dir / "history.json"
    if history_path.exists():
        data = json.loads(history_path.read_text(encoding="utf-8"))
        steps = data.get("step", [])
        losses = data.get("eval_loss", [])
        if steps and losses:
            return int(steps[-1]), float(losses[-1])
    states = sorted(run_dir.glob("checkpoint-*/trainer_state.json"), key=lambda p: int(p.parent.name.split("-")[-1]))
    if states:
        state = json.loads(states[-1].read_text(encoding="utf-8"))
        eval_entries = [x for x in state.get("log_history", []) if "eval_loss" in x]
        if eval_entries:
            last = eval_entries[-1]
            return int(last.get("step", 0)), float(last["eval_loss"])
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate main C4-noise runs")
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        match = RUN_RE.match(run_dir.name)
        if not match:
            continue
        final_step, final_loss = final_eval_loss(run_dir)
        if final_loss is None:
            continue
        rows.append(
            {
                "exp_id": run_dir.name,
                "method_id": match.group("method"),
                "source_noise_pct": int(match.group("noise")),
                "seed": int(match.group("seed")),
                "final_step": final_step,
                "final_c4_loss": final_loss,
            }
        )

    if not rows:
        raise RuntimeError(f"No finished PT runs found under {runs_dir}")

    df = pd.DataFrame(rows).sort_values(["source_noise_pct", "seed", "method_id"])
    base = df[df.method_id == "baseline"][["source_noise_pct", "seed", "final_c4_loss"]]
    base = base.rename(columns={"final_c4_loss": "baseline_c4_loss"})
    df = df.merge(base, on=["source_noise_pct", "seed"], how="left")
    df["delta_vs_baseline"] = df["baseline_c4_loss"] - df["final_c4_loss"]
    df.to_csv(out_dir / "final_c4_by_experiment.csv", index=False)

    summary = (
        df.groupby(["source_noise_pct", "method_id"])
        .agg(
            n=("final_c4_loss", "count"),
            mean_final_c4_loss=("final_c4_loss", "mean"),
            std_final_c4_loss=("final_c4_loss", "std"),
            mean_delta_vs_baseline=("delta_vs_baseline", "mean"),
            std_delta_vs_baseline=("delta_vs_baseline", "std"),
        )
        .reset_index()
        .sort_values(["source_noise_pct", "mean_final_c4_loss"])
    )
    summary.to_csv(out_dir / "summary_by_method_noise.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
