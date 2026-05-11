#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue

import pandas as pd


@dataclass
class Task:
    stage: str
    exp_id: str
    method_id: str
    noise_pct: int
    source_model_path: Path | None
    data_dir: Path
    eval_data_dir: Path
    output_dir: Path
    log_path: Path


def parse_space_list(text: str) -> list[str]:
    out = [x.strip() for x in text.replace(",", " ").split() if x.strip()]
    if not out:
        raise RuntimeError(f"empty list: {text}")
    return out


def read_history(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "history.json"
    if not path.exists():
        return pd.DataFrame(columns=["step", "eval_loss"])
    data = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame({"step": data.get("step", []), "eval_loss": data.get("eval_loss", [])})
    if not df.empty:
        df["step"] = df["step"].astype(int)
    return df


def final_and_best(run_dir: Path) -> tuple[tuple[int | None, float | None], tuple[int | None, float | None]]:
    df = read_history(run_dir)
    if df.empty:
        return (None, None), (None, None)
    df = df[df["eval_loss"].notna()].sort_values("step")
    if df.empty:
        return (None, None), (None, None)
    final = (int(df.iloc[-1]["step"]), float(df.iloc[-1]["eval_loss"]))
    best_row = df.loc[df["eval_loss"].idxmin()]
    return final, (int(best_row["step"]), float(best_row["eval_loss"]))


def run_command(cmd: list[str], env: dict[str, str], log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write("$ " + shlex.join(cmd) + "\n")
        f.flush()
        subprocess.run(cmd, check=True, env=env, stdout=f, stderr=f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tuned paper-like 1B reproduction at noise 0/15")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--python", default="")
    parser.add_argument("--model-name", default=os.environ.get("MODEL_NAME", "EleutherAI/pythia-1b"))
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--methods", default=os.environ.get("METHODS", "baseline rnn1000 shuffdyck c4ppt"))
    parser.add_argument("--noise-pcts", default=os.environ.get("NOISE_PCTS", "0 15"))
    parser.add_argument("--gpus", default=os.environ.get("GPUS", "0 1 2 3"))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "3407")))
    parser.add_argument("--ppt-step", type=int, default=int(os.environ.get("PPT_STEP", "1000")))
    parser.add_argument("--pt-steps", type=int, default=int(os.environ.get("PT_STEPS", "25000")))
    parser.add_argument("--save-steps", type=int, default=int(os.environ.get("SAVE_STEPS", "5000")))
    parser.add_argument("--bsz", type=int, default=int(os.environ.get("BSZ", "16")))
    parser.add_argument("--gradient-accumulation-steps", type=int, default=int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "2")))
    parser.add_argument("--warmup-steps", type=int, default=int(os.environ.get("WARMUP_STEPS", "1000")))
    parser.add_argument("--lr", default=os.environ.get("LR", "5e-4"))
    parser.add_argument("--min-lr-rate", default=os.environ.get("MIN_LR_RATE", "0.1"))
    parser.add_argument("--weight-decay", default=os.environ.get("WEIGHT_DECAY", "0.1"))
    parser.add_argument("--eval-steps", type=int, default=int(os.environ.get("EVAL_STEPS", "100")))
    parser.add_argument("--eval-samples", type=int, default=int(os.environ.get("EVAL_SAMPLES", "10000")))
    parser.add_argument("--max-seq-length", type=int, default=int(os.environ.get("MAX_SEQ_LENGTH", "2048")))
    parser.add_argument("--report-to", default=os.environ.get("REPORT_TO", "none"))
    parser.add_argument("--ppt-source-root", default=os.environ.get("PPT_SOURCE_ROOT", ""))
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not args.python:
        args.python = str(root / ".." / "pre-pretraining" / ".venv" / "bin" / "python")
    data_dir = Path(args.data_dir).resolve() if args.data_dir else root / "data" / "tokenized"
    out_dir = Path(args.out_dir).resolve() if args.out_dir else root / "outputs" / "paperlike_1b_0_15"
    train_module = "prepretrain.train"
    eval_data = data_dir / "c4_val"
    methods = parse_space_list(args.methods)
    noise_pcts = [int(x) for x in parse_space_list(args.noise_pcts)]
    gpus = [int(x) for x in parse_space_list(args.gpus)]

    if "baseline" not in methods:
        methods = ["baseline"] + methods
    valid_methods = {"baseline", "rnn1000", "shuffdyck", "c4ppt"}
    bad = sorted(set(methods) - valid_methods)
    if bad:
        raise RuntimeError(f"Unknown method(s): {bad}")

    data_map = {
        "rnn1000": data_dir / "rnn_m1000",
        "shuffdyck": data_dir / "shuffdyck_k128_d8",
        "c4ppt": data_dir / "c4_train",
    }
    prefix_map = {"rnn1000": "rnn1000_1b_align", "shuffdyck": "shuff_1b_align", "c4ppt": "c4_1b_align"}

    for p in [eval_data] + [data_map[m] for m in methods if m != "baseline"]:
        if not p.exists():
            raise RuntimeError(f"Missing required dataset: {p}")
    for pct in noise_pcts:
        pt_data = data_dir / f"ptmix_1b_p{pct:03d}"
        if not pt_data.exists():
            raise RuntimeError(f"Missing PT dataset for noise={pct}: {pt_data}")

    runs_dir = out_dir / "runs"
    logs_dir = out_dir / "logs"
    metrics_dir = out_dir / "metrics"
    for p in [runs_dir, logs_dir, metrics_dir]:
        p.mkdir(parents=True, exist_ok=True)

    ppt_sources: dict[str, Path] = {}
    source_root = Path(args.ppt_source_root).resolve() if args.ppt_source_root else None
    for method in methods:
        if method == "baseline":
            continue
        external = None
        if source_root is not None:
            external = source_root / f"{prefix_map[method]}_ppt{args.ppt_step}_sd{args.seed}" / f"checkpoint-{args.ppt_step}"
        if external is not None and (external / "model.safetensors").exists():
            ppt_sources[method] = external
        else:
            ppt_sources[method] = runs_dir / f"{method}_ppt{args.ppt_step}_sd{args.seed}" / f"checkpoint-{args.ppt_step}"

    ppt_tasks: list[Task] = []
    for method, source_path in ppt_sources.items():
        if (source_path / "model.safetensors").exists():
            continue
        run_dir = source_path.parent
        ppt_tasks.append(
            Task(
                stage="ppt",
                exp_id=run_dir.name,
                method_id=method,
                noise_pct=-1,
                source_model_path=None,
                data_dir=data_map[method],
                eval_data_dir=eval_data,
                output_dir=run_dir,
                log_path=logs_dir / f"{run_dir.name}.log",
            )
        )

    pt_tasks: list[Task] = []
    for noise_pct in noise_pcts:
        pt_data = data_dir / f"ptmix_1b_p{noise_pct:03d}"
        for method in methods:
            exp_id = f"paperlike1b_{method}_n{noise_pct:03d}_sd{args.seed}_pt{args.pt_steps}"
            src = None if method == "baseline" else ppt_sources[method]
            pt_tasks.append(
                Task(
                    stage="pt",
                    exp_id=exp_id,
                    method_id=method,
                    noise_pct=noise_pct,
                    source_model_path=src,
                    data_dir=pt_data,
                    eval_data_dir=eval_data,
                    output_dir=runs_dir / exp_id,
                    log_path=logs_dir / f"{exp_id}.log",
                )
            )

    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    status_rows: list[dict[str, object]] = []
    lock = threading.Lock()

    def run_task(t: Task, gpu: int) -> dict[str, object]:
        final_marker = t.output_dir / "final" / "model.safetensors"
        ppt_marker = t.output_dir / f"checkpoint-{args.ppt_step}" / "model.safetensors"
        marker = ppt_marker if t.stage == "ppt" else final_marker
        if args.resume and marker.exists():
            return {"stage": t.stage, "exp_id": t.exp_id, "method_id": t.method_id, "noise_pct": t.noise_pct, "gpu": gpu, "status": "skipped", "error": "", "elapsed_seconds": 0.0}

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        src_path = str(root / "src")
        env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        if t.stage == "ppt":
            cmd = [
                args.python, "-m", train_module,
                "--data_dir", str(t.data_dir),
                "--model_name", args.model_name,
                "--reinit", "True",
                "--max_steps", str(args.ppt_step),
                "--save_steps", str(args.ppt_step),
                "--bsz", str(args.bsz),
                "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
                "--warmup_steps", str(args.warmup_steps),
                "--lr", args.lr,
                "--min_lr_rate", args.min_lr_rate,
                "--weight_decay", args.weight_decay,
                "--eval_data_dir", str(t.eval_data_dir),
                "--eval_steps", str(args.eval_steps),
                "--eval_samples", str(args.eval_samples),
                "--output_dir", str(t.output_dir),
                "--seed", str(args.seed),
                "--report_to", args.report_to,
                "--override_packing", "True",
                "--max_seq_length", str(args.max_seq_length),
            ]
        else:
            cmd = [
                args.python, "-m", train_module,
                "--data_dir", str(t.data_dir),
                "--model_name", args.model_name,
                "--max_steps", str(args.pt_steps),
                "--save_steps", str(args.save_steps),
                "--bsz", str(args.bsz),
                "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
                "--warmup_steps", str(args.warmup_steps),
                "--lr", args.lr,
                "--min_lr_rate", args.min_lr_rate,
                "--weight_decay", args.weight_decay,
                "--eval_data_dir", str(t.eval_data_dir),
                "--eval_steps", str(args.eval_steps),
                "--eval_samples", str(args.eval_samples),
                "--output_dir", str(t.output_dir),
                "--seed", str(args.seed),
                "--report_to", args.report_to,
                "--max_seq_length", str(args.max_seq_length),
            ]
            if t.method_id == "baseline":
                cmd.extend(["--reinit", "True"])
            else:
                if t.source_model_path is None or not (t.source_model_path / "model.safetensors").exists():
                    raise RuntimeError(f"Missing source model for {t.exp_id}: {t.source_model_path}")
                cmd.extend(["--model_path", str(t.source_model_path)])
        t0 = time.time()
        try:
            run_command(cmd, env, t.log_path)
            return {"stage": t.stage, "exp_id": t.exp_id, "method_id": t.method_id, "noise_pct": t.noise_pct, "gpu": gpu, "status": "done", "error": "", "elapsed_seconds": round(time.time() - t0, 3)}
        except Exception as exc:
            return {"stage": t.stage, "exp_id": t.exp_id, "method_id": t.method_id, "noise_pct": t.noise_pct, "gpu": gpu, "status": "failed", "error": f"{type(exc).__name__}: {exc}", "elapsed_seconds": round(time.time() - t0, 3)}

    def run_stage(tasks: list[Task], stage: str) -> None:
        q: Queue[Task] = Queue()
        for task in tasks:
            q.put(task)

        def worker(gpu: int) -> None:
            while True:
                try:
                    task = q.get_nowait()
                except Empty:
                    return
                row = run_task(task, gpu)
                with lock:
                    status_rows.append(row)
                q.task_done()

        threads = []
        for gpu in gpus:
            th = threading.Thread(target=worker, args=(gpu,), daemon=True)
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
        status = pd.DataFrame(status_rows)
        status.to_csv(metrics_dir / "job_status.csv", index=False)
        if not status.empty and ((status["stage"] == stage) & (status["status"] == "failed")).any():
            raise SystemExit(1)

    if ppt_tasks:
        run_stage(ppt_tasks, "ppt")
    run_stage(pt_tasks, "pt")

    rows = []
    for task in pt_tasks:
        (final_step, final_loss), (best_step, best_loss) = final_and_best(task.output_dir)
        rows.append({"exp_id": task.exp_id, "method_id": task.method_id, "noise_pct": task.noise_pct, "final_step": final_step, "final_c4_loss": final_loss, "best_step": best_step, "best_c4_loss": best_loss})
    df = pd.DataFrame(rows).sort_values(["noise_pct", "method_id"])
    all_rows = []
    summary_rows = []
    for noise_pct in sorted(df["noise_pct"].unique().tolist()):
        cur = df[df["noise_pct"] == noise_pct].copy()
        baseline = float(cur[cur["method_id"] == "baseline"].iloc[0]["final_c4_loss"])
        cur["baseline_c4_loss"] = baseline
        cur["delta_vs_baseline"] = baseline - cur["final_c4_loss"]
        all_rows.extend(cur.to_dict("records"))
        ranking = " > ".join(cur.sort_values("final_c4_loss")["method_id"].tolist())
        row = {"noise_pct": int(noise_pct), "baseline_c4_loss": baseline, "ranking": ranking}
        for r in cur.itertuples(index=False):
            row[f"{r.method_id}_final_c4_loss"] = r.final_c4_loss
            row[f"{r.method_id}_delta_vs_baseline"] = r.delta_vs_baseline
        summary_rows.append(row)
    pd.DataFrame(all_rows).to_csv(metrics_dir / "results_1b_0_15.csv", index=False)
    summary = pd.DataFrame(summary_rows).sort_values("noise_pct")
    summary.to_csv(metrics_dir / "summary_1b_0_15.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
