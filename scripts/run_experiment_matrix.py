#!/usr/bin/env python3
"""Expand and optionally execute the GPU experiment matrix.

Without `--execute`, this command is a dry run and writes the exact plan. Runs
with an existing `metrics.csv` are skipped unless `--force` is supplied.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def slug(value: str) -> str:
    return "".join(char if char.isalnum() else "-" for char in value).strip("-")


def expand_matrix(config: dict[str, Any]) -> list[dict[str, Any]]:
    defaults = config["defaults"]
    seeds = config["seeds"]
    runs: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for block in config["blocks"]:
        for model, view, max_length, seed in itertools.product(
            block["models"], block["views"], block["max_lengths"], seeds
        ):
            run = {
                **defaults,
                **model,
                "block": block["name"],
                "view": view,
                "max_length": int(max_length),
                "seed": int(seed),
                "frozen_encoder": bool(block.get("frozen_encoder", False)),
                "data": config["data"],
                "output_root": config["output_root"],
            }
            key = (
                run["model_name"],
                run["model_revision"],
                run["view"],
                run["max_length"],
                run["seed"],
                run["frozen_encoder"],
            )
            if key in seen:
                continue
            seen.add(key)
            runs.append(run)
    return runs


def run_id(run: dict[str, Any]) -> str:
    return (
        f"{slug(run['model_name'])}__{run['view']}__seed{run['seed']}"
        f"__L{run['max_length']}"
        + ("__frozen" if run["frozen_encoder"] else "__full")
    )


def command_for(run: dict[str, Any]) -> list[str]:
    command = [
        sys.executable,
        "scripts/run_encoder_experiment.py",
        "--data",
        run["data"],
        "--model-name",
        run["model_name"],
        "--model-revision",
        run["model_revision"],
        "--view",
        run["view"],
        "--seed",
        str(run["seed"]),
        "--max-length",
        str(run["max_length"]),
        "--epochs",
        str(run["epochs"]),
        "--learning-rate",
        str(run["learning_rate"]),
        "--weight-decay",
        str(run["weight_decay"]),
        "--warmup-ratio",
        str(run["warmup_ratio"]),
        "--train-batch-size",
        str(run["train_batch_size"]),
        "--eval-batch-size",
        str(run["eval_batch_size"]),
        "--gradient-accumulation-steps",
        str(run["gradient_accumulation_steps"]),
        "--output-root",
        run["output_root"],
    ]
    if run["frozen_encoder"]:
        command.append("--frozen-encoder")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiment_matrix.json")
    )
    parser.add_argument("--plan", type=Path, default=Path("artifacts/experiment_plan.json"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--block",
        action="append",
        help="Run only named block(s); may be supplied multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    runs = expand_matrix(config)
    if args.block:
        requested = set(args.block)
        runs = [run for run in runs if run["block"] in requested]
        missing = requested.difference({run["block"] for run in runs})
        if missing:
            raise ValueError(f"Unknown or empty blocks: {sorted(missing)}")

    plan = []
    for index, run in enumerate(runs, start=1):
        identifier = run_id(run)
        metric_path = Path(run["output_root"]) / identifier / "metrics.csv"
        plan.append(
            {
                "index": index,
                "run_id": identifier,
                "block": run["block"],
                "command": command_for(run),
                "metrics_path": str(metric_path),
            }
        )
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    args.plan.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    unpinned = sorted(
        {
            run["model_name"]
            for run in runs
            if run.get("model_revision") in {None, "", "main"}
        }
    )
    if unpinned:
        print(
            "WARNING: pin immutable Hugging Face commit revisions before final paper runs: "
            + ", ".join(unpinned),
            file=sys.stderr,
        )

    print(f"Planned {len(plan)} runs; plan written to {args.plan}")
    for item in plan:
        print(f"[{item['index']:02d}/{len(plan):02d}] {item['block']}: {item['run_id']}")
        if not args.execute:
            print("  " + " ".join(item["command"]))

    if not args.execute:
        return

    for item in plan:
        metric_path = Path(item["metrics_path"])
        if metric_path.exists() and not args.force:
            print(f"SKIP completed: {item['run_id']}")
            continue
        print(f"RUN: {item['run_id']}")
        subprocess.run(item["command"], check=True)


if __name__ == "__main__":
    main()
