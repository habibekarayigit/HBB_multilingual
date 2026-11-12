#!/usr/bin/env python3
"""Expand and optionally execute the GPU experiment matrix.

Without `--execute`, this command is a dry run and writes the exact plan. Runs
with an existing `metrics.csv` are skipped unless `--force` is supplied.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def slug(value: str) -> str:
    return "".join(char if char.isalnum() else "-" for char in value).strip("-")


def is_immutable_revision(value: Any) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]{40,64}", str(value or "")))


def revision_slug(value: str) -> str:
    return slug(value)[:12] or "unknown"


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
                "data": block.get("data", config.get("data")),
                "output_root": block.get("output_root", config["output_root"]),
            }
            if not run["data"]:
                raise ValueError(f"Block {block['name']!r} has no data path")
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
        f"{slug(run['model_name'])}__rev{revision_slug(run['model_revision'])}"
        f"__{run['view']}__seed{run['seed']}__L{run['max_length']}"
        + ("__frozen" if run["frozen_encoder"] else "__full")
    )


def command_for(
    run: dict[str, Any], *, allow_mutable_revision: bool = False
) -> list[str]:
    command = [
        "python",
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
    if allow_mutable_revision:
        command.append("--allow-mutable-revision")
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
        "--allow-mutable-revisions",
        action="store_true",
        help="Permit tags such as main. Never use this for final paper runs.",
    )
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
    expected_count = config.get("expected_run_count")
    if expected_count is not None and len(runs) != int(expected_count):
        raise ValueError(
            f"Matrix expands to {len(runs)} runs, expected {int(expected_count)}"
        )
    if args.block:
        requested = set(args.block)
        runs = [run for run in runs if run["block"] in requested]
        missing = requested.difference({run["block"] for run in runs})
        if missing:
            raise ValueError(f"Unknown or empty blocks: {sorted(missing)}")

    plan = []
    for index, run in enumerate(runs, start=1):
        identifier = run_id(run)
        run_dir = Path(run["output_root"]) / identifier
        required_outputs = [
            run_dir / "run_config.json",
            run_dir / "history.json",
            run_dir / "metrics.csv",
            run_dir / "predictions.csv",
        ]
        plan.append(
            {
                "index": index,
                "run_id": identifier,
                "block": run["block"],
                "command": command_for(
                    run, allow_mutable_revision=args.allow_mutable_revisions
                ),
                "run_dir": str(run_dir),
                "metrics_path": str(run_dir / "metrics.csv"),
                "required_outputs": [str(path) for path in required_outputs],
            }
        )
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    args.plan.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    unpinned = sorted(
        {
            f"{run['model_name']}@{run.get('model_revision')}"
            for run in runs
            if not is_immutable_revision(run.get("model_revision"))
        }
    )
    if unpinned:
        message = (
            "Pin immutable Hugging Face commit revisions before final paper runs: "
            + ", ".join(unpinned)
        )
        if args.execute and not args.allow_mutable_revisions:
            raise ValueError(message)
        print("WARNING: " + message, file=sys.stderr)

    print(f"Planned {len(plan)} runs; plan written to {args.plan}")
    for item in plan:
        print(f"[{item['index']:02d}/{len(plan):02d}] {item['block']}: {item['run_id']}")
        if not args.execute:
            print("  " + " ".join(item["command"]))

    if not args.execute:
        return

    for item in plan:
        required_outputs = [Path(path) for path in item["required_outputs"]]
        complete = all(path.exists() for path in required_outputs)
        if complete and not args.force:
            print(f"SKIP completed: {item['run_id']}")
            continue
        missing = [str(path) for path in required_outputs if not path.exists()]
        if missing and any(path.exists() for path in required_outputs):
            print(f"RERUN partial output for {item['run_id']}; missing: {missing}")
        else:
            print(f"RUN: {item['run_id']}")
        execution_command = [sys.executable, *item["command"][1:]]
        subprocess.run(execution_command, check=True)


if __name__ == "__main__":
    main()
