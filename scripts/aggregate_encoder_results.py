#!/usr/bin/env python3
"""Aggregate complete encoder runs into mean±SD tables.

Scientifically invalid smoke-test rows are excluded by default. Conditions that
do not contain every expected seed cause a hard failure unless
`--allow-incomplete` is supplied.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = [
    "accuracy",
    "macro_f1",
    "f1_abusive",
    "macro_precision",
    "macro_recall",
    "mcc",
    "false_positive_rate",
    "false_negative_rate",
]

GROUP_COLUMNS = [
    "model",
    "view",
    "max_length",
    "frozen_encoder",
    "language",
    "source_dataset",
    "split",
]


def discover_metrics(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("**/metrics.csv"))
    if not files:
        raise FileNotFoundError(f"No metrics.csv files found under {root}")
    parts = []
    for path in files:
        frame = pd.read_csv(path)
        frame["metrics_path"] = str(path)
        parts.append(frame)
    combined = pd.concat(parts, ignore_index=True)
    if "run_id" in combined and combined["run_id"].duplicated().any():
        # One run_id legitimately has one row per language. Duplicate the full
        # condition/language key is not legitimate.
        key = ["run_id", "language", "source_dataset", "split"]
        duplicates = combined.duplicated(key, keep=False)
        if duplicates.any():
            paths = combined.loc[duplicates, "metrics_path"].unique().tolist()
            raise ValueError(f"Duplicate run/language metrics found in {paths}")
    return combined


def aggregate(
    frame: pd.DataFrame,
    *,
    expected_seeds: set[int],
    allow_incomplete: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = set(GROUP_COLUMNS + METRICS + ["seed", "scientifically_valid"])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Metrics files are missing columns: {sorted(missing)}")
    valid = frame.loc[frame["scientifically_valid"].astype(bool)].copy()
    if valid.empty:
        raise ValueError("No scientifically valid rows found")

    completeness_rows = []
    for key, group in valid.groupby(GROUP_COLUMNS, dropna=False, sort=True):
        present = set(group["seed"].astype(int).tolist())
        missing_seeds = sorted(expected_seeds.difference(present))
        extra_seeds = sorted(present.difference(expected_seeds))
        completeness_rows.append(
            dict(zip(GROUP_COLUMNS, key))
            | {
                "present_seeds": ",".join(map(str, sorted(present))),
                "missing_seeds": ",".join(map(str, missing_seeds)),
                "extra_seeds": ",".join(map(str, extra_seeds)),
                "complete": not missing_seeds and not extra_seeds,
            }
        )
    completeness = pd.DataFrame(completeness_rows)
    if not allow_incomplete and not completeness["complete"].all():
        incomplete = completeness.loc[~completeness["complete"]]
        raise ValueError(
            "Incomplete seed sets detected:\n" + incomplete.to_string(index=False)
        )

    grouped = valid.groupby(GROUP_COLUMNS, dropna=False, sort=True)
    mean = grouped[METRICS].mean().add_suffix("_mean")
    std = grouped[METRICS].std(ddof=1).add_suffix("_std")
    count = grouped["seed"].nunique().rename("seed_count")
    summary = pd.concat([mean, std, count], axis=1).reset_index()
    return summary, completeness


def markdown_table(summary: pd.DataFrame) -> str:
    lines = [
        "| Model | View | L | Frozen | Language | N seeds | Macro-F1 mean±SD | Accuracy mean±SD |",
        "|---|---|---:|:---:|---|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        macro_std = row["macro_f1_std"]
        accuracy_std = row["accuracy_std"]
        macro = f"{row['macro_f1_mean']:.3f}±{macro_std:.3f}" if pd.notna(macro_std) else f"{row['macro_f1_mean']:.3f}"
        accuracy = f"{row['accuracy_mean']:.3f}±{accuracy_std:.3f}" if pd.notna(accuracy_std) else f"{row['accuracy_mean']:.3f}"
        lines.append(
            f"| {row['model']} | {row['view']} | {int(row['max_length'])} | "
            f"{'yes' if row['frozen_encoder'] else 'no'} | {row['language']} | "
            f"{int(row['seed_count'])} | {macro} | {accuracy} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/encoder"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/results"))
    parser.add_argument("--expected-seeds", default="42,123,456")
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected_seeds = {int(value) for value in args.expected_seeds.split(",") if value}
    frame = discover_metrics(args.root)
    summary, completeness = aggregate(
        frame,
        expected_seeds=expected_seeds,
        allow_incomplete=args.allow_incomplete,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "encoder_summary.csv", index=False)
    completeness.to_csv(args.output_dir / "seed_completeness.csv", index=False)
    (args.output_dir / "ENCODER_RESULTS.md").write_text(
        "# Encoder results\n\n" + markdown_table(summary), encoding="utf-8"
    )
    print(markdown_table(summary))


if __name__ == "__main__":
    main()
