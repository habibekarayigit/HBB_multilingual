#!/usr/bin/env python3
"""Paired bootstrap comparison of L=128 versus full-text linear diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bootstrap_linear_diagnostics import interval, macro_f1_binary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--l128",
        type=Path,
        default=Path("artifacts/linear_diagnostics/l128/predictions.csv"),
    )
    parser.add_argument(
        "--full",
        type=Path,
        default=Path("artifacts/linear_diagnostics/full_text/predictions.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/linear_diagnostics/length_sensitivity_bootstrap.csv"),
    )
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    l128 = pd.read_csv(args.l128).rename(columns={"prediction": "prediction_l128"})
    full = pd.read_csv(args.full).rename(columns={"prediction": "prediction_full"})
    keys = ["example_id", "language", "model", "label"]
    merged = l128[keys + ["prediction_l128"]].merge(
        full[keys + ["prediction_full"]], on=keys, validate="one_to_one"
    )
    rng = np.random.default_rng(args.seed)
    rows = []
    for (language, model), group in merged.groupby(["language", "model"], sort=True):
        labels = group["label"].to_numpy(dtype=int)
        pred_l128 = group["prediction_l128"].to_numpy(dtype=int)
        pred_full = group["prediction_full"].to_numpy(dtype=int)
        n = len(group)
        indices = rng.integers(0, n, size=(args.replicates, n))
        differences = np.fromiter(
            (
                macro_f1_binary(labels[index], pred_l128[index])
                - macro_f1_binary(labels[index], pred_full[index])
                for index in indices
            ),
            dtype=float,
            count=args.replicates,
        )
        estimate = macro_f1_binary(labels, pred_l128) - macro_f1_binary(
            labels, pred_full
        )
        low, high = interval(differences)
        rows.append(
            {
                "language": language,
                "model": model,
                "l128_minus_full_macro_f1": estimate,
                "ci_low": low,
                "ci_high": high,
                "probability_gt_zero": float(np.mean(differences > 0)),
                "changed_predictions": int(np.sum(pred_l128 != pred_full)),
                "n": n,
                "replicates": args.replicates,
            }
        )
    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
