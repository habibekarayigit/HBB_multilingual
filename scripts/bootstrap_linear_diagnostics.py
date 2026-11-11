#!/usr/bin/env python3
"""Paired bootstrap intervals for the L=128 linear diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


COMPARISONS = [
    ("suffix_2_5", "prefix_2_5"),
    ("suffix_2_5", "char_wb_3_5"),
    ("char_wb_3_5", "word_1_2"),
]


def macro_f1_binary(labels: np.ndarray, predictions: np.ndarray) -> float:
    scores: list[float] = []
    for label in (0, 1):
        tp = np.sum((labels == label) & (predictions == label))
        fp = np.sum((labels != label) & (predictions == label))
        fn = np.sum((labels == label) & (predictions != label))
        denominator = 2 * tp + fp + fn
        scores.append(float(2 * tp / denominator) if denominator else 0.0)
    return sum(scores) / 2


def interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("artifacts/linear_diagnostics/l128/predictions.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/linear_diagnostics/l128/bootstrap_comparisons.csv"),
    )
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.predictions)
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    for language, group in frame.groupby("language", sort=True):
        labels_by_id = group.groupby("example_id")["label"].nunique()
        if labels_by_id.max() != 1:
            raise ValueError(f"Gold labels disagree within {language}")
        labels = group.drop_duplicates("example_id").set_index("example_id")["label"]
        predictions = group.pivot(
            index="example_id", columns="model", values="prediction"
        ).loc[labels.index]
        n = len(labels)
        sampled_indices = rng.integers(0, n, size=(args.replicates, n))
        y = labels.to_numpy(dtype=int)
        score_samples: dict[str, np.ndarray] = {}
        for model in predictions.columns:
            pred = predictions[model].to_numpy(dtype=int)
            point = macro_f1_binary(y, pred)
            samples = np.fromiter(
                (
                    macro_f1_binary(y[index], pred[index])
                    for index in sampled_indices
                ),
                dtype=float,
                count=args.replicates,
            )
            score_samples[model] = samples
            low, high = interval(samples)
            rows.append(
                {
                    "language": language,
                    "comparison": model,
                    "estimate": point,
                    "ci_low": low,
                    "ci_high": high,
                    "probability_gt_zero": np.nan,
                    "replicates": args.replicates,
                }
            )
        for left, right in COMPARISONS:
            if left not in predictions or right not in predictions:
                continue
            left_point = macro_f1_binary(y, predictions[left].to_numpy(dtype=int))
            right_point = macro_f1_binary(y, predictions[right].to_numpy(dtype=int))
            differences = score_samples[left] - score_samples[right]
            low, high = interval(differences)
            rows.append(
                {
                    "language": language,
                    "comparison": f"{left} minus {right}",
                    "estimate": left_point - right_point,
                    "ci_low": low,
                    "ci_high": high,
                    "probability_gt_zero": float(np.mean(differences > 0)),
                    "replicates": args.replicates,
                }
            )

    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
