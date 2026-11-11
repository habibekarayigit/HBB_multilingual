#!/usr/bin/env python3
"""Create a blinded, balanced sample for qualitative error annotation.

Input columns (CSV or Parquet):
    example_id, language, text, label, prediction
Optional columns:
    model, probability_1

The script writes two files:
- annotation_blind.csv: text and empty linguistic coding fields
- annotation_key.csv: gold/prediction/outcome key, kept away from annotators
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


PHENOMENON_COLUMNS = [
    "inflectional_suffix",
    "derivational_suffix",
    "negation",
    "quotation_or_reported_speech",
    "target_is_explicit",
    "profanity_without_abuse",
    "obfuscation",
    "compound_or_spacing",
    "irony_or_reclaimed_use",
    "annotation_ambiguous",
    "domain_or_cultural_reference",
]


def read_predictions(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("Predictions must be CSV or Parquet")


def outcome_name(label: int, prediction: int) -> str:
    return {
        (0, 0): "TN",
        (0, 1): "FP",
        (1, 0): "FN",
        (1, 1): "TP",
    }[(int(label), int(prediction))]


def create_sample(
    frame: pd.DataFrame, *, per_cell: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {"example_id", "language", "text", "label", "prediction"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")

    work = frame.copy()
    work["label"] = pd.to_numeric(work["label"], errors="raise").astype(int)
    work["prediction"] = pd.to_numeric(work["prediction"], errors="raise").astype(int)
    if not set(work["label"].unique()).issubset({0, 1}):
        raise ValueError("label must be binary")
    if not set(work["prediction"].unique()).issubset({0, 1}):
        raise ValueError("prediction must be binary")

    work["outcome"] = [
        outcome_name(label, prediction)
        for label, prediction in zip(work["label"], work["prediction"])
    ]

    sampled_parts: list[pd.DataFrame] = []
    availability: list[dict[str, object]] = []
    for (language, outcome), group in work.groupby(["language", "outcome"], sort=True):
        take = min(len(group), per_cell)
        availability.append(
            {
                "language": language,
                "outcome": outcome,
                "available": int(len(group)),
                "sampled": int(take),
            }
        )
        if take:
            sampled_parts.append(group.sample(n=take, random_state=seed))

    if not sampled_parts:
        raise ValueError("No rows were available for sampling")
    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sampled["annotation_id"] = [
        "ANN-" + hashlib.sha256(f"{seed}|{index}|{row.example_id}".encode()).hexdigest()[:12]
        for index, row in sampled.iterrows()
    ]

    blind = sampled[["annotation_id", "language", "text"]].copy()
    blind["annotator_id"] = ""
    for column in PHENOMENON_COLUMNS:
        blind[column] = ""
    blind["primary_phenomenon"] = ""
    blind["confidence_1_to_5"] = ""
    blind["notes"] = ""

    key_columns = [
        "annotation_id",
        "example_id",
        "language",
        "label",
        "prediction",
        "outcome",
    ]
    for optional in ("model", "probability_1"):
        if optional in sampled.columns:
            key_columns.append(optional)
    key = sampled[key_columns].copy()
    availability_frame = pd.DataFrame(availability)
    return blind, key, availability_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/error_annotation"))
    parser.add_argument("--per-cell", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        help="If the input contains several models, retain only this model name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = read_predictions(args.predictions)
    if args.model is not None:
        if "model" not in frame.columns:
            raise ValueError("--model was supplied but the input has no model column")
        frame = frame.loc[frame["model"].astype(str).eq(args.model)].copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    blind, key, availability = create_sample(
        frame, per_cell=args.per_cell, seed=args.seed
    )
    blind.to_csv(args.output_dir / "annotation_blind.csv", index=False)
    key.to_csv(args.output_dir / "annotation_key.csv", index=False)
    availability.to_csv(args.output_dir / "sampling_availability.csv", index=False)
    print(f"Wrote {len(blind)} blinded items to {args.output_dir}")


if __name__ == "__main__":
    main()
