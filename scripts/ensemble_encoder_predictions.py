#!/usr/bin/env python3
"""Average complete multi-seed probabilities for a pre-specified condition."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from scripts.bootstrap_encoder_comparisons import boolean_series, discover_predictions
except ModuleNotFoundError:  # Direct execution
    from bootstrap_encoder_comparisons import boolean_series, discover_predictions


def build_ensemble(
    frame: pd.DataFrame,
    *,
    model: str,
    view: str,
    max_length: int,
    frozen_encoder: bool,
    expected_seeds: set[int],
) -> pd.DataFrame:
    selected = frame.loc[
        frame["model"].astype(str).eq(model)
        & frame["view"].astype(str).eq(view)
        & frame["max_length"].astype(int).eq(max_length)
        & boolean_series(frame["frozen_encoder"]).eq(frozen_encoder)
    ].copy()
    if selected.empty:
        raise ValueError("No predictions match the requested ensemble condition")
    revisions = selected["model_revision"].astype(str).unique().tolist()
    if len(revisions) != 1:
        raise ValueError(f"Condition contains multiple model revisions: {revisions}")

    key = ["example_id", "language", "source_dataset"]
    seed_sets = selected.groupby(key)["seed"].agg(lambda values: set(map(int, values)))
    incomplete = seed_sets.loc[seed_sets.map(lambda values: values != expected_seeds)]
    if not incomplete.empty:
        raise ValueError(f"Incomplete seed predictions for {len(incomplete)} examples")
    if selected.duplicated([*key, "seed"]).any():
        raise ValueError("Duplicate example/seed predictions found")

    metadata_columns = [
        "example_id",
        "language",
        "source_dataset",
        "split",
        "label",
        "text",
        "text_sha256",
    ]
    metadata = selected.groupby(key, sort=True)[metadata_columns[3:]].first().reset_index()
    for column in ("label", "text", "text_sha256"):
        disagreements = selected.groupby(key)[column].nunique(dropna=False)
        if disagreements.gt(1).any():
            raise ValueError(f"{column} differs across seeds")
    probabilities = selected.groupby(key, sort=True)["probability_1"].mean().rename("probability_1")
    output = metadata.merge(probabilities.reset_index(), on=key, validate="one_to_one")
    output["prediction"] = (output["probability_1"] >= 0.5).astype(int)
    output["model"] = model
    output["model_revision"] = revisions[0]
    output["view"] = view
    output["max_length"] = max_length
    output["frozen_encoder"] = frozen_encoder
    output["seed"] = "ensemble_" + "_".join(map(str, sorted(expected_seeds)))
    output["run_id"] = f"ensemble::{model}::{view}::L{max_length}"
    output["scientifically_valid"] = True
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/encoder"))
    parser.add_argument("--model", default="xlm-roberta-base")
    parser.add_argument("--view", default="surface")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--frozen-encoder", action="store_true")
    parser.add_argument("--expected-seeds", default="42,123,456")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/error_annotation/xlmr_base_surface_ensemble_predictions.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = discover_predictions(args.root)
    seeds = {int(value) for value in args.expected_seeds.split(",") if value}
    ensemble = build_ensemble(
        frame,
        model=args.model,
        view=args.view,
        max_length=args.max_length,
        frozen_encoder=args.frozen_encoder,
        expected_seeds=seeds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ensemble.to_csv(args.output, index=False)
    print(f"Wrote {len(ensemble)} ensemble predictions to {args.output}")


if __name__ == "__main__":
    main()
