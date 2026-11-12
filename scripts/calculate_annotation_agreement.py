#!/usr/bin/env python3
"""Calculate blinded pairwise annotation agreement and disagreement sheets."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

try:
    from scripts.sample_linguistic_errors import PHENOMENON_COLUMNS
except ModuleNotFoundError:  # Direct execution
    from sample_linguistic_errors import PHENOMENON_COLUMNS


CATEGORICAL_COLUMNS = [*PHENOMENON_COLUMNS, "primary_phenomenon"]


def read_annotation(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"annotation_id", *CATEGORICAL_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame["annotation_id"].duplicated().any():
        raise ValueError(f"{path} has duplicate annotation_id values")
    return frame


def safe_kappa(a: pd.Series, b: pd.Series, *, weights: str | None = None) -> float:
    if len(a) == 0:
        return float("nan")
    value = cohen_kappa_score(a, b, weights=weights)
    return float(value) if np.isfinite(value) else float("nan")


def agreement_for_pair(
    a: pd.DataFrame,
    b: pd.DataFrame,
    *,
    label_a: str,
    label_b: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = a.merge(
        b,
        on="annotation_id",
        how="outer",
        suffixes=("_a", "_b"),
        indicator=True,
        validate="one_to_one",
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError(f"{label_a} and {label_b} do not contain identical annotation IDs")

    rows: list[dict[str, object]] = []
    disagreement_masks: list[pd.Series] = []
    for column in CATEGORICAL_COLUMNS:
        values_a = merged[f"{column}_a"].astype(str).str.strip()
        values_b = merged[f"{column}_b"].astype(str).str.strip()
        nonblank = values_a.ne("") & values_b.ne("")
        if column in PHENOMENON_COLUMNS:
            valid_values = {"0", "1", "?"}
            invalid_a = set(values_a[values_a.ne("")]).difference(valid_values)
            invalid_b = set(values_b[values_b.ne("")]).difference(valid_values)
            if invalid_a or invalid_b:
                raise ValueError(
                    f"Invalid values in {column}: {sorted(invalid_a | invalid_b)}"
                )
            decidable = values_a.isin({"0", "1"}) & values_b.isin({"0", "1"})
        else:
            decidable = nonblank
        raw_agreement = (
            float(values_a[nonblank].eq(values_b[nonblank]).mean())
            if nonblank.any()
            else float("nan")
        )
        rows.append(
            {
                "annotator_a": label_a,
                "annotator_b": label_b,
                "column": column,
                "n_both_coded": int(nonblank.sum()),
                "n_decidable": int(decidable.sum()),
                "raw_agreement": raw_agreement,
                "cohen_kappa": safe_kappa(values_a[decidable], values_b[decidable]),
            }
        )
        disagreement_masks.append(nonblank & values_a.ne(values_b))

    disagreement = pd.concat(disagreement_masks, axis=1).any(axis=1)
    keep = ["annotation_id"]
    for shared in ("language", "text"):
        a_name = f"{shared}_a"
        if a_name in merged.columns:
            keep.append(a_name)
    for column in CATEGORICAL_COLUMNS:
        keep.extend([f"{column}_a", f"{column}_b"])
    disagreements = merged.loc[disagreement, keep].copy()
    disagreements.insert(1, "annotator_pair", f"{label_a}__{label_b}")
    return pd.DataFrame(rows), disagreements


def calculate(
    frames: list[pd.DataFrame], labels: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    agreement_parts: list[pd.DataFrame] = []
    disagreement_parts: list[pd.DataFrame] = []
    for first, second in combinations(range(len(frames)), 2):
        agreement, disagreements = agreement_for_pair(
            frames[first],
            frames[second],
            label_a=labels[first],
            label_b=labels[second],
        )
        agreement_parts.append(agreement)
        disagreement_parts.append(disagreements)
    if not agreement_parts:
        raise ValueError("At least two annotation files are required")
    return (
        pd.concat(agreement_parts, ignore_index=True),
        pd.concat(disagreement_parts, ignore_index=True),
    )


def markdown_summary(frame: pd.DataFrame) -> str:
    lines = [
        "| Annotators | Column | N decidable | Raw agreement | Cohen's κ |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in frame.iterrows():
        kappa = "NA" if pd.isna(row["cohen_kappa"]) else f"{row['cohen_kappa']:.3f}"
        raw = "NA" if pd.isna(row["raw_agreement"]) else f"{row['raw_agreement']:.3f}"
        lines.append(
            f"| {row['annotator_a']} / {row['annotator_b']} | {row['column']} | "
            f"{int(row['n_decidable'])} | {raw} | {kappa} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", nargs="+", type=Path)
    parser.add_argument(
        "--labels",
        help="Comma-separated annotator labels; defaults to file stems.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/error_annotation/agreement")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = (
        [value.strip() for value in args.labels.split(",")]
        if args.labels
        else [path.stem for path in args.annotations]
    )
    if len(labels) != len(args.annotations):
        raise ValueError("--labels must contain one label per annotation file")
    frames = [read_annotation(path) for path in args.annotations]
    agreement, disagreements = calculate(frames, labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    agreement.to_csv(args.output_dir / "pairwise_agreement.csv", index=False)
    disagreements.to_csv(args.output_dir / "disagreements_for_adjudication.csv", index=False)
    report = "# Annotation agreement\n\n" + markdown_summary(agreement)
    (args.output_dir / "AGREEMENT.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Disagreements requiring adjudication: {len(disagreements)}")


if __name__ == "__main__":
    main()
