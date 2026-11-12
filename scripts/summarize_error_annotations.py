#!/usr/bin/env python3
"""Summarize adjudicated linguistic phenomena by language and error outcome."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from scripts.sample_linguistic_errors import PHENOMENON_COLUMNS
except ModuleNotFoundError:  # Direct execution
    from sample_linguistic_errors import PHENOMENON_COLUMNS


def summarize(annotations: pd.DataFrame, key: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_annotations = {"annotation_id", *PHENOMENON_COLUMNS, "primary_phenomenon"}
    missing = required_annotations.difference(annotations.columns)
    if missing:
        raise ValueError(f"Annotations are missing columns: {sorted(missing)}")
    required_key = {"annotation_id", "language", "outcome", "label", "prediction"}
    missing_key = required_key.difference(key.columns)
    if missing_key:
        raise ValueError(f"Annotation key is missing columns: {sorted(missing_key)}")
    if annotations["annotation_id"].duplicated().any() or key["annotation_id"].duplicated().any():
        raise ValueError("annotation_id must be unique in both files")

    merged = annotations.merge(
        key,
        on="annotation_id",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("Annotations and key do not contain identical annotation IDs")

    rows: list[dict[str, object]] = []
    for (language, outcome), group in merged.groupby(["language", "outcome"], sort=True):
        for phenomenon in PHENOMENON_COLUMNS:
            values = group[phenomenon].astype(str).str.strip()
            invalid = set(values[values.ne("")]).difference({"0", "1", "?"})
            if invalid:
                raise ValueError(f"Invalid values in {phenomenon}: {sorted(invalid)}")
            decidable = values.isin({"0", "1"})
            present = values.eq("1")
            rows.append(
                {
                    "language": language,
                    "outcome": outcome,
                    "phenomenon": phenomenon,
                    "n_sampled": int(len(group)),
                    "n_decidable": int(decidable.sum()),
                    "n_present": int(present.sum()),
                    "prevalence": float(present.sum() / decidable.sum())
                    if decidable.any()
                    else float("nan"),
                }
            )
    prevalence = pd.DataFrame(rows)
    primary = (
        merged.assign(
            primary_phenomenon=merged["primary_phenomenon"].astype(str).str.strip().replace("", "UNSPECIFIED")
        )
        .groupby(["language", "outcome", "primary_phenomenon"], sort=True)
        .size()
        .rename("n")
        .reset_index()
    )
    return prevalence, primary


def markdown_table(frame: pd.DataFrame) -> str:
    lines = [
        "| Language | Outcome | Phenomenon | N decidable | Prevalence |",
        "|---|---|---|---:|---:|",
    ]
    for _, row in frame.iterrows():
        prevalence = "NA" if pd.isna(row["prevalence"]) else f"{row['prevalence']:.3f}"
        lines.append(
            f"| {row['language']} | {row['outcome']} | {row['phenomenon']} | "
            f"{int(row['n_decidable'])} | {prevalence} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("key", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/error_annotation/summary")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotations = pd.read_csv(args.annotations, dtype=str, keep_default_na=False)
    key = pd.read_csv(args.key, dtype=str, keep_default_na=False)
    prevalence, primary = summarize(annotations, key)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prevalence.to_csv(args.output_dir / "phenomenon_prevalence.csv", index=False)
    primary.to_csv(args.output_dir / "primary_phenomenon_counts.csv", index=False)
    with pd.ExcelWriter(args.output_dir / "linguistic_error_analysis.xlsx") as writer:
        prevalence.to_excel(writer, sheet_name="prevalence", index=False)
        primary.to_excel(writer, sheet_name="primary_counts", index=False)
    report = "# Linguistic error analysis\n\n" + markdown_table(prevalence)
    (args.output_dir / "LINGUISTIC_ERROR_RESULTS.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
