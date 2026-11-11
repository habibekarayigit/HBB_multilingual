#!/usr/bin/env python3
"""Generate manuscript-ready dataset tables from the standardized registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DISPLAY_NAMES = {
    "ATC": "Turkish",
    "BEEP": "Korean",
    "RTP-LX-FI": "Finnish",
    "Hungarian-Toxic-Comments": "Hungarian",
    "TweetEval-Offensive": "English",
}

SOURCE_NAMES = {
    "ATC": "ATC workbook",
    "BEEP": "BEEP! train+dev",
    "RTP-LX-FI": "RTP-LX FI",
    "Hungarian-Toxic-Comments": "Hungarian Toxic Comments",
    "TweetEval-Offensive": "TweetEval Offensive",
}

SPLIT_LABELS = {
    "train": "Train",
    "dev": "Dev",
    "adaptation": "Adaptation pool",
    "test": "Test",
}


def percentage(value: float) -> str:
    return f"{100 * value:.1f}"


def table_line(cells: list[Any], numeric_from: int = 0) -> str:
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("artifacts/datasets/dataset_registry.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/datasets/PAPER_DATA_TABLES.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    summary = pd.DataFrame(registry["summary"])
    summary = summary.loc[~summary["source_dataset"].eq("RTP-LX-FI-PAIRED")].copy()

    overview_lines = [
        "| Language | Source used | Rows in protocol | Negative (0) | Positive (1) | Positive % |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for dataset in DISPLAY_NAMES:
        group = summary.loc[summary["source_dataset"].eq(dataset)]
        if group.empty:
            continue
        negative = int(group["negative_0"].sum())
        positive = int(group["positive_1"].sum())
        total = negative + positive
        overview_lines.append(
            table_line(
                [
                    DISPLAY_NAMES[dataset],
                    SOURCE_NAMES[dataset],
                    f"{total:,}",
                    f"{negative:,}",
                    f"{positive:,}",
                    percentage(positive / total),
                ]
            )
        )

    split_lines = [
        "| Language | Role/split | Total | Negative (0) | Positive (1) | Positive % |",
        "|---|---|---:|---:|---:|---:|",
    ]
    dataset_order = list(DISPLAY_NAMES)
    split_order = {"train": 0, "dev": 1, "adaptation": 2, "test": 3}
    for dataset in dataset_order:
        group = summary.loc[summary["source_dataset"].eq(dataset)].copy()
        if group.empty:
            continue
        group["_order"] = group["split"].map(split_order).fillna(99)
        for _, row in group.sort_values("_order").iterrows():
            role = SPLIT_LABELS.get(row["split"], row["split"])
            if dataset == "BEEP" and row["split"] == "test":
                role = "Test (public BEEP dev)"
            total = int(row["n"])
            split_lines.append(
                table_line(
                    [
                        DISPLAY_NAMES[dataset],
                        role,
                        f"{total:,}",
                        f"{int(row['negative_0']):,}",
                        f"{int(row['positive_1']):,}",
                        percentage(float(row["positive_rate"])),
                    ]
                )
            )

    hungarian_note = (
        "Hungarian is intentionally omitted until its exact source file is supplied and hashed."
        if "HU" not in registry["languages"]
        else "The Hungarian source file and hash are recorded in the registry."
    )
    content = f"""# Corrected dataset tables for the revised manuscript

Generated from `{args.registry}` by `scripts/make_dataset_tables.py`. {hungarian_note}

## Corpus/source overview

{chr(10).join(overview_lines)}

The previous manuscript's English total of 11,916 was only the official training split, whereas the Korean total combined train and development. The revised table applies a consistent definition.

## Fixed study splits

{chr(10).join(split_lines)}

## Important reporting qualifications

- The Korean public test labels are undisclosed. The revised protocol uses the repository's labeled development set as a transparent held-out test and names it explicitly.
- TweetEval contains normalized-text duplicates crossing official splits. Keep official splits for benchmark comparability, but add a deduplicated sensitivity analysis before few-shot/monolingual claims.
- The Finnish test condition is heavily truncated at 128 tokens; see `artifacts/tokenization/REPORT.md`.
- Do not add a language row until its exact dataset file, license, label semantics, and split manifest are available.
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
