#!/usr/bin/env python3
"""Audit the two local datasets and create leakage-safe split manifests.

The script never writes message text to the generated reports/manifests.  It
uses source row IDs and SHA-256 hashes so that the manifests can be shared
without redistributing the datasets.

Example
-------
python scripts/audit_and_build_splits.py \
  --atc "ATC (2).xlsx" \
  --finnish "RTP_LX_FI (1).xlsx" \
  --output-dir artifacts/audit
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def normalize_text(value: Any) -> str:
    """Return a conservative form used only for duplicate grouping.

    NFKC, Unicode case-folding, zero-width-character removal, and whitespace
    collapsing catch formatting duplicates without aggressively replacing
    users, URLs, punctuation, or words.
    """

    text = unicodedata.normalize("NFKC", "" if pd.isna(value) else str(value))
    text = re.sub(r"[\u200b-\u200d\ufeff]", "", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def text_hash(normalized_text: str) -> str:
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _python_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def value_counts(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts(dropna=False).sort_index()
    return {str(_python_scalar(key)): int(value) for key, value in counts.items()}


def allocate_binary_counts(size: int, positive_rate: float) -> dict[int, int]:
    positives = int(round(size * positive_rate))
    return {0: size - positives, 1: positives}


def build_stratified_manifest(
    frame: pd.DataFrame,
    *,
    row_id_col: str,
    label_col: str,
    split_sizes: dict[str, int],
    seed: int,
    hash_col: str | None = None,
) -> pd.DataFrame:
    """Sample disjoint rows with approximately equal class rates per split."""

    labels = set(frame[label_col].astype(int).unique().tolist())
    if labels != {0, 1}:
        raise ValueError(f"Expected binary labels {{0, 1}}, found {sorted(labels)}")

    positive_rate = float(frame[label_col].astype(int).mean())
    rng = random.Random(seed)
    pools: dict[int, list[Any]] = {}
    indexed = frame.set_index(row_id_col, drop=False)
    for label in (0, 1):
        ids = frame.loc[frame[label_col].astype(int).eq(label), row_id_col].tolist()
        rng.shuffle(ids)
        pools[label] = ids

    records: list[dict[str, Any]] = []
    offsets = {0: 0, 1: 0}
    for split, size in split_sizes.items():
        requested = allocate_binary_counts(size, positive_rate)
        for label in (0, 1):
            start = offsets[label]
            stop = start + requested[label]
            chosen = pools[label][start:stop]
            if len(chosen) != requested[label]:
                raise ValueError(
                    f"Not enough class-{label} rows for split {split}: "
                    f"requested {requested[label]}, got {len(chosen)}"
                )
            offsets[label] = stop
            for row_id in chosen:
                source = indexed.loc[row_id]
                record = {
                    "row_id": _python_scalar(row_id),
                    "label": int(label),
                    "split": split,
                    "seed": int(seed),
                }
                if hash_col is not None:
                    record["text_sha256"] = str(source[hash_col])
                records.append(record)

    manifest = pd.DataFrame.from_records(records)
    # A source row may never cross splits.
    if manifest["row_id"].duplicated().any():
        raise AssertionError("A source row was allocated more than once")
    if hash_col is not None and manifest["text_sha256"].duplicated().any():
        raise AssertionError("A normalized-text group was allocated more than once")
    if manifest.groupby("row_id")["split"].nunique().max() != 1:
        raise AssertionError("A source row crosses split boundaries")
    return manifest.sort_values(["split", "label", "row_id"]).reset_index(drop=True)


def audit_atc(path: Path, seed: int) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = pd.read_excel(path)
    required = {"message", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"ATC is missing columns: {sorted(missing)}")

    frame = frame.reset_index(names="row_id")
    frame["normalized_text"] = frame["message"].map(normalize_text)
    frame["text_sha256"] = frame["normalized_text"].map(text_hash)
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)

    blank = frame["normalized_text"].eq("")
    exact_duplicates = int(frame.duplicated(subset=["message", "label"]).sum())
    duplicate_messages = int(frame.duplicated(subset=["message"]).sum())
    normalized_duplicates = int(frame.duplicated(subset=["normalized_text"]).sum())

    group_label_counts = frame.loc[~blank].groupby("normalized_text")["label"].nunique()
    conflicting_groups = set(group_label_counts[group_label_counts.gt(1)].index)
    conflicting_rows = frame["normalized_text"].isin(conflicting_groups)

    eligible = (
        frame.loc[~blank & ~conflicting_rows]
        .drop_duplicates(subset=["normalized_text"], keep="first")
        .copy()
    )

    manifest = build_stratified_manifest(
        eligible,
        row_id_col="row_id",
        label_col="label",
        split_sizes={"train": 8000, "dev": 1000, "test": 1000},
        seed=seed,
        hash_col="text_sha256",
    )

    report = {
        "path": str(path),
        "sha256": file_hash(path),
        "rows": int(len(frame)),
        "class_counts": value_counts(frame["label"]),
        "blank_normalized_text_rows": int(blank.sum()),
        "exact_duplicate_rows_including_label": exact_duplicates,
        "duplicate_message_rows": duplicate_messages,
        "duplicate_normalized_text_rows": normalized_duplicates,
        "conflicting_normalized_text_groups": int(len(conflicting_groups)),
        "rows_in_conflicting_groups": int(conflicting_rows.sum()),
        "eligible_unique_nonconflicting_rows": int(len(eligible)),
        "eligible_class_counts": value_counts(eligible["label"]),
        "proposed_manifest_rows": int(len(manifest)),
        "proposed_split_class_counts": manifest.groupby(["split", "label"]).size().unstack(fill_value=0).astype(int).to_dict(orient="index"),
    }
    return report, manifest


def _parse_rating_list(value: Any) -> list[float]:
    if isinstance(value, list):
        return [float(item) for item in value]
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a list of ratings, got: {value!r}")
    return [float(item) for item in parsed]


def audit_finnish(
    path: Path, seed: int, toxicity_threshold: float
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    frame = pd.read_excel(path).reset_index(names="row_id")
    required = {
        "Completion",
        "BenignCompletion",
        "CompletionAnnotations.Toxicity",
        "CompletionAnnotationsRaw.Toxicity",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Finnish RTP-LX is missing columns: {sorted(missing)}")

    score = pd.to_numeric(frame["CompletionAnnotations.Toxicity"], errors="raise")
    # The distributed aggregate column contains integers.  Therefore >=3.5 is
    # operationally identical to >=4.0; we retain the requested threshold in
    # metadata to make that choice explicit.
    frame["label"] = score.ge(toxicity_threshold).astype(int)

    manifest = build_stratified_manifest(
        frame,
        row_id_col="row_id",
        label_col="label",
        split_sizes={"train": 560, "dev": 99, "test": 440},
        seed=seed,
    )

    raw_ratings = frame["CompletionAnnotationsRaw.Toxicity"].map(_parse_rating_list)
    raw_means = raw_ratings.map(lambda values: sum(values) / len(values))

    thresholds = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    threshold_counts = {
        str(threshold): int(score.ge(threshold).sum()) for threshold in thresholds
    }
    raw_mean_threshold_counts = {
        str(threshold): int(raw_means.ge(threshold).sum()) for threshold in thresholds
    }

    positive_rows = frame.loc[frame["label"].eq(1), ["row_id"]].copy()
    paired_records: list[dict[str, Any]] = []
    for row_id in positive_rows["row_id"].tolist():
        paired_records.extend(
            [
                {
                    "pair_id": int(row_id),
                    "row_id": int(row_id),
                    "variant": "toxic_completion",
                    "text_column": "Completion",
                    "label": 1,
                },
                {
                    "pair_id": int(row_id),
                    "row_id": int(row_id),
                    "variant": "benign_completion",
                    "text_column": "BenignCompletion",
                    "label": 0,
                },
            ]
        )
    paired_manifest = pd.DataFrame.from_records(paired_records)

    report = {
        "path": str(path),
        "sha256": file_hash(path),
        "rows": int(len(frame)),
        "aggregate_toxicity_score_counts": value_counts(score),
        "binary_threshold": float(toxicity_threshold),
        "binary_class_counts": value_counts(frame["label"]),
        "binary_positive_rate": float(frame["label"].mean()),
        "positive_counts_by_aggregate_threshold": threshold_counts,
        "positive_counts_by_raw_mean_threshold": raw_mean_threshold_counts,
        "duplicate_completions": int(frame["Completion"].duplicated().sum()),
        "duplicate_benign_completions": int(frame["BenignCompletion"].duplicated().sum()),
        "completion_character_length": {
            "median": float(frame["Completion"].astype(str).str.len().median()),
            "mean": float(frame["Completion"].astype(str).str.len().mean()),
            "max": int(frame["Completion"].astype(str).str.len().max()),
        },
        "benign_completion_character_length": {
            "median": float(frame["BenignCompletion"].astype(str).str.len().median()),
            "mean": float(frame["BenignCompletion"].astype(str).str.len().mean()),
            "max": int(frame["BenignCompletion"].astype(str).str.len().max()),
        },
        "proposed_split_class_counts": manifest.groupby(["split", "label"]).size().unstack(fill_value=0).astype(int).to_dict(orient="index"),
        "paired_counterfactual_rows": int(len(paired_manifest)),
        "paired_counterfactual_pairs": int(len(positive_rows)),
    }
    return report, manifest, paired_manifest


def manifest_summary(manifest: pd.DataFrame) -> str:
    table = manifest.groupby(["split", "label"]).size().unstack(fill_value=0)
    lines = [
        "| split | non_abusive_0 | abusive_1 | total |",
        "|---|---:|---:|---:|",
    ]
    for split, row in table.iterrows():
        negative = int(row.get(0, 0))
        positive = int(row.get(1, 0))
        lines.append(f"| {split} | {negative} | {positive} | {negative + positive} |")
    return "\n".join(lines)


def render_markdown(report: dict[str, Any], atc_manifest: pd.DataFrame, fi_manifest: pd.DataFrame) -> str:
    atc = report["atc"]
    fi = report["finnish_rtp_lx"]
    return f"""# Initial dataset audit

Generated by `scripts/audit_and_build_splits.py`. No message text is included in this report.

## ATC

- Rows in the supplied workbook: **{atc['rows']:,}**
- Class counts: `{atc['class_counts']}`
- Exact duplicate rows (message + label): **{atc['exact_duplicate_rows_including_label']:,}**
- Repeated exact messages: **{atc['duplicate_message_rows']:,}**
- Repeated conservatively normalized messages: **{atc['duplicate_normalized_text_rows']:,}**
- Conflicting normalized-text groups: **{atc['conflicting_normalized_text_groups']:,}** ({atc['rows_in_conflicting_groups']:,} rows)
- Blank normalized texts: **{atc['blank_normalized_text_rows']:,}**
- Unique, nonblank, nonconflicting rows eligible for splitting: **{atc['eligible_unique_nonconflicting_rows']:,}**

### Proposed ATC split (deduplicated and leakage-safe)

{manifest_summary(atc_manifest)}

The supplied workbook has {atc['rows']:,} rows and class counts {atc['class_counts']}; these values do not match the manuscript's 30,354 / 10,528 / 19,826 figures. Resolve the source version before reporting final results.

## Finnish RTP-LX

- Rows: **{fi['rows']:,}**
- Aggregate score counts: `{fi['aggregate_toxicity_score_counts']}`
- Binary rule audited: aggregate toxicity **>= {fi['binary_threshold']}**
- Resulting class counts: `{fi['binary_class_counts']}`
- Resulting positive rate: **{100 * fi['binary_positive_rate']:.1f}%**
- Positive counts under alternative aggregate thresholds: `{fi['positive_counts_by_aggregate_threshold']}`
- Median completion length: **{fi['completion_character_length']['median']:.0f} characters** (maximum {fi['completion_character_length']['max']:,})
- Median benign-completion length: **{fi['benign_completion_character_length']['median']:.0f} characters**

### Proposed Finnish split

{manifest_summary(fi_manifest)}

The aggregate score column is integer-valued, so `>=3.5` is identical to `>=4`. In the supplied workbook this yields **{fi['binary_class_counts'].get('1', 0)} positives**, not 157. The number 157 is approximately the positive count in a 40% test split, not the positive count in the full 1,099-row dataset. The manuscript's Finnish dataset and split tables must therefore be recomputed.

## Paired Finnish analysis opportunity

For every completion meeting the threshold, the workbook also contains a `BenignCompletion`. The generated paired manifest contains **{fi['paired_counterfactual_pairs']} matched pairs** ({fi['paired_counterfactual_rows']} texts). These are useful as a paired diagnostic, but they are not morphology-controlled minimal pairs: completion lengths differ substantially. Pairs must stay together whenever adaptation/train/dev/test partitions are made, and length-matched sensitivity analysis is required.

## Required action before another submission

1. Freeze dataset file hashes and split manifests.
2. Deduplicate/group before splitting; never split and then deduplicate.
3. Justify one Finnish threshold *a priori* and report sensitivity at neighboring thresholds.
4. Re-run every table from a clean environment and retain per-example predictions.
5. Generate paper tables from result files rather than copying values by hand.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atc", type=Path, default=Path("ATC (2).xlsx"))
    parser.add_argument(
        "--finnish", type=Path, default=Path("RTP_LX_FI (1).xlsx")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/audit"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--finnish-threshold", type=float, default=3.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    atc_report, atc_manifest = audit_atc(args.atc, args.seed)
    fi_report, fi_manifest, fi_pairs = audit_finnish(
        args.finnish, args.seed, args.finnish_threshold
    )
    report = {
        "audit_version": 1,
        "seed": int(args.seed),
        "atc": atc_report,
        "finnish_rtp_lx": fi_report,
    }

    atc_manifest.to_csv(
        args.output_dir / f"atc_seed{args.seed}_manifest.csv", index=False
    )
    fi_manifest.to_csv(
        args.output_dir / f"finnish_seed{args.seed}_manifest.csv", index=False
    )
    fi_pairs.to_csv(args.output_dir / "finnish_paired_counterfactual_manifest.csv", index=False)
    (args.output_dir / "data_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "data_audit.md").write_text(
        render_markdown(report, atc_manifest, fi_manifest), encoding="utf-8"
    )
    print(f"Wrote audit and manifests to {args.output_dir}")


if __name__ == "__main__":
    main()
