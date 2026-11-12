#!/usr/bin/env python3
"""Create a blinded, language-balanced manual QC sheet for Stanza views."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


VIEW_COLUMNS = ["text", "lemma_text", "morph_text", "randomized_morph_text"]
QC_COLUMNS = [
    "tokenization_ok_0_1",
    "lemma_ok_0_1",
    "upos_ok_0_1",
    "features_ok_0_1",
    "social_media_artifact_0_1",
    "annotator_id",
    "notes",
]


def read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("Input must be CSV or Parquet")


def stratified_language_sample(
    group: pd.DataFrame, *, total: int, seed: int
) -> pd.DataFrame:
    grouping = [column for column in ("source_dataset", "split") if column in group.columns]
    if not grouping:
        return group.sample(n=min(total, len(group)), random_state=seed)
    strata = list(group.groupby(grouping, sort=True, dropna=False))
    base, remainder = divmod(total, len(strata))
    sampled_parts: list[pd.DataFrame] = []
    sampled_indices: set[int] = set()
    for position, (key, stratum) in enumerate(strata):
        requested = base + (1 if position < remainder else 0)
        take = min(requested, len(stratum))
        if take:
            key_seed = seed + int(
                hashlib.sha256(str(key).encode()).hexdigest()[:8], 16
            )
            sampled = stratum.sample(n=take, random_state=key_seed)
            sampled_parts.append(sampled)
            sampled_indices.update(sampled.index.tolist())
    sampled = pd.concat(sampled_parts) if sampled_parts else group.iloc[0:0]
    shortfall = min(total, len(group)) - len(sampled)
    if shortfall > 0:
        remaining = group.loc[~group.index.isin(sampled_indices)]
        sampled = pd.concat(
            [sampled, remaining.sample(n=shortfall, random_state=seed + 1)]
        )
    return sampled


def create_qc_sample(
    frame: pd.DataFrame, *, per_language: int, seed: int
) -> pd.DataFrame:
    required = {"example_id", "language", "split", *VIEW_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Morphology data is missing columns: {sorted(missing)}")
    if frame[VIEW_COLUMNS].isna().any().any():
        raise ValueError("Morphology views contain null values")

    parts: list[pd.DataFrame] = []
    for language, group in frame.groupby("language", sort=True):
        parts.append(
            stratified_language_sample(group, total=per_language, seed=seed)
        )
    sampled = pd.concat(parts, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sampled["qc_id"] = [
        "MORPH-"
        + hashlib.sha256(f"{seed}|{index}|{row.example_id}".encode()).hexdigest()[:12]
        for index, row in sampled.iterrows()
    ]
    identity_columns = ["qc_id", "example_id", "language"]
    if "source_dataset" in sampled.columns:
        identity_columns.append("source_dataset")
    identity_columns.extend(["split", *VIEW_COLUMNS])
    output = sampled[identity_columns].copy()
    for column in QC_COLUMNS:
        output[column] = ""
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        default=Path("data/processed/standardized_with_morphology.parquet"),
        nargs="?",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/morphology_qc/morphology_qc_sample.csv"),
    )
    parser.add_argument("--per-language", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.per_language < 1:
        raise ValueError("--per-language must be positive")
    frame = read_frame(args.input)
    sample = create_qc_sample(frame, per_language=args.per_language, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.output, index=False)
    counts = sample.groupby("language").size().to_dict()
    print(f"Wrote {len(sample)} QC rows to {args.output}: {counts}")


if __name__ == "__main__":
    main()
