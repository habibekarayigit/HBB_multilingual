#!/usr/bin/env python3
"""Prepare standardized, leakage-aware datasets for the revised study.

Raw/processed message text is written below `data/`, which is intentionally
ignored by Git. Shareable row-ID/hash manifests and aggregate summaries are
written below `artifacts/datasets/`.

The script can prepare:
- Turkish ATC from the local workbook and audited split manifest;
- Finnish RTP-LX from the local workbook and audited split manifest;
- Korean BEEP! from a pinned local checkout of kocohub/korean-hate-speech;
- English TweetEval Offensive from a pinned local checkout;
- optional Hungarian data supplied as CSV/XLSX.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from audit_and_build_splits import build_stratified_manifest, file_hash, normalize_text, text_hash


KO_COMMIT = "f8d05dce2b22007bb149e5139c0060c68ad8f94b"
EN_COMMIT = "4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66"


def git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def public_repo_metadata(path: Path, repository: str, expected_commit: str) -> dict[str, Any]:
    actual = git_head(path)
    if actual is not None and actual != expected_commit:
        raise ValueError(
            f"{repository} is at {actual}, expected pinned commit {expected_commit}. "
            f"Run: git -C {path} checkout {expected_commit}"
        )
    return {
        "repository": repository,
        "expected_commit": expected_commit,
        "actual_commit": actual,
        "pinned_commit_verified": actual == expected_commit,
    }


def standardized_frame(
    *,
    example_id: pd.Series,
    language: str,
    text: pd.Series,
    label: pd.Series,
    split: pd.Series,
    source_dataset: str,
    source_row_id: pd.Series,
) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "example_id": example_id.astype(str),
            "language": language,
            "text": text.fillna("").astype(str),
            "label": pd.to_numeric(label, errors="raise").astype(int),
            "split": split.astype(str),
            "source_dataset": source_dataset,
            "source_row_id": source_row_id.astype(str),
        }
    )
    if not set(result["label"].unique()).issubset({0, 1}):
        raise ValueError(f"{source_dataset} contains nonbinary labels")
    result["text_sha256"] = result["text"].map(normalize_text).map(text_hash)
    return result


def prepare_atc(workbook: Path, manifest_path: Path) -> pd.DataFrame:
    raw = pd.read_excel(workbook).reset_index(names="row_id")
    manifest = pd.read_csv(manifest_path)
    joined = manifest.merge(raw, on="row_id", how="left", validate="one_to_one")
    if joined["message"].isna().any():
        raise ValueError("ATC manifest references missing workbook rows")
    current_hash = joined["message"].map(normalize_text).map(text_hash)
    if not current_hash.eq(joined["text_sha256"]).all():
        raise ValueError("ATC workbook changed after its split manifest was created")
    return standardized_frame(
        example_id="ATC-" + joined["row_id"].astype(str),
        language="TR",
        text=joined["message"],
        label=joined["label_x"],
        split=joined["split"],
        source_dataset="ATC",
        source_row_id=joined["row_id"],
    )


def prepare_finnish(workbook: Path, manifest_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(workbook).reset_index(names="row_id")
    manifest = pd.read_csv(manifest_path)
    joined = manifest.merge(raw, on="row_id", how="left", validate="one_to_one")
    if joined["Completion"].isna().any():
        raise ValueError("Finnish manifest references missing workbook rows")
    standard = standardized_frame(
        example_id="RTP-FI-" + joined["row_id"].astype(str),
        language="FI",
        text=joined["Completion"],
        label=joined["label"],
        split=joined["split"],
        source_dataset="RTP-LX-FI",
        source_row_id=joined["row_id"],
    )

    positive = joined.loc[joined["label"].eq(1)].copy()
    toxic = standardized_frame(
        example_id="RTP-FI-PAIR-" + positive["row_id"].astype(str) + "-T",
        language="FI",
        text=positive["Completion"],
        label=pd.Series(1, index=positive.index),
        split=pd.Series("paired_test", index=positive.index),
        source_dataset="RTP-LX-FI-PAIRED",
        source_row_id=positive["row_id"],
    )
    benign = standardized_frame(
        example_id="RTP-FI-PAIR-" + positive["row_id"].astype(str) + "-B",
        language="FI",
        text=positive["BenignCompletion"],
        label=pd.Series(0, index=positive.index),
        split=pd.Series("paired_test", index=positive.index),
        source_dataset="RTP-LX-FI-PAIRED",
        source_row_id=positive["row_id"],
    )
    toxic["pair_id"] = "RTP-FI-PAIR-" + positive["row_id"].astype(str).values
    benign["pair_id"] = "RTP-FI-PAIR-" + positive["row_id"].astype(str).values
    paired = pd.concat([toxic, benign], ignore_index=True)
    return standard, paired


def prepare_korean(repo: Path, seed: int) -> pd.DataFrame:
    train_path = repo / "labeled" / "train.tsv"
    dev_path = repo / "labeled" / "dev.tsv"
    if not train_path.exists() or not dev_path.exists():
        raise FileNotFoundError("Korean checkout must contain labeled/train.tsv and labeled/dev.tsv")
    train = pd.read_csv(train_path, sep="\t").reset_index(names="row_number")
    test = pd.read_csv(dev_path, sep="\t").reset_index(names="row_number")
    for frame in (train, test):
        frame["label"] = frame["hate"].ne("none").astype(int)

    train["source_row_id"] = "train:" + train["row_number"].astype(str)
    allocation_input = train[["source_row_id", "label"]].copy()
    allocation = build_stratified_manifest(
        allocation_input,
        row_id_col="source_row_id",
        label_col="label",
        split_sizes={"train": 6396, "dev": 500, "adaptation": 1000},
        seed=seed,
    )
    train = train.merge(
        allocation[["row_id", "split"]],
        left_on="source_row_id",
        right_on="row_id",
        validate="one_to_one",
    )
    train_standard = standardized_frame(
        example_id="BEEP-TRAIN-" + train["row_number"].astype(str),
        language="KO",
        text=train["comments"],
        label=train["label"],
        split=train["split"],
        source_dataset="BEEP",
        source_row_id=train["source_row_id"],
    )

    test_standard = standardized_frame(
        example_id="BEEP-DEV-" + test["row_number"].astype(str),
        language="KO",
        text=test["comments"],
        label=test["label"],
        split=pd.Series("test", index=test.index),
        source_dataset="BEEP",
        source_row_id="dev:" + test["row_number"].astype(str),
    )
    return pd.concat([train_standard, test_standard], ignore_index=True)


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def prepare_english(repo: Path) -> pd.DataFrame:
    base = repo / "datasets" / "offensive"
    parts: list[pd.DataFrame] = []
    split_map = {"train": "train", "val": "dev", "test": "test"}
    for source_split, split in split_map.items():
        texts = _read_lines(base / f"{source_split}_text.txt")
        labels = _read_lines(base / f"{source_split}_labels.txt")
        if len(texts) != len(labels):
            raise ValueError(f"TweetEval {source_split} text/label lengths differ")
        row_number = pd.Series(range(len(texts)))
        parts.append(
            standardized_frame(
                example_id=pd.Series(
                    [f"TWEETEVAL-OFF-{source_split.upper()}-{index}" for index in range(len(texts))]
                ),
                language="EN",
                text=pd.Series(texts),
                label=pd.Series(labels),
                split=pd.Series(split, index=range(len(texts))),
                source_dataset="TweetEval-Offensive",
                source_row_id=source_split + ":" + row_number.astype(str),
            )
        )
    return pd.concat(parts, ignore_index=True)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported Hungarian file: {path}")


def prepare_hungarian(
    path: Path,
    *,
    text_column: str,
    label_column: str,
    seed: int,
    threshold: float,
) -> pd.DataFrame:
    raw = read_table(path).reset_index(names="row_number")
    missing = {text_column, label_column}.difference(raw.columns)
    if missing:
        raise ValueError(f"Hungarian data is missing columns: {sorted(missing)}")
    raw["label"] = pd.to_numeric(raw[label_column], errors="raise").ge(threshold).astype(int)
    raw["source_row_id"] = raw["row_number"].astype(str)
    test_size = int(round(len(raw) * 0.9))
    adaptation_size = len(raw) - test_size
    allocation = build_stratified_manifest(
        raw[["source_row_id", "label"]],
        row_id_col="source_row_id",
        label_col="label",
        split_sizes={"adaptation": adaptation_size, "test": test_size},
        seed=seed,
    )
    raw = raw.merge(
        allocation[["row_id", "split"]],
        left_on="source_row_id",
        right_on="row_id",
        validate="one_to_one",
    )
    return standardized_frame(
        example_id="HU-TOXIC-" + raw["row_number"].astype(str),
        language="HU",
        text=raw[text_column],
        label=raw["label"],
        split=raw["split"],
        source_dataset="Hungarian-Toxic-Comments",
        source_row_id=raw["source_row_id"],
    )


def dataset_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for (dataset, language, split), group in frame.groupby(
        ["source_dataset", "language", "split"], sort=True
    ):
        lengths = group["text"].astype(str).str.len()
        records.append(
            {
                "source_dataset": dataset,
                "language": language,
                "split": split,
                "n": int(len(group)),
                "negative_0": int(group["label"].eq(0).sum()),
                "positive_1": int(group["label"].eq(1).sum()),
                "positive_rate": float(group["label"].mean()),
                "duplicate_text_hash_rows": int(group["text_sha256"].duplicated().sum()),
                "median_characters": float(lengths.median()),
                "mean_characters": float(lengths.mean()),
                "max_characters": int(lengths.max()),
            }
        )
    return records


def write_outputs(
    datasets: list[pd.DataFrame],
    *,
    processed_dir: Path,
    artifacts_dir: Path,
    source_metadata: dict[str, Any],
) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(datasets, ignore_index=True)
    if combined["example_id"].duplicated().any():
        raise AssertionError("example_id values are not globally unique")
    combined.to_csv(processed_dir / "standardized_all.csv", index=False)
    for language, group in combined.groupby("language"):
        group.to_csv(processed_dir / f"standardized_{language.lower()}.csv", index=False)

    manifest = combined.drop(columns=["text"]).copy()
    manifest.to_csv(artifacts_dir / "standardized_manifest.csv", index=False)
    summaries = dataset_summary(combined)
    pd.DataFrame(summaries).to_csv(artifacts_dir / "dataset_summary.csv", index=False)
    quality: list[dict[str, Any]] = []
    for dataset, group in combined.groupby("source_dataset", sort=True):
        by_hash = group.groupby("text_sha256")
        quality.append(
            {
                "source_dataset": dataset,
                "duplicate_text_rows": int(group["text_sha256"].duplicated().sum()),
                "cross_split_duplicate_groups": int(by_hash["split"].nunique().gt(1).sum()),
                "conflicting_label_groups": int(by_hash["label"].nunique().gt(1).sum()),
            }
        )
    pd.DataFrame(quality).to_csv(artifacts_dir / "dataset_quality.csv", index=False)
    registry = {
        "schema_version": 1,
        "columns": combined.columns.tolist(),
        "total_rows": int(len(combined)),
        "languages": sorted(combined["language"].unique().tolist()),
        "sources": source_metadata,
        "summary": summaries,
        "quality": quality,
    }
    (artifacts_dir / "dataset_registry.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atc", type=Path, default=Path("ATC (2).xlsx"))
    parser.add_argument("--finnish", type=Path, default=Path("RTP_LX_FI (1).xlsx"))
    parser.add_argument(
        "--atc-manifest", type=Path, default=Path("artifacts/audit/atc_seed42_manifest.csv")
    )
    parser.add_argument(
        "--finnish-manifest",
        type=Path,
        default=Path("artifacts/audit/finnish_seed42_manifest.csv"),
    )
    parser.add_argument("--korean-repo", type=Path)
    parser.add_argument("--tweeteval-repo", type=Path)
    parser.add_argument("--hungarian", type=Path)
    parser.add_argument("--hungarian-text-column", default="text")
    parser.add_argument("--hungarian-label-column", default="toxic")
    parser.add_argument("--hungarian-threshold", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/datasets"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    atc = prepare_atc(args.atc, args.atc_manifest)
    finnish, finnish_paired = prepare_finnish(args.finnish, args.finnish_manifest)
    datasets = [atc, finnish, finnish_paired]
    metadata: dict[str, Any] = {
        "ATC": {"path": str(args.atc), "sha256": file_hash(args.atc)},
        "RTP-LX-FI": {"path": str(args.finnish), "sha256": file_hash(args.finnish)},
    }
    if args.korean_repo:
        datasets.append(prepare_korean(args.korean_repo, args.seed))
        metadata["BEEP"] = public_repo_metadata(
            args.korean_repo, "kocohub/korean-hate-speech", KO_COMMIT
        )
    if args.tweeteval_repo:
        datasets.append(prepare_english(args.tweeteval_repo))
        metadata["TweetEval-Offensive"] = public_repo_metadata(
            args.tweeteval_repo, "cardiffnlp/tweeteval", EN_COMMIT
        )
    if args.hungarian:
        datasets.append(
            prepare_hungarian(
                args.hungarian,
                text_column=args.hungarian_text_column,
                label_column=args.hungarian_label_column,
                seed=args.seed,
                threshold=args.hungarian_threshold,
            )
        )
        metadata["Hungarian-Toxic-Comments"] = {
            "path": str(args.hungarian),
            "sha256": file_hash(args.hungarian),
            "threshold": args.hungarian_threshold,
        }
    write_outputs(
        datasets,
        processed_dir=args.processed_dir,
        artifacts_dir=args.artifacts_dir,
        source_metadata=metadata,
    )
    print(f"Prepared {sum(len(frame) for frame in datasets):,} standardized rows")
    if not args.hungarian:
        print("Hungarian data was not supplied; HU is intentionally absent from the registry")


if __name__ == "__main__":
    main()
