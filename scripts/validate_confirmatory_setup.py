#!/usr/bin/env python3
"""Fail-fast validation for the confirmatory encoder experiment matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

try:
    from scripts.run_encoder_experiment import VIEW_COLUMNS
    from scripts.run_experiment_matrix import expand_matrix, is_immutable_revision
except ModuleNotFoundError:  # Direct execution: python scripts/validate_confirmatory_setup.py
    from run_encoder_experiment import VIEW_COLUMNS
    from run_experiment_matrix import expand_matrix, is_immutable_revision


BASE_COLUMNS = {
    "example_id",
    "language",
    "text",
    "label",
    "split",
    "source_dataset",
    "text_sha256",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_validation_columns(path: Path, required: set[str]) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        header = pd.read_csv(path, nrows=0).columns.tolist()
    elif path.suffix.lower() in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("pyarrow is required to inspect morphology parquet files") from exc
        header = pq.ParquetFile(path).schema.names
    else:
        raise ValueError(f"Unsupported data file: {path}")
    missing = required.difference(header)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    columns = sorted(required)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, usecols=columns)
    return pd.read_parquet(path, columns=columns)


def validate_data_file(
    path: Path,
    *,
    required_columns: set[str],
    required_languages: set[str],
) -> dict[str, Any]:
    frame = read_validation_columns(path, required_columns)
    problems: list[str] = []
    if frame.empty:
        problems.append("file has no rows")
    if frame["example_id"].duplicated().any():
        problems.append("example_id is not unique")
    labels = set(pd.to_numeric(frame["label"], errors="coerce").dropna().astype(int))
    if not labels.issubset({0, 1}) or not labels:
        problems.append(f"labels are not nonempty binary values: {sorted(labels)}")
    languages = set(frame["language"].astype(str).str.upper())
    missing_languages = required_languages.difference(languages)
    if missing_languages:
        problems.append(f"missing required languages: {sorted(missing_languages)}")
    primary_tests = frame.loc[
        frame["split"].astype(str).eq("test")
        & ~frame["source_dataset"].astype(str).eq("RTP-LX-FI-PAIRED")
    ]
    test_languages = set(primary_tests["language"].astype(str).str.upper())
    missing_tests = required_languages.difference(test_languages)
    if missing_tests:
        problems.append(f"missing primary test rows for: {sorted(missing_tests)}")
    for column in required_columns.intersection(set(VIEW_COLUMNS.values())):
        blank = frame[column].fillna("").astype(str).str.strip().eq("")
        if blank.any():
            problems.append(f"{column} has {int(blank.sum())} blank rows")
    if problems:
        raise ValueError("; ".join(problems))
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": int(len(frame)),
        "languages": sorted(languages),
        "columns_checked": sorted(required_columns),
    }


def environment_record() -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    record: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": cuda,
    }
    if cuda:
        properties = torch.cuda.get_device_properties(0)
        record.update(
            {
                "cuda_device": torch.cuda.get_device_name(0),
                "cuda_runtime": torch.version.cuda,
                "gpu_memory_gib": round(properties.total_memory / 1024**3, 2),
            }
        )
    return record


def validate(
    config: dict[str, Any],
    *,
    require_cuda: bool,
    allow_mutable_revisions: bool,
    required_languages: set[str],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        runs = expand_matrix(config)
    except Exception as exc:
        return {"ok": False, "errors": [str(exc)], "warnings": [], "environment": environment_record()}

    expected_count = config.get("expected_run_count")
    if expected_count is not None and len(runs) != int(expected_count):
        errors.append(f"matrix has {len(runs)} runs; expected {expected_count}")
    seeds = sorted({int(run["seed"]) for run in runs})
    if seeds != [42, 123, 456]:
        errors.append(f"expected seeds [42, 123, 456], found {seeds}")
    effective_batches = {
        int(run["train_batch_size"]) * int(run["gradient_accumulation_steps"])
        for run in runs
    }
    if len(effective_batches) != 1:
        errors.append(f"effective train batch sizes differ across conditions: {sorted(effective_batches)}")

    revisions_by_model: dict[str, set[str]] = {}
    for run in runs:
        revisions_by_model.setdefault(str(run["model_name"]), set()).add(
            str(run["model_revision"])
        )
    inconsistent_revisions = {
        model: sorted(revisions)
        for model, revisions in revisions_by_model.items()
        if len(revisions) != 1
    }
    if inconsistent_revisions:
        errors.append(f"models use inconsistent revisions across blocks: {inconsistent_revisions}")

    mutable = sorted(
        {
            f"{run['model_name']}@{run['model_revision']}"
            for run in runs
            if not is_immutable_revision(run.get("model_revision"))
        }
    )
    if mutable and not allow_mutable_revisions:
        errors.append("mutable model revisions: " + ", ".join(mutable))
    elif mutable:
        warnings.append("mutable model revisions allowed for non-final validation")

    requirements_by_path: dict[Path, set[str]] = {}
    for run in runs:
        path = Path(run["data"])
        requirements_by_path.setdefault(path, set(BASE_COLUMNS)).add(VIEW_COLUMNS[run["view"]])

    data_records: list[dict[str, Any]] = []
    for path, columns in sorted(requirements_by_path.items(), key=lambda item: str(item[0])):
        if not path.exists():
            errors.append(f"missing data file: {path}")
            continue
        try:
            data_records.append(
                validate_data_file(
                    path,
                    required_columns=columns,
                    required_languages=required_languages,
                )
            )
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    environment = environment_record()
    if require_cuda and not environment["cuda_available"]:
        errors.append("CUDA GPU is required for publication runs")
    if environment["cuda_available"] and environment.get("gpu_memory_gib", 0) < 30:
        warnings.append(
            "GPU has less than 30 GiB memory; XLM-R-large may require smaller batches"
        )

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "environment": environment,
        "run_count": len(runs),
        "seeds": seeds,
        "effective_train_batch_size": sorted(effective_batches),
        "model_revisions": {
            model: sorted(revisions) for model, revisions in revisions_by_model.items()
        },
        "required_languages": sorted(required_languages),
        "data": data_records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_matrix.pinned.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/confirmatory_preflight.json")
    )
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--allow-mutable-revisions", action="store_true")
    parser.add_argument("--require-languages", default="TR,KO,FI,EN")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    languages = {
        value.strip().upper()
        for value in args.require_languages.split(",")
        if value.strip()
    }
    report = validate(
        config,
        require_cuda=args.require_cuda,
        allow_mutable_revisions=args.allow_mutable_revisions,
        required_languages=languages,
    )
    report["config_path"] = str(args.config)
    report["config_sha256"] = sha256_file(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit("Confirmatory setup validation failed")


if __name__ == "__main__":
    main()
