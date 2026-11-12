#!/usr/bin/env python3
"""Hierarchical paired bootstrap for pre-specified encoder comparisons.

Each replicate resamples seeds and, within each selected seed, paired test
examples. Conditions must contain the complete pre-specified seed set and the
same example IDs. Macro-F1 is recomputed from predictions in every replicate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


CONDITION_COLUMNS = ["model", "view", "max_length", "frozen_encoder"]


def boolean_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    mapped = series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if mapped.isna().any():
        raise ValueError(f"Cannot parse booleans: {series[mapped.isna()].unique().tolist()}")
    return mapped.astype(bool)


def discover_predictions(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("**/predictions.csv"))
    if not files:
        raise FileNotFoundError(f"No predictions.csv files found under {root}")
    parts: list[pd.DataFrame] = []
    for path in files:
        frame = pd.read_csv(path)
        frame["predictions_path"] = str(path)
        parts.append(frame)
    combined = pd.concat(parts, ignore_index=True)
    required = {
        "run_id",
        "example_id",
        "language",
        "source_dataset",
        "label",
        "prediction",
        "seed",
        "model_revision",
        "scientifically_valid",
        *CONDITION_COLUMNS,
    }
    missing = required.difference(combined.columns)
    if missing:
        raise ValueError(f"Prediction files are missing columns: {sorted(missing)}")
    combined["scientifically_valid"] = boolean_series(combined["scientifically_valid"])
    combined["frozen_encoder"] = boolean_series(combined["frozen_encoder"])
    combined = combined.loc[combined["scientifically_valid"]].copy()
    duplicate = combined.duplicated(["run_id", "example_id"], keep=False)
    if duplicate.any():
        raise ValueError("Duplicate run/example predictions were found")
    return combined


def select_condition(frame: pd.DataFrame, condition: dict[str, Any]) -> pd.DataFrame:
    selected = frame.copy()
    for column in CONDITION_COLUMNS:
        if column not in condition:
            raise ValueError(f"Condition is missing {column}: {condition}")
        expected = condition[column]
        if column == "frozen_encoder":
            selected = selected.loc[selected[column].eq(bool(expected))]
        elif column == "max_length":
            selected = selected.loc[selected[column].astype(int).eq(int(expected))]
        else:
            selected = selected.loc[selected[column].astype(str).eq(str(expected))]
    if selected.empty:
        raise ValueError(f"No predictions match condition: {condition}")
    revisions = selected["model_revision"].dropna().astype(str).unique().tolist()
    if len(revisions) != 1:
        raise ValueError(
            f"Condition matched {len(revisions)} model revisions; clean the output root: {revisions}"
        )
    return selected


def macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(
        f1_score(
            labels,
            predictions,
            labels=[0, 1],
            average="macro",
            zero_division=0,
        )
    )


def align_seed(
    a: pd.DataFrame,
    b: pd.DataFrame,
    *,
    seed: int,
    language: str,
    source_dataset: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    filters = (
        a["seed"].astype(int).eq(seed)
        & a["language"].astype(str).eq(language)
        & a["source_dataset"].astype(str).eq(source_dataset)
    )
    a_seed = a.loc[filters].copy()
    filters_b = (
        b["seed"].astype(int).eq(seed)
        & b["language"].astype(str).eq(language)
        & b["source_dataset"].astype(str).eq(source_dataset)
    )
    b_seed = b.loc[filters_b].copy()
    if a_seed["example_id"].duplicated().any() or b_seed["example_id"].duplicated().any():
        raise ValueError(f"Duplicate examples for seed {seed}, {language}/{source_dataset}")
    joined = a_seed[["example_id", "label", "prediction"]].merge(
        b_seed[["example_id", "label", "prediction"]],
        on="example_id",
        suffixes=("_a", "_b"),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError(f"Conditions have different examples for seed {seed}, {language}")
    if not joined["label_a"].eq(joined["label_b"]).all():
        raise ValueError(f"Gold labels differ for seed {seed}, {language}")
    joined = joined.sort_values("example_id")
    return (
        joined["label_a"].to_numpy(dtype=int),
        joined["prediction_a"].to_numpy(dtype=int),
        joined["prediction_b"].to_numpy(dtype=int),
    )


def hierarchical_paired_bootstrap(
    pairs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    seeds = sorted(pairs)
    a_scores = [macro_f1(pairs[seed][0], pairs[seed][1]) for seed in seeds]
    b_scores = [macro_f1(pairs[seed][0], pairs[seed][2]) for seed in seeds]
    point_difference = float(np.mean(np.subtract(a_scores, b_scores)))

    samples = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        selected = rng.choice(seeds, size=len(seeds), replace=True)
        seed_differences: list[float] = []
        for seed in selected:
            labels, pred_a, pred_b = pairs[int(seed)]
            indices = rng.integers(0, len(labels), size=len(labels))
            seed_differences.append(
                macro_f1(labels[indices], pred_a[indices])
                - macro_f1(labels[indices], pred_b[indices])
            )
        samples[replicate] = float(np.mean(seed_differences))

    lower, upper = np.quantile(samples, [0.025, 0.975])
    probability_nonpositive = (np.count_nonzero(samples <= 0) + 1) / (replicates + 1)
    probability_nonnegative = (np.count_nonzero(samples >= 0) + 1) / (replicates + 1)
    p_two_sided = min(1.0, 2 * min(probability_nonpositive, probability_nonnegative))
    return {
        "a_macro_f1_mean": float(np.mean(a_scores)),
        "a_macro_f1_sd": float(np.std(a_scores, ddof=1)),
        "b_macro_f1_mean": float(np.mean(b_scores)),
        "b_macro_f1_sd": float(np.std(b_scores, ddof=1)),
        "difference_a_minus_b": point_difference,
        "ci_low": float(lower),
        "ci_high": float(upper),
        "p_two_sided_bootstrap": float(p_two_sided),
    }


def holm_adjust(p_values: pd.Series) -> pd.Series:
    values = p_values.astype(float).to_numpy()
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return pd.Series(adjusted, index=p_values.index)


def comparison_rows(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    expected_seeds = {int(seed) for seed in config["expected_seeds"]}
    replicates = int(config.get("replicates", 5000))
    base_seed = int(config.get("bootstrap_seed", 42))
    rows: list[dict[str, Any]] = []

    for comparison in config["comparisons"]:
        a = select_condition(frame, comparison["a"])
        b = select_condition(frame, comparison["b"])
        allowed_languages = set(comparison.get("languages", []))
        groups_a = set(zip(a["language"].astype(str), a["source_dataset"].astype(str)))
        groups_b = set(zip(b["language"].astype(str), b["source_dataset"].astype(str)))
        groups = sorted(groups_a & groups_b)
        if allowed_languages:
            groups = [group for group in groups if group[0] in allowed_languages]
        if not groups:
            raise ValueError(f"No shared evaluation groups for {comparison['name']}")

        for language, source_dataset in groups:
            seeds_a = set(
                a.loc[
                    a["language"].astype(str).eq(language)
                    & a["source_dataset"].astype(str).eq(source_dataset),
                    "seed",
                ].astype(int)
            )
            seeds_b = set(
                b.loc[
                    b["language"].astype(str).eq(language)
                    & b["source_dataset"].astype(str).eq(source_dataset),
                    "seed",
                ].astype(int)
            )
            if seeds_a != expected_seeds or seeds_b != expected_seeds:
                raise ValueError(
                    f"Incomplete seeds for {comparison['name']} {language}: "
                    f"A={sorted(seeds_a)}, B={sorted(seeds_b)}, expected={sorted(expected_seeds)}"
                )
            pairs = {
                seed: align_seed(
                    a,
                    b,
                    seed=seed,
                    language=language,
                    source_dataset=source_dataset,
                )
                for seed in sorted(expected_seeds)
            }
            fingerprint = hashlib.sha256(
                f"{comparison['name']}|{language}|{source_dataset}".encode()
            ).digest()
            offset = int.from_bytes(fingerprint[:4], "little")
            stats = hierarchical_paired_bootstrap(
                pairs,
                replicates=replicates,
                rng=np.random.default_rng(base_seed + offset),
            )
            example_counts = sorted({len(values[0]) for values in pairs.values()})
            rows.append(
                {
                    "comparison": comparison["name"],
                    "language": language,
                    "source_dataset": source_dataset,
                    "n_seeds": len(pairs),
                    "n_examples_per_seed": ",".join(map(str, example_counts)),
                    "replicates": replicates,
                    "a_model_revision": a["model_revision"].iloc[0],
                    "b_model_revision": b["model_revision"].iloc[0],
                    **stats,
                }
            )
    result = pd.DataFrame(rows)
    result["p_holm_all_prespecified_tests"] = holm_adjust(
        result["p_two_sided_bootstrap"]
    )
    return result


def markdown_table(frame: pd.DataFrame) -> str:
    lines = [
        "| Comparison (A−B) | Language | A Macro-F1 | B Macro-F1 | Difference | 95% CI | Bootstrap p | Holm p |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"| {row['comparison']} | {row['language']} | "
            f"{row['a_macro_f1_mean']:.3f} | {row['b_macro_f1_mean']:.3f} | "
            f"{row['difference_a_minus_b']:+.3f} | "
            f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}] | "
            f"{row['p_two_sided_bootstrap']:.4f} | "
            f"{row['p_holm_all_prespecified_tests']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/encoder"))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/encoder_comparisons.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = discover_predictions(args.root)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    results = comparison_rows(predictions, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "encoder_paired_bootstrap.csv", index=False)
    results.to_excel(args.output_dir / "encoder_paired_bootstrap.xlsx", index=False)
    report = "# Pre-specified encoder comparisons\n\n" + markdown_table(results)
    (args.output_dir / "ENCODER_COMPARISONS.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
