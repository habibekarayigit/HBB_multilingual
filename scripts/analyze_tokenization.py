#!/usr/bin/env python3
"""Analyze XLM-R tokenization, truncation, and observed token overlap.

This is descriptive evidence only. Subword fertility is not treated as a direct
measure of morphological complexity.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from tokenizers import Tokenizer


def percentile(values: list[int], q: float) -> float:
    series = pd.Series(values, dtype=float)
    return float(series.quantile(q))


def script_counts(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        if "HANGUL" in name:
            counts["Hangul"] += 1
        elif "LATIN" in name:
            counts["Latin"] += 1
        else:
            counts["OtherAlphabetic"] += 1
    return counts


def analyze_group(
    group: pd.DataFrame,
    tokenizer: Tokenizer,
    *,
    content_limit: int,
) -> tuple[dict[str, Any], Counter[int]]:
    token_counts: list[int] = []
    word_counts: list[int] = []
    character_counts: list[int] = []
    token_inventory: Counter[int] = Counter()
    scripts: Counter[str] = Counter()
    for text in group["text"].fillna("").astype(str):
        token_ids = tokenizer.encode(text, add_special_tokens=False).ids
        words = re.findall(r"\S+", text)
        token_counts.append(len(token_ids))
        word_counts.append(max(len(words), 1))
        character_counts.append(len(text))
        token_inventory.update(token_ids)
        scripts.update(script_counts(text))

    total_tokens = sum(token_counts)
    total_words = sum(word_counts)
    truncated = [count > content_limit for count in token_counts]
    retained = [min(count, content_limit) / max(count, 1) for count in token_counts]
    alpha_total = sum(scripts.values())
    record = {
        "n": int(len(group)),
        "negative_0": int(group["label"].eq(0).sum()),
        "positive_1": int(group["label"].eq(1).sum()),
        "mean_characters": float(sum(character_counts) / len(character_counts)),
        "median_characters": percentile(character_counts, 0.5),
        "mean_words": float(sum(word_counts) / len(word_counts)),
        "mean_subword_tokens": float(total_tokens / len(token_counts)),
        "median_subword_tokens": percentile(token_counts, 0.5),
        "p95_subword_tokens": percentile(token_counts, 0.95),
        "corpus_subword_fertility": float(total_tokens / max(total_words, 1)),
        "truncated_n_at_128": int(sum(truncated)),
        "truncated_rate_at_128": float(sum(truncated) / len(truncated)),
        "mean_token_retained_fraction_at_128": float(sum(retained) / len(retained)),
        "unique_observed_token_ids": int(len(token_inventory)),
        "latin_alpha_fraction": float(scripts["Latin"] / alpha_total) if alpha_total else 0.0,
        "hangul_alpha_fraction": float(scripts["Hangul"] / alpha_total) if alpha_total else 0.0,
        "other_alpha_fraction": float(scripts["OtherAlphabetic"] / alpha_total) if alpha_total else 0.0,
    }
    return record, token_inventory


def overlap_record(source: Counter[int], target: Counter[int]) -> dict[str, float]:
    source_types = set(source)
    target_types = set(target)
    intersection = source_types & target_types
    union = source_types | target_types
    target_total = sum(target.values())
    covered_target_occurrences = sum(count for token, count in target.items() if token in source_types)
    return {
        "type_jaccard": len(intersection) / max(len(union), 1),
        "target_type_coverage_by_source": len(intersection) / max(len(target_types), 1),
        "target_token_occurrence_coverage_by_source": covered_target_occurrences / max(target_total, 1),
    }


def select_primary_tests(frame: pd.DataFrame) -> pd.DataFrame:
    primary = frame.loc[
        frame["split"].eq("test")
        & ~frame["source_dataset"].eq("RTP-LX-FI-PAIRED")
    ].copy()
    if primary.empty:
        raise ValueError("No primary test rows found")
    return primary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("data/processed/standardized_all.csv")
    )
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer.json"))
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/tokenization")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.data)
    required = {"language", "text", "label", "split", "source_dataset"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Standardized data is missing columns: {sorted(missing)}")

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    content_limit = args.max_length - 2

    primary = select_primary_tests(frame)
    records: list[dict[str, Any]] = []
    inventories: dict[str, Counter[int]] = {}
    for language, group in primary.groupby("language", sort=True):
        record, inventory = analyze_group(
            group, tokenizer, content_limit=content_limit
        )
        records.append(
            {
                "scope": "primary_test",
                "language": language,
                "label": "all",
                **record,
            }
        )
        inventories[language] = inventory
        for label, label_group in group.groupby("label", sort=True):
            label_record, _ = analyze_group(
                label_group, tokenizer, content_limit=content_limit
            )
            records.append(
                {
                    "scope": "primary_test",
                    "language": language,
                    "label": str(int(label)),
                    **label_record,
                }
            )

    paired = frame.loc[frame["source_dataset"].eq("RTP-LX-FI-PAIRED")].copy()
    if not paired.empty:
        for label, group in paired.groupby("label", sort=True):
            record, _ = analyze_group(group, tokenizer, content_limit=content_limit)
            records.append(
                {
                    "scope": "finnish_paired",
                    "language": "FI",
                    "label": str(int(label)),
                    **record,
                }
            )

    overlaps: list[dict[str, Any]] = []
    tr_source = frame.loc[
        frame["language"].eq("TR")
        & frame["source_dataset"].eq("ATC")
        & frame["split"].eq("train")
    ]
    if not tr_source.empty:
        _, tr_source_inventory = analyze_group(
            tr_source, tokenizer, content_limit=content_limit
        )
        for language, inventory in sorted(inventories.items()):
            if language == "TR":
                continue
            overlaps.append(
                {
                    "source": "TR_train",
                    "target": language,
                    **overlap_record(tr_source_inventory, inventory),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = pd.DataFrame(records)
    overlap = pd.DataFrame(overlaps)
    stats.to_csv(args.output_dir / "tokenization_stats.csv", index=False)
    overlap.to_csv(args.output_dir / "token_overlap_from_tr.csv", index=False)

    all_rows = stats.loc[(stats["scope"] == "primary_test") & (stats["label"] == "all")]
    summary = {
        "max_length": args.max_length,
        "content_token_limit": content_limit,
        "primary_test": all_rows.set_index("language").to_dict(orient="index"),
        "tr_to_target_overlap": overlap.set_index("target").to_dict(orient="index") if not overlap.empty else {},
        "interpretation_warning": "Subword fertility reflects tokenizer behavior and is not a direct measure of morphological complexity.",
    }
    (args.output_dir / "tokenization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Analyzed {len(primary):,} primary-test texts")


if __name__ == "__main__":
    main()
