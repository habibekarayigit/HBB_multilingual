#!/usr/bin/env python3
"""Generate manuscript figures from audited result files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LANGUAGE_ORDER = ["TR", "KO", "FI", "HU", "EN"]
LANGUAGE_NAMES = {
    "TR": "Turkish",
    "KO": "Korean",
    "FI": "Finnish",
    "HU": "Hungarian",
    "EN": "English",
}


def available_languages(values) -> list[str]:
    present = set(map(str, values))
    return [language for language in LANGUAGE_ORDER if language in present] + sorted(
        present.difference(LANGUAGE_ORDER)
    )
MODEL_NAMES = {
    "word_1_2": "Word 1–2",
    "char_wb_3_5": "Character 3–5",
    "prefix_2_5": "Word-initial 2–5",
    "suffix_2_5": "Word-final 2–5",
}
COLORS = {
    "word_1_2": "#4C78A8",
    "char_wb_3_5": "#F58518",
    "prefix_2_5": "#54A24B",
    "suffix_2_5": "#E45756",
}


def save(fig: plt.Figure, directory: Path, stem: str) -> None:
    fig.savefig(directory / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def tokenization_figure(stats_path: Path, output_dir: Path) -> None:
    stats = pd.read_csv(stats_path)
    stats = stats.loc[
        stats["scope"].eq("primary_test") & stats["label"].astype(str).eq("all")
    ].set_index("language")
    stats = stats.loc[available_languages(stats.index)]
    x = np.arange(len(stats))
    labels = [LANGUAGE_NAMES[lang] for lang in stats.index]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(x, stats["corpus_subword_fertility"], color="#4C78A8")
    axes[0].set_xticks(x, labels, rotation=20)
    axes[0].set_ylabel("Subword tokens per whitespace unit")
    axes[0].set_title("(a) XLM-R subword fertility")
    axes[0].set_ylim(0, max(stats["corpus_subword_fertility"]) * 1.2)

    truncation = 100 * stats["truncated_rate_at_128"]
    axes[1].bar(x, truncation, color="#E45756")
    axes[1].set_xticks(x, labels, rotation=20)
    axes[1].set_ylabel("Test texts truncated (%)")
    axes[1].set_title("(b) Truncation at max_length=128")
    axes[1].set_ylim(0, 105)
    for index, value in enumerate(truncation):
        axes[1].text(index, value + 2, f"{value:.1f}", ha="center", fontsize=9)

    fig.suptitle("Tokenizer behavior and sequence-length asymmetry")
    fig.tight_layout()
    save(fig, output_dir, "figure_tokenization_truncation")


def linear_figure(
    metrics_path: Path, bootstrap_path: Path, output_dir: Path
) -> None:
    metrics = pd.read_csv(metrics_path)
    bootstrap = pd.read_csv(bootstrap_path)
    ci = bootstrap.loc[~bootstrap["comparison"].str.contains(" minus ")].rename(
        columns={"comparison": "model"}
    )
    metrics = metrics.merge(
        ci[["language", "model", "ci_low", "ci_high"]],
        on=["language", "model"],
        validate="one_to_one",
    )
    languages = available_languages(metrics["language"].unique())
    models = list(MODEL_NAMES)
    x = np.arange(len(languages))
    width = 0.19

    fig, ax = plt.subplots(figsize=(10, 5.2))
    for offset, model in enumerate(models):
        group = metrics.loc[metrics["model"].eq(model)].set_index("language").loc[languages]
        positions = x + (offset - 1.5) * width
        values = group["macro_f1"].to_numpy()
        errors = np.vstack(
            [
                values - group["ci_low"].to_numpy(),
                group["ci_high"].to_numpy() - values,
            ]
        )
        ax.bar(
            positions,
            values,
            width,
            label=MODEL_NAMES[model],
            color=COLORS[model],
            yerr=errors,
            capsize=2,
            linewidth=0.5,
            edgecolor="white",
        )
    ax.set_xticks(x, [LANGUAGE_NAMES[lang] for lang in languages])
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.0)
    ax.set_title("Turkish-trained lexical and orthographic diagnostics (L=128)")
    ax.legend(ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, output_dir, "figure_linear_diagnostics")


def overlap_figure(overlap_path: Path, output_dir: Path) -> None:
    overlap = pd.read_csv(overlap_path).set_index("target")
    targets = [language for language in available_languages(overlap.index) if language != "TR"]
    overlap = overlap.loc[targets]
    x = np.arange(len(overlap))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(
        x - width / 2,
        100 * overlap["target_type_coverage_by_source"],
        width,
        label="Type coverage",
        color="#72B7B2",
    )
    ax.bar(
        x + width / 2,
        100 * overlap["target_token_occurrence_coverage_by_source"],
        width,
        label="Token-occurrence coverage",
        color="#4C78A8",
    )
    ax.set_xticks(x, [LANGUAGE_NAMES[lang] for lang in overlap.index])
    ax.set_ylabel("Covered by Turkish training tokens (%)")
    maximum = 100 * max(
        overlap["target_type_coverage_by_source"].max(),
        overlap["target_token_occurrence_coverage_by_source"].max(),
    )
    ax.set_ylim(0, min(105, max(10, maximum * 1.15)))
    ax.set_title("Observed XLM-R token overlap from Turkish training")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, output_dir, "figure_token_overlap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenization",
        type=Path,
        default=Path("artifacts/tokenization/tokenization_stats.csv"),
    )
    parser.add_argument(
        "--overlap",
        type=Path,
        default=Path("artifacts/tokenization/token_overlap_from_tr.csv"),
    )
    parser.add_argument(
        "--linear-metrics",
        type=Path,
        default=Path("artifacts/linear_diagnostics/l128/metrics.csv"),
    )
    parser.add_argument(
        "--linear-bootstrap",
        type=Path,
        default=Path("artifacts/linear_diagnostics/l128/bootstrap_comparisons.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("manuscript/figures")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
        }
    )
    tokenization_figure(args.tokenization, args.output_dir)
    overlap_figure(args.overlap, args.output_dir)
    linear_figure(args.linear_metrics, args.linear_bootstrap, args.output_dir)
    print(f"Wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
