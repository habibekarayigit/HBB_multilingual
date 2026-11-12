#!/usr/bin/env python3
"""Generate confirmatory encoder figures from complete multi-seed summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LANGUAGE_ORDER = ["TR", "KO", "FI", "HU", "EN"]
MODEL_LABELS = {
    "bert-base-multilingual-cased": "mBERT",
    "xlm-roberta-base": "XLM-R-base",
    "xlm-roberta-large": "XLM-R-large",
    "cis-lmu/glot500-base": "Glot500-base",
}
VIEW_LABELS = {
    "surface": "Surface",
    "lemma": "Lemma",
    "morph": "Lemma + UD",
    "randomized_morph": "Randomized UD",
}
COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]


def ordered_languages(frame: pd.DataFrame) -> list[str]:
    present = set(frame["language"].astype(str))
    return [language for language in LANGUAGE_ORDER if language in present] + sorted(
        present.difference(LANGUAGE_ORDER)
    )


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def grouped_bars(
    frame: pd.DataFrame,
    *,
    category_column: str,
    categories: list[str],
    category_labels: dict[str, str],
    title: str,
    output_dir: Path,
    stem: str,
) -> None:
    languages = ordered_languages(frame)
    x = np.arange(len(languages))
    width = min(0.8 / max(len(categories), 1), 0.22)
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(languages)), 5.2))
    plotted = 0
    for category_index, category in enumerate(categories):
        group = frame.loc[frame[category_column].astype(str).eq(category)].set_index("language")
        if group.empty:
            continue
        values = np.array(
            [group.loc[language, "macro_f1_mean"] if language in group.index else np.nan for language in languages],
            dtype=float,
        )
        errors = np.array(
            [group.loc[language, "macro_f1_std"] if language in group.index else np.nan for language in languages],
            dtype=float,
        )
        positions = x + (category_index - (len(categories) - 1) / 2) * width
        ax.bar(
            positions,
            values,
            width,
            yerr=errors,
            capsize=2,
            color=COLORS[category_index % len(COLORS)],
            label=category_labels.get(category, category),
        )
        plotted += 1
    if not plotted:
        raise ValueError(f"No rows available for {stem}")
    ax.set_xticks(x, languages)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro-F1 (mean ± SD across seeds)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save(fig, output_dir, stem)


def make_figures(summary: pd.DataFrame, output_dir: Path) -> None:
    required = {
        "model",
        "view",
        "max_length",
        "frozen_encoder",
        "language",
        "macro_f1_mean",
        "macro_f1_std",
        "seed_count",
    }
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"Summary is missing columns: {sorted(missing)}")
    if not summary["seed_count"].astype(int).eq(3).all():
        raise ValueError("Figures require complete three-seed conditions")
    summary = summary.copy()
    if not pd.api.types.is_bool_dtype(summary["frozen_encoder"]):
        summary["frozen_encoder"] = summary["frozen_encoder"].astype(str).str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )
    if summary["frozen_encoder"].isna().any():
        raise ValueError("Could not parse frozen_encoder as boolean")
    output_dir.mkdir(parents=True, exist_ok=True)

    primary = summary.loc[
        summary["view"].eq("surface")
        & summary["max_length"].astype(int).eq(128)
        & ~summary["frozen_encoder"].astype(bool)
        & summary["model"].isin(MODEL_LABELS)
    ]
    grouped_bars(
        primary,
        category_column="model",
        categories=list(MODEL_LABELS),
        category_labels=MODEL_LABELS,
        title="Confirmatory surface-model comparison",
        output_dir=output_dir,
        stem="figure_encoder_primary_models",
    )

    mechanism = summary.loc[
        summary["model"].eq("xlm-roberta-base")
        & summary["view"].isin(VIEW_LABELS)
        & summary["max_length"].astype(int).eq(128)
        & ~summary["frozen_encoder"].astype(bool)
    ]
    grouped_bars(
        mechanism,
        category_column="view",
        categories=list(VIEW_LABELS),
        category_labels=VIEW_LABELS,
        title="XLM-R-base morphology intervention",
        output_dir=output_dir,
        stem="figure_encoder_morphology_views",
    )

    finnish = summary.loc[
        summary["model"].eq("xlm-roberta-base")
        & summary["view"].eq("surface")
        & summary["language"].eq("FI")
        & ~summary["frozen_encoder"].astype(bool)
        & summary["max_length"].astype(int).isin([128, 256, 512])
    ].sort_values("max_length")
    if len(finnish) != 3:
        raise ValueError("Finnish length figure requires L=128, 256, and 512")
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.errorbar(
        finnish["max_length"],
        finnish["macro_f1_mean"],
        yerr=finnish["macro_f1_std"],
        color="#4C78A8",
        marker="o",
        linewidth=2,
        capsize=4,
    )
    ax.set_xticks([128, 256, 512])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Maximum sequence length")
    ax.set_ylabel("Finnish Macro-F1 (mean ± SD)")
    ax.set_title("Finnish sequence-length sensitivity")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, output_dir, "figure_encoder_finnish_length")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/results/encoder_summary.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/results/figures")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.summary)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    make_figures(summary, args.output_dir)
    print(f"Wrote confirmatory figures to {args.output_dir}")


if __name__ == "__main__":
    main()
