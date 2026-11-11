#!/usr/bin/env python3
"""Run lightweight lexical/orthographic transfer diagnostics.

These models are controls, not substitutes for multilingual encoders. They test
whether word, character, word-initial, or word-final string overlap alone can
explain transfer from Turkish.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenizers import Tokenizer
from sklearn.linear_model import LogisticRegression

from run_encoder_experiment import binary_metrics, sha256_file


WORD_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def normalized_words(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return WORD_PATTERN.findall(normalized)


def edge_analyzer(edge: str) -> Callable[[str], list[str]]:
    if edge not in {"prefix", "suffix"}:
        raise ValueError(edge)

    def analyze(text: str) -> list[str]:
        features: list[str] = []
        for word in normalized_words(text):
            for width in range(2, min(5, len(word)) + 1):
                piece = word[:width] if edge == "prefix" else word[-width:]
                features.append(f"{edge}:{width}:{piece}")
        return features

    return analyze


def vectorizer_for(name: str) -> TfidfVectorizer:
    common = {
        "lowercase": True,
        "min_df": 2,
        "max_features": 100_000,
        "sublinear_tf": True,
        "dtype": np.float32,
    }
    if name == "word_1_2":
        return TfidfVectorizer(analyzer="word", ngram_range=(1, 2), **common)
    if name == "char_wb_3_5":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), **common)
    if name == "prefix_2_5":
        return TfidfVectorizer(analyzer=edge_analyzer("prefix"), **common)
    if name == "suffix_2_5":
        return TfidfVectorizer(analyzer=edge_analyzer("suffix"), **common)
    raise ValueError(name)


def truncate_like_encoder(
    texts: pd.Series, *, tokenizer_path: Path, max_length: int
) -> pd.Series:
    if max_length == 0:
        return texts.fillna("").astype(str)
    if max_length < 8:
        raise ValueError("--max-length must be 0 or at least 8")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    content_limit = max_length - 2
    values: list[str] = []
    for text in texts.fillna("").astype(str):
        ids = tokenizer.encode(text, add_special_tokens=False).ids[:content_limit]
        values.append(tokenizer.decode(ids, skip_special_tokens=True))
    return pd.Series(values, index=texts.index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("data/processed/standardized_all.csv")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/linear_diagnostics/l128")
    )
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer.json"))
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="XLM-R-equivalent total length; use 0 for full text.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.data)
    frame = frame.loc[~frame["source_dataset"].eq("RTP-LX-FI-PAIRED")].copy()
    frame["_analysis_text"] = truncate_like_encoder(
        frame["text"], tokenizer_path=args.tokenizer, max_length=args.max_length
    )
    train = frame.loc[
        frame["language"].eq("TR") & frame["split"].eq("train")
    ].copy()
    tests = frame.loc[frame["split"].eq("test")].copy()
    if train.empty or tests.empty:
        raise ValueError("Turkish training and primary test rows are required")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    prediction_parts: list[pd.DataFrame] = []
    model_names = ["word_1_2", "char_wb_3_5", "prefix_2_5", "suffix_2_5"]
    for model_name in model_names:
        print(f"Fitting {model_name}...")
        vectorizer = vectorizer_for(model_name)
        x_train = vectorizer.fit_transform(train["_analysis_text"].astype(str))
        classifier = LogisticRegression(
            C=1.0,
            max_iter=2000,
            class_weight="balanced",
            random_state=args.seed,
            solver="liblinear",
        )
        classifier.fit(x_train, train["label"].astype(int))

        for (language, dataset), group in tests.groupby(
            ["language", "source_dataset"], sort=True
        ):
            x_test = vectorizer.transform(group["_analysis_text"].astype(str))
            probabilities = classifier.predict_proba(x_test)[:, 1]
            predictions = (probabilities >= 0.5).astype(int)
            labels = group["label"].astype(int).to_numpy()
            metrics = binary_metrics(labels, predictions)
            nonzero = np.asarray(x_test.getnnz(axis=1))
            metric_rows.append(
                {
                    "model": model_name,
                    "seed": args.seed,
                    "max_length": args.max_length,
                    "language": language,
                    "source_dataset": dataset,
                    "n": len(group),
                    "vocabulary_size": len(vectorizer.vocabulary_),
                    "zero_feature_documents": int((nonzero == 0).sum()),
                    "mean_nonzero_features": float(nonzero.mean()),
                    "predicted_positive_rate": float(predictions.mean()),
                    **metrics,
                }
            )
            output = group[
                [
                    "example_id",
                    "language",
                    "source_dataset",
                    "label",
                    "text_sha256",
                ]
            ].copy()
            output["prediction"] = predictions
            output["probability_1"] = probabilities
            output["model"] = model_name
            output["seed"] = args.seed
            output["max_length"] = args.max_length
            prediction_parts.append(output)

    metrics_frame = pd.DataFrame(metric_rows)
    predictions_frame = pd.concat(prediction_parts, ignore_index=True)
    metrics_frame.to_csv(args.output_dir / "metrics.csv", index=False)
    predictions_frame.to_csv(args.output_dir / "predictions.csv", index=False)
    metadata = {
        "purpose": "lexical/orthographic diagnostic; not a multilingual encoder result",
        "data_sha256": sha256_file(args.data),
        "seed": args.seed,
        "max_length": args.max_length,
        "tokenizer": str(args.tokenizer) if args.max_length else None,
        "train_rows": int(len(train)),
        "models": model_names,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(metrics_frame.to_string(index=False))


if __name__ == "__main__":
    main()
