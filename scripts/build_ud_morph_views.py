#!/usr/bin/env python3
"""Build lemma and Universal-Dependencies morphology views with Stanza.

Expected input columns:
    example_id, language, text, label, split

Supported language codes are TR, KO, FI, HU, and EN (case-insensitive). The
output retains the input columns and adds `lemma_text`, `morph_text`, a
frequency-matched `randomized_morph_text` placebo, and descriptive counts.

Stanza is imported lazily so the lightweight audit utilities do not require it.
Install `requirements-morphology.txt` before running this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd


LANGUAGE_MAP = {
    "TR": "tr",
    "KO": "ko",
    "FI": "fi",
    "HU": "hu",
    "EN": "en",
}


def clean_tag_piece(value: Any) -> str:
    """Make a stable, whitespace-free tag token payload."""

    return re.sub(r"\s+", "_", str(value).strip())


def parse_feats(feats: str | None) -> list[str]:
    if not feats or feats == "_":
        return []
    return sorted(piece for piece in str(feats).split("|") if piece)


def word_bundle(word: Any) -> tuple[str, list[str]]:
    surface = str(getattr(word, "text", "") or "")
    lemma = str(getattr(word, "lemma", "") or surface)
    upos = getattr(word, "upos", None)
    tags: list[str] = []
    if upos:
        tags.append(f"UPOS={clean_tag_piece(upos)}")
    tags.extend(clean_tag_piece(piece) for piece in parse_feats(getattr(word, "feats", None)))
    return lemma, tags


def document_to_views(document: Any) -> dict[str, Any]:
    """Serialize a Stanza Document into comparable text views."""

    lemmas: list[str] = []
    bundles: list[list[str]] = []
    sentence_lengths: list[int] = []
    for sentence in getattr(document, "sentences", []):
        count = 0
        for word in getattr(sentence, "words", []):
            lemma, tags = word_bundle(word)
            if not lemma:
                continue
            lemmas.append(lemma)
            bundles.append(tags)
            count += 1
        if count:
            sentence_lengths.append(count)

    morph_parts: list[str] = []
    for lemma, tags in zip(lemmas, bundles):
        morph_parts.append(lemma)
        morph_parts.extend(f"<M:{tag}>" for tag in tags)

    return {
        "lemma_text": " ".join(lemmas),
        "morph_text": " ".join(morph_parts),
        "_lemmas": lemmas,
        "_bundles": bundles,
        "ud_word_count": len(lemmas),
        "ud_morph_feature_count": sum(len(bundle) for bundle in bundles),
        "ud_sentence_count": len(sentence_lengths),
    }


def make_randomized_morph_texts(
    rows: list[dict[str, Any]], *, seed: int
) -> list[str]:
    """Shuffle complete tag bundles across words while retaining lemmas.

    The shuffle is performed over a language-specific collection passed by the
    caller. It preserves the corpus tag-bundle distribution but breaks the
    relationship between a lemma/token and its analysis.
    """

    rng = random.Random(seed)
    bundles = [bundle for row in rows for bundle in row["_bundles"]]
    rng.shuffle(bundles)
    cursor = 0
    randomized: list[str] = []
    for row in rows:
        parts: list[str] = []
        for lemma in row["_lemmas"]:
            bundle = bundles[cursor]
            cursor += 1
            parts.append(lemma)
            parts.extend(f"<M:{tag}>" for tag in bundle)
        randomized.append(" ".join(parts))
    if cursor != len(bundles):
        raise AssertionError("Not every shuffled feature bundle was consumed")
    return randomized


def source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("Input must be CSV or Parquet")


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
        return
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False)
        return
    raise ValueError("Output must be CSV or Parquet")


def create_pipeline(
    language: str, *, download: bool, model_dir: Path | None, use_gpu: bool
):
    try:
        import stanza
    except ImportError as exc:
        raise RuntimeError(
            "Stanza is not installed. Run: pip install -r requirements-morphology.txt"
        ) from exc

    pipeline_kwargs: dict[str, Any] = {}
    download_kwargs: dict[str, Any] = {}
    if model_dir is not None:
        pipeline_kwargs["dir"] = str(model_dir)
        download_kwargs["model_dir"] = str(model_dir)
    if download:
        stanza.download(
            language,
            processors="tokenize,pos,lemma",
            **download_kwargs,
        )
    return stanza.Pipeline(
        lang=language,
        processors="tokenize,pos,lemma",
        use_gpu=use_gpu,
        verbose=False,
        **pipeline_kwargs,
    )


def build_views(
    frame: pd.DataFrame,
    *,
    seed: int,
    download: bool,
    model_dir: Path | None,
    use_gpu: bool,
    batch_documents: int,
) -> pd.DataFrame:
    required = {"example_id", "language", "text", "label", "split"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    result_parts: list[pd.DataFrame] = []
    for language_value, language_frame in frame.groupby("language", sort=True):
        code = str(language_value).upper()
        if code not in LANGUAGE_MAP:
            raise ValueError(
                f"Unsupported language {language_value!r}; expected one of {sorted(LANGUAGE_MAP)}"
            )
        pipeline = create_pipeline(
            LANGUAGE_MAP[code],
            download=download,
            model_dir=model_dir,
            use_gpu=use_gpu,
        )
        analyses: list[dict[str, Any]] = []
        texts = language_frame["text"].fillna("").astype(str).tolist()
        for start in range(0, len(texts), batch_documents):
            documents = pipeline.bulk_process(texts[start : start + batch_documents])
            analyses.extend(document_to_views(document) for document in documents)
        randomized = make_randomized_morph_texts(
            analyses, seed=seed + sum(ord(char) for char in code)
        )

        public_analyses = []
        for analysis, randomized_text in zip(analyses, randomized):
            public_analyses.append(
                {
                    key: value
                    for key, value in analysis.items()
                    if not key.startswith("_")
                }
                | {"randomized_morph_text": randomized_text}
            )
        enriched = language_frame.reset_index(drop=True).join(
            pd.DataFrame(public_analyses)
        )
        result_parts.append(enriched)

    return pd.concat(result_parts, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true", help="Download Stanza models first")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--batch-documents", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_documents < 1:
        raise ValueError("--batch-documents must be positive")
    source = read_frame(args.input)
    output = build_views(
        source,
        seed=args.seed,
        download=args.download,
        model_dir=args.model_dir,
        use_gpu=args.use_gpu,
        batch_documents=args.batch_documents,
    )
    write_frame(output, args.output)
    metadata = {
        "source": str(args.input),
        "source_sha256": source_sha256(args.input),
        "rows": int(len(output)),
        "seed": int(args.seed),
        "use_gpu": bool(args.use_gpu),
        "batch_documents": int(args.batch_documents),
        "language_map": LANGUAGE_MAP,
        "columns": output.columns.tolist(),
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {len(output)} rows to {args.output}")


if __name__ == "__main__":
    main()
