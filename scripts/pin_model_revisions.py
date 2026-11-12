#!/usr/bin/env python3
"""Resolve mutable Hugging Face model revisions to immutable commit hashes.

The source matrix is not modified by default. A pinned copy and a provenance
manifest are written so final paper runs cannot silently change model weights.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REVISION_PATTERN = re.compile(r"[0-9a-fA-F]{40,64}")


def is_immutable_revision(value: Any) -> bool:
    return bool(REVISION_PATTERN.fullmatch(str(value or "")))


def pin_config(
    config: dict[str, Any],
    resolver: Callable[[str, str], str],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    pinned = copy.deepcopy(config)
    cache: dict[tuple[str, str], str] = {}
    records: list[dict[str, str]] = []
    for block in pinned.get("blocks", []):
        for model in block.get("models", []):
            name = str(model["model_name"])
            requested = str(model.get("model_revision") or "main")
            key = (name, requested)
            if key not in cache:
                resolved = requested if is_immutable_revision(requested) else resolver(name, requested)
                if not is_immutable_revision(resolved):
                    raise ValueError(
                        f"Resolver returned a non-immutable revision for {name}: {resolved!r}"
                    )
                cache[key] = resolved.lower()
                records.append(
                    {
                        "model_name": name,
                        "requested_revision": requested,
                        "resolved_revision": cache[key],
                    }
                )
            model["model_revision"] = cache[key]
    if not records:
        raise ValueError("No models were found in the matrix")
    return pinned, records


def huggingface_resolver(model_name: str, revision: str) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required. Install requirements-experiments.txt first."
        ) from exc
    info = HfApi().model_info(model_name, revision=revision, files_metadata=False)
    if not info.sha:
        raise RuntimeError(f"Hugging Face returned no commit SHA for {model_name}@{revision}")
    return str(info.sha)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiment_matrix.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/experiment_matrix.pinned.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/model_revisions.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = json.loads(args.config.read_text(encoding="utf-8"))
    pinned, records = pin_config(source, huggingface_resolver)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(pinned, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "resolved_utc": datetime.now(UTC).isoformat(),
        "source_config": str(args.config),
        "pinned_config": str(args.output),
        "models": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Pinned {len(records)} unique model revisions in {args.output}")
    for record in records:
        print(
            f"  {record['model_name']}@{record['requested_revision']} -> "
            f"{record['resolved_revision']}"
        )


if __name__ == "__main__":
    main()
