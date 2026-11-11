#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"

"$PYTHON" scripts/audit_and_build_splits.py
scripts/fetch_public_datasets.sh
"$PYTHON" scripts/prepare_standardized_datasets.py \
  --korean-repo data/raw/korean-hate-speech \
  --tweeteval-repo data/raw/tweeteval
"$PYTHON" scripts/analyze_tokenization.py
"$PYTHON" scripts/run_linear_diagnostics.py \
  --output-dir artifacts/linear_diagnostics/l128
"$PYTHON" scripts/run_linear_diagnostics.py \
  --max-length 0 \
  --output-dir artifacts/linear_diagnostics/full_text
"$PYTHON" scripts/bootstrap_linear_diagnostics.py
"$PYTHON" scripts/bootstrap_length_sensitivity.py
"$PYTHON" scripts/make_dataset_tables.py
"$PYTHON" scripts/make_figures.py
"$PYTHON" -m unittest discover -s tests -v

printf '\nVerified CPU analysis completed successfully.\n'
