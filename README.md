# HBB multilingual abusive-language transfer

This repository contains the local ATC and Finnish RTP-LX workbooks, the original exploratory Colab notebook, and an initial reproducibility audit for the manuscript on morphology and cross-lingual abusive-language detection.

> **Important:** `Cross_lingual.ipynb` is an exploratory, stateful notebook. It is not currently a clean end-to-end reproduction of every table claimed in the manuscript. Do not treat its cached outputs as final paper evidence.

## Start here

- Turkish editorial-response and redesign plan: [`docs/REVISION_ROADMAP_TR.md`](docs/REVISION_ROADMAP_TR.md)
- Notebook/manuscript reproducibility audit: [`docs/REPRODUCIBILITY_AUDIT_TR.md`](docs/REPRODUCIBILITY_AUDIT_TR.md)
- Generated dataset audit: [`artifacts/audit/data_audit.md`](artifacts/audit/data_audit.md)
- Standardized dataset registry: [`artifacts/datasets/dataset_registry.json`](artifacts/datasets/dataset_registry.json)
- Tokenization/truncation audit: [`artifacts/tokenization/REPORT.md`](artifacts/tokenization/REPORT.md)
- Lexical/orthographic diagnostic results: [`artifacts/linear_diagnostics/REPORT.md`](artifacts/linear_diagnostics/REPORT.md)
- Audited revised manuscript working draft: [`manuscript/REVISED_MANUSCRIPT_WORKING_DRAFT.md`](manuscript/REVISED_MANUSCRIPT_WORKING_DRAFT.md)
- Clean Colab launcher for the revised pipeline: [`Cross_lingual_REVISED.ipynb`](Cross_lingual_REVISED.ipynb)
- Turkish manuscript change log: [`manuscript/REVISION_CHANGELOG_TR.md`](manuscript/REVISION_CHANGELOG_TR.md)
- Linguistic error-annotation guide: [`docs/LINGUISTIC_ERROR_ANNOTATION_GUIDE_TR.md`](docs/LINGUISTIC_ERROR_ANNOTATION_GUIDE_TR.md)

## Reproduce the verified CPU analysis

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-analysis.txt -r requirements-experiments.txt
PYTHON=.venv/bin/python scripts/run_verified_analysis.sh
```

This one command rebuilds the audits, standardized manifests, tokenization analysis, linear diagnostics, 5,000-replicate bootstrap intervals, corrected paper tables, figures, and tests.

To run only the initial data audit:

```bash
.venv/bin/python scripts/audit_and_build_splits.py
```

The audit creates shareable manifests under `artifacts/audit/`. They contain row IDs, labels, splits, and text hashes—not message text.

Key checks performed:

- ATC duplicate, blank, and conflicting-label detection;
- leakage-safe, deduplicated 8,000/1,000/1,000 ATC split manifest;
- Finnish toxicity-threshold sensitivity;
- corrected 560/99/440 Finnish split counts;
- a paired Finnish toxic/benign diagnostic manifest (with a documented length confound; not a minimal-pair benchmark).

## Prepare the standardized datasets

The Korean and English sources are fetched at immutable Git commits with sparse checkouts:

```bash
scripts/fetch_public_datasets.sh
.venv/bin/python scripts/prepare_standardized_datasets.py \
  --korean-repo data/raw/korean-hate-speech \
  --tweeteval-repo data/raw/tweeteval
.venv/bin/python scripts/analyze_tokenization.py
.venv/bin/python scripts/make_dataset_tables.py
```

Raw/processed text below `data/` is ignored by Git. Shareable hashes, manifests, corrected paper tables, and aggregate audits are retained under `artifacts/`.

Hungarian is deliberately absent until the exact source file is supplied. Once available, pass `--hungarian FILE` plus the correct text/label column names and threshold; the script will hash it and create a fixed 10% adaptation/90% test manifest.

## Build controlled morphology views

After standardization, install the morphology dependencies and build surface-aligned lemma/UD-feature views (normally in Colab, where Stanza model downloads are available):

```bash
.venv/bin/pip install -r requirements-morphology.txt
.venv/bin/python scripts/build_ud_morph_views.py \
  data/processed/standardized_all.csv \
  data/processed/standardized_with_morphology.parquet \
  --download --use-gpu
```

The output includes `lemma_text`, `morph_text`, and a frequency-matched `randomized_morph_text` placebo for controlled ablations. Analyzer quality still needs manual checking on social-media samples in every language.

## Run the clean encoder pipeline

Install the experiment dependencies and validate the full train/evaluate/output path with a local random tiny model:

```bash
.venv/bin/pip install -r requirements-experiments.txt
.venv/bin/python scripts/run_encoder_experiment.py --smoke-test --epochs 1
```

Smoke metrics are marked `scientifically_valid=false`. Publication runs require CUDA and pretrained weights. Inspect the 30-run plan before execution:

```bash
.venv/bin/python scripts/run_experiment_matrix.py
# On an A100 runtime, after pinning immutable model revisions:
.venv/bin/python scripts/run_experiment_matrix.py --execute
```

Each real run writes `run_config.json`, training history, aggregate metrics, and per-example predictions. The corrected frozen baseline logs trainable/frozen parameter counts and trains only the classification head. After all seeds finish, aggregate with a strict completeness check:

```bash
.venv/bin/python scripts/aggregate_encoder_results.py
```

## Prepare a blinded linguistic error sample

After a model run, export one row per example with:

```text
example_id,language,text,label,prediction,model,probability_1
```

Then run:

```bash
.venv/bin/python scripts/sample_linguistic_errors.py predictions.csv \
  --model xlm-roberta-base \
  --per-cell 25 \
  --seed 42
```

Annotators should receive only `annotation_blind.csv`; keep `annotation_key.csv` hidden until coding is complete.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```
