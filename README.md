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
- Same manuscript with added/substantively rewritten passages in **bold**: [Markdown](manuscript/REVISED_MANUSCRIPT_CHANGES_BOLDED.md) · [editable Word document](manuscript/REVISED_MANUSCRIPT_CHANGES_BOLDED.docx)
- Reviewer-auditable end-to-end Colab pipeline: [`Cross_lingual_REVISED.ipynb`](Cross_lingual_REVISED.ipynb)
- Turkish guide to the confirmatory Colab workflow: [`docs/COLAB_CONFIRMATORY_PIPELINE_TR.md`](docs/COLAB_CONFIRMATORY_PIPELINE_TR.md)
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

The output includes `lemma_text`, `morph_text`, and a frequency-matched `randomized_morph_text` placebo for controlled ablations. Analyzer quality must be checked on a fixed language-balanced sample before the morphology block is interpreted:

```bash
.venv/bin/python scripts/sample_morphology_qc.py \
  data/processed/standardized_with_morphology.parquet \
  --per-language 100 \
  --seed 42
```

## Run the clean encoder pipeline

Install the experiment dependencies and validate the full train/evaluate/output path with a local random tiny model:

```bash
.venv/bin/pip install -r requirements-experiments.txt
.venv/bin/python scripts/run_encoder_experiment.py --smoke-test --epochs 1
```

Smoke metrics are marked `scientifically_valid=false`. Publication runs require CUDA and pretrained weights. The matrix contains 12 primary model/seed runs, 9 morphology-view runs, 6 Finnish length-sensitivity runs, and 3 truly frozen-encoder runs. Each trained run evaluates every available target language, so languages are not incorrectly counted as separate training jobs. Resolve model tags to immutable Hugging Face commits, validate every input, and inspect the exact 30-run plan before execution:

```bash
.venv/bin/python scripts/pin_model_revisions.py
.venv/bin/python scripts/validate_confirmatory_setup.py \
  --config configs/experiment_matrix.pinned.json \
  --require-cuda
.venv/bin/python scripts/run_experiment_matrix.py \
  --config configs/experiment_matrix.pinned.json
.venv/bin/python scripts/run_experiment_matrix.py \
  --config configs/experiment_matrix.pinned.json \
  --execute
```

Mutable revisions cause a hard failure during real execution. Surface blocks read `standardized_all.csv`; only the morphology mechanism block requires `standardized_with_morphology.parquet`. Every real run writes `run_config.json`, training history, aggregate metrics, and per-example predictions including the original text for later blinded annotation. The corrected frozen baseline logs trainable/frozen parameter counts and trains only the classification head.

After every seed finishes, aggregate and run the pre-specified hierarchical paired bootstrap. Both commands reject incomplete conditions:

```bash
.venv/bin/python scripts/aggregate_encoder_results.py
.venv/bin/python scripts/bootstrap_encoder_comparisons.py
.venv/bin/python scripts/make_encoder_figures.py
```

## Prepare a blinded linguistic error sample

Build a stable three-seed probability ensemble and draw a pre-specified error sample:

```bash
.venv/bin/python scripts/ensemble_encoder_predictions.py
.venv/bin/python scripts/sample_linguistic_errors.py \
  artifacts/error_annotation/xlmr_base_surface_ensemble_predictions.csv \
  --outcomes FP,FN \
  --per-cell 100 \
  --seed 42
```

Annotators should receive independent copies of only `annotation_blind.csv`; keep `annotation_key.csv` hidden until coding is complete. After both copies are returned, calculate agreement before adjudication, then summarize the adjudicated sheet:

```bash
.venv/bin/python scripts/calculate_annotation_agreement.py \
  annotation_annotator_A.csv annotation_annotator_B.csv \
  --labels annotator_A,annotator_B
.venv/bin/python scripts/summarize_error_annotations.py \
  annotation_adjudicated.csv annotation_key.csv
```

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```
