# Corrected dataset tables for the revised manuscript

Generated from `artifacts/datasets/dataset_registry.json` by `scripts/make_dataset_tables.py`. Hungarian is intentionally omitted until its exact source file is supplied and hashed.

## Corpus/source overview

| Language | Source used | Rows in protocol | Negative (0) | Positive (1) | Positive % |
|---|---|---:|---:|---:|---:|
| Turkish | ATC workbook | 10,000 | 6,593 | 3,407 | 34.1 |
| Korean | BEEP! train+dev | 8,367 | 3,646 | 4,721 | 56.4 |
| Finnish | RTP-LX FI | 1,099 | 707 | 392 | 35.7 |
| English | TweetEval Offensive | 14,100 | 9,460 | 4,640 | 32.9 |

The previous manuscript's English total of 11,916 was only the official training split, whereas the Korean total combined train and development. The revised table applies a consistent definition.

## Fixed study splits

| Language | Role/split | Total | Negative (0) | Positive (1) | Positive % |
|---|---|---:|---:|---:|---:|
| Turkish | Train | 8,000 | 5,275 | 2,725 | 34.1 |
| Turkish | Dev | 1,000 | 659 | 341 | 34.1 |
| Turkish | Test | 1,000 | 659 | 341 | 34.1 |
| Korean | Train | 6,396 | 2,824 | 3,572 | 55.8 |
| Korean | Dev | 500 | 221 | 279 | 55.8 |
| Korean | Adaptation pool | 1,000 | 441 | 559 | 55.9 |
| Korean | Test (public BEEP dev) | 471 | 160 | 311 | 66.0 |
| Finnish | Train | 560 | 360 | 200 | 35.7 |
| Finnish | Dev | 99 | 64 | 35 | 35.4 |
| Finnish | Test | 440 | 283 | 157 | 35.7 |
| English | Train | 11,916 | 7,975 | 3,941 | 33.1 |
| English | Dev | 1,324 | 865 | 459 | 34.7 |
| English | Test | 860 | 620 | 240 | 27.9 |

## Important reporting qualifications

- The Korean public test labels are undisclosed. The revised protocol uses the repository's labeled development set as a transparent held-out test and names it explicitly.
- TweetEval contains normalized-text duplicates crossing official splits. Keep official splits for benchmark comparability, but add a deduplicated sensitivity analysis before few-shot/monolingual claims.
- The Finnish test condition is heavily truncated at 128 tokens; see `artifacts/tokenization/REPORT.md`.
- Do not add a language row until its exact dataset file, license, label semantics, and split manifest are available.
