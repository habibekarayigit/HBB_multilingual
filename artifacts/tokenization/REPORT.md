# Tokenization and truncation audit

Computed with the repository's XLM-R tokenizer (`tokenizer.json`) on the revised primary test splits. The 128-token model limit leaves room for 126 content tokens plus special tokens.

| Language | N | Subword fertility | Texts truncated | Mean retained-token fraction |
|---|---:|---:|---:|---:|
| Turkish | 1,000 | 2.211 | 14 (1.4%) | 99.7% |
| Korean | 471 | 2.889 | 0 (0.0%) | 100.0% |
| Finnish | 440 | 1.931 | 436 (99.1%) | 39.0% |
| English | 860 | 1.745 | 1 (0.1%) | 100.0% |

## Consequences

1. The Finnish condition is not comparable at `max_length=128`: almost every Finnish completion is truncated and, on average, only 39% of its subword tokens are retained.
2. The previous fragmentation ratio is a tokenizer-behavior statistic, not direct evidence of morphological complexity.
3. The Finnish paired RTP-LX set has a severe length confound: benign completions have a median of 194 characters and no truncation, whereas toxic completions have a median of 1,205 characters and 99.0% truncation. These are paired prompts but not valid morphology-controlled minimal pairs without length/content matching.
4. Primary model experiments therefore need at least 128/256/512 sensitivity conditions for Finnish. A length-matched Finnish evaluation should be the primary controlled analysis.

## Observed target-token coverage by tokens seen in the Turkish training sample

| Target | Type coverage | Token-occurrence coverage |
|---|---:|---:|
| Korean | 3.6% | 11.1% |
| Finnish | 27.5% | 49.6% |
| English | 32.7% | 64.8% |

These overlap values are descriptive and sample-dependent. They must not be interpreted as a causal estimate of typological similarity.
