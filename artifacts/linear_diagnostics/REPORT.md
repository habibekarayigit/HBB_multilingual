# Lexical and orthographic transfer diagnostics

Four logistic-regression controls were trained on the audited 8,000-example Turkish training split and evaluated on the revised held-out tests. All texts were truncated with the local XLM-R tokenizer to the same 128-token budget used by the legacy encoder experiments. These controls do **not** replace multilingual encoders; they test whether surface string overlap alone can explain transfer.

## Macro-F1 at L=128

| Feature view | Turkish | Korean | Finnish | English |
|---|---:|---:|---:|---:|
| Word TF-IDF (1–2 grams) | 0.898 | 0.254 | 0.452 | 0.516 |
| Character TF-IDF (3–5 grams) | 0.929 | 0.374 | 0.563 | 0.441 |
| Word-initial strings (2–5 chars) | 0.932 | 0.252 | 0.479 | 0.451 |
| Word-final strings (2–5 chars) | 0.876 | 0.254 | 0.403 | 0.485 |

## Paired-bootstrap findings (5,000 replicates)

- Finnish: character n-grams outperformed word-final strings by **+0.161 Macro-F1**, 95% CI **[+0.100, +0.221]**.
- Finnish: word-final strings underperformed word-initial strings by **−0.076**, 95% CI **[−0.138, −0.012]**.
- Korean: character n-grams outperformed word-final strings by **+0.120**, 95% CI **[+0.084, +0.155]**.
- Turkish: word-final strings underperformed word-initial strings by **−0.057**, 95% CI **[−0.076, −0.038]**.
- English was the only target for which word-final strings exceeded general character n-grams (+0.044, 95% CI [+0.010, +0.078]).

## Interpretation

The analyzer-free suffix proxy provides **no evidence that shared agglutinative morphology alone produces direct orthographic transfer**. It is substantially weaker than general character features for Finnish and Korean. This does not disprove a representation-level typological effect: Turkish, Finnish, and Korean do not share suffix strings or script, and character endings are only a crude morphology proxy. It does show that the manuscript's original typological claim cannot be explained or validated merely by word-final surface overlap.

The general character model yields the strongest lightweight target result for Finnish (0.563) and Korean (0.374), but Korean remains weak because 317/471 test documents have no Turkish-trained character features after vectorization. This is direct evidence that script mismatch strongly limits surface-feature transfer.

## Length sensitivity

Full-text Finnish results differ materially from the L=128 condition because almost all Finnish texts are truncated in the encoder setting. The character model changes from 0.535 on full text to 0.563 at L=128 (difference +0.029; 95% CI includes zero). The word-final model changes from 0.323 to 0.403 (+0.080, 95% CI [+0.039, +0.121]), and the word-initial model changes from 0.352 to 0.479 (+0.127, 95% CI [+0.075, +0.179]). The corresponding changes for the other languages are negligible. Results from different length conditions must therefore not be mixed.

Detailed files:

- `l128/metrics.csv`
- `l128/predictions.csv`
- `l128/bootstrap_comparisons.csv`
- `full_text/metrics.csv`
