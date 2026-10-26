# Completed banking multi-label run

The combined checkpoint is trained on real NLU++ banking annotations and retains the existing emotion and original task heads.

Checkpoint: `checkpoints/multilabel-banking`. Run status: **complete**. Seed: 42.

Training: 3 epochs on 1,422 NLU++ banking messages, with 64 Banking77, 64 SST-5, and 256 GoEmotions replay examples per epoch.
NLU++ validation/calibration/test counts: 206 / 207 / 204. Source folds: 0–13 / 14–15 / 16–17 / 18–19 for training/validation/calibration/test.
A separate banking head, temperature, and threshold share the same encoder with the emotion task.

| Supplied-candidate metric | Jevlet | TF-IDF logistic |
|---|---:|---:|
| Micro-F1 | 0.8651 | 0.8127 |
| Macro-F1 | 0.8270 | 0.6991 |
| Precision | 0.8638 | 0.8312 |
| Recall | 0.8665 | 0.7950 |
| Exact candidate-set accuracy | 0.6716 | 0.5490 |
| Hamming loss | 0.0691 | 0.0937 |
| Mean Brier | 0.0542 | 0.1290 |
| Binary NLL | 0.1917 | 0.3925 |
| Label-probability ECE | 0.0177 | 0.1696 |

Banking validation-selected threshold: **0.53**. Candidate recall: **0.7093**.

This is one fixed source-fold split with 2–10 supplied candidates, including omitted-positive cases. It does not measure full-catalogue retrieval across all 48 modules, reproduce the NLU++ paper’s cross-validation result, or convert Banking77 into a multi-label dataset. Rare-label support is limited.

## Real demo response

Input: “I am trying to make a transfer but it is not working.”

Selected: **Make / open / activate**, **Transfer / payment / deposit**, and **Incorrect / not working**.

| Catalogue value | Probability |
|---|---:|
| make open apply setup get activate | 0.7931 |
| transfer payment deposit | 0.9526 |
| wrong notworking notshowing | 0.9599 |
| card | 0.0033 |
| balance | 0.0057 |
| lost stolen | 0.0049 |

Independent probabilities do not need to sum to one. The raw request and response retain the catalogue values, while the UI uses short display titles.

## Validation evidence

- 57 CPU tests passed, including separate emotion/banking losses, masks, calibration, thresholds, version-3 save/reload, API catalogue isolation, and source-fold/rehearsal leakage checks.
- Both multi-label pages passed real-checkpoint desktop/mobile browser checks, threshold reuse, candidate editing, visible truncation, API error recovery, and integration-snippet syntax.
- The original banking choice, sentiment, and routing pages passed browser regression checks with the combined checkpoint.
- All combined training, validation, calibration, and test text sets are disjoint.
- NLU++ banking test text has no exact normalized overlap with the initial CPU or GoEmotions checkpoint training manifests.
- One emotion-calibration replay text matching a banking-training text was removed before recalibration. The training, validation, and test manifests and trained weights stayed fixed during that refit.
- CUDA-specific comparison checks were not run because no CUDA device is available.

## Saved artifacts

- Scores: `checkpoints/multilabel-banking/banking_multilabel_results.json`.
- Full five-head evaluation: `results/multilabel-banking_jevlet.json`.
- Exact source rows, fold IDs, and hashes: `checkpoints/multilabel-banking/data.json`.
- Baseline predictions: `checkpoints/multilabel-banking/banking_multilabel_baseline.json`.
- Browser responses/screenshots: `.cache/banking-multilabel-dashboard/verification.json`, `banking_multi-desktop.png`, and `banking_multi-mobile.png`.
- CLI request and actual response: `examples/banking-multilabel.json` and `examples/banking-multilabel-result.json`.

See [BANKING_MULTILABEL.md](BANKING_MULTILABEL.md) for the source dataset, label definitions, reproducible training command, API, and limits.
