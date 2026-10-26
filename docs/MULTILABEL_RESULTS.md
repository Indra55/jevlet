# Completed multi-label emotion run

Trained and evaluated locally on CPU using real human-annotated GoEmotions text.

Checkpoint: `checkpoints/multilabel-emotions`. Run status: **complete**.

Training: 3 epochs on 4,096 emotion texts, with 64 Banking77 and 64 SST-5 rehearsal texts. Total pipeline time: 24.8 minutes.
Validation and calibration: 384 emotion texts each. Test: 768 emotion texts. Seed: 42.

| Candidate-based metric | Jevlet | TF-IDF logistic |
|---|---:|---:|
| Micro-F1 | 0.7038 | 0.6088 |
| Macro-F1 | 0.5779 | 0.2784 |
| Precision | 0.6931 | 0.6546 |
| Recall | 0.7149 | 0.5689 |
| Exact candidate-set accuracy | 0.5820 | 0.4857 |
| Hamming loss | 0.0945 | 0.1149 |
| Mean Brier | 0.0647 | 0.0992 |
| Binary NLL | 0.2182 | 0.3246 |
| Label-probability ECE | 0.0147 | 0.0710 |

Shared validation-selected threshold: **0.38**. Candidate recall: **0.8236**.

These are one-seed subset results over 2–10 supplied labels, including shortlists with omitted true labels. They do not measure full-catalogue retrieval across all 28 emotions. Rare-label support is small; see per-label counts in the JSON results.

## Live example

Input: “Thank you so much! I am really happy you helped me.”

Selected: **gratitude, joy** at the saved threshold.

| Supplied label | Probability |
|---|---:|
| gratitude | 0.9628 |
| joy | 0.9174 |
| admiration | 0.1092 |
| sadness | 0.0222 |
| anger | 0.0086 |
| neutral | 0.0123 |

This is a real checkpoint response. Probabilities are independent and need not sum to one.

## Validation evidence

- 52 CPU tests passed, including the real tiny-model training/calibration/baseline/reload workflow and API checks.
- Existing banking, sentiment, and routing browser checks passed with both the original CPU checkpoint and the completed multi-label checkpoint.
- Trained multi-label checkpoint passed real desktop/mobile browser checks, selection-threshold reuse, editable candidates, truncation, integration snippet syntax, and API error recovery.
- Text and source-ID intersections between training, validation, calibration, and test were empty.
- CUDA-specific comparison tests were not run: this machine reports no CUDA device.

Screenshots and responses: `.cache/multilabel-dashboard/verification.json`, `emotions-desktop.png`, and `emotions-mobile.png`.

Model and baseline scores: `checkpoints/multilabel-emotions/multilabel_results.json`.
Full evaluation: `results/multilabel-emotions_jevlet.json`.
Source provenance and exact rows: `checkpoints/multilabel-emotions/data.json`.

Run `make demo` from the repository root, or explicitly serve the checkpoint using the commands in [MULTILABEL.md](MULTILABEL.md).
