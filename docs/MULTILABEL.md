# Multi-label emotion selection

Banking is now supported alongside emotions in the combined checkpoint. See
[BANKING_MULTILABEL.md](BANKING_MULTILABEL.md) for the NLU++ task and training recipe.
The default demo loads `checkpoints/multilabel-banking`; the emotion-only checkpoint
remains available explicitly.


Jevlet can now select zero, one, or several supplied emotion labels for the same
text. This extends the marker-based shared encoder with a dedicated option scorer.
The original single-choice, ordinal-sentiment, and binary tasks remain distinct.

## Run the finished demo

From the repository root, with the existing Python environment:

```sh
make demo
# Emotion-only checkpoint:
.venv/bin/python -m jevlet.cli serve --checkpoint checkpoints/multilabel-emotions
```

Open http://127.0.0.1:8000/cases/emotions. Choose 2–10 candidate emotions, edit the
text, and select **Find emotions**. The slider changes the selected set using the
existing probabilities and makes no additional model request. The response JSON
retains the server's original threshold; the display states the current slider
threshold. Existing checkpoints can be served explicitly with `--checkpoint`.

The catalogue contains 27 emotions plus Neutral. Neutral is an annotated label;
it is different from an empty prediction. No selected labels means no supplied
candidate passed the threshold. It does not prove that text is out of domain.
Predictions describe text annotations, not a person's psychological condition.

## Train reproducibly

```sh
.venv/bin/python -m jevlet.cli multilabel --go-emotions \
  --checkpoint checkpoints/cpu-full --run multilabel-emotions --epochs 3
```

Use a new run name for each invocation. Existing runs are not overwritten and
training does not resume an interrupted run. This command fine-tunes the existing
checkpoint, calibrates probabilities, selects a threshold, evaluates held-out
examples, and runs a TF-IDF logistic baseline on the same data split and candidate
lists. It runs on CPU by default; `--device cuda` explicitly selects a GPU.

Default subsets: 4,096 GoEmotions training texts, 384 validation texts, 384
calibration texts, and 768 test texts. Each split additionally includes up to 64
examples from each existing task domain for rehearsal and task regression checks.
`--train-samples`, `--eval-samples`, and `--legacy-per-domain` change those counts;
the test subset is twice `--eval-samples`. These are subset results, not a claim
that the complete dataset was trained or evaluated.

Data: [GoEmotions simplified](https://huggingface.co/datasets/google-research-datasets/go_emotions),
revision `add492243ff905527e67aeb8b80c082af02207c3`, Apache-2.0.
[Original Google Research source and citation](https://github.com/google-research/google-research/tree/master/goemotions).
The data consists of human-annotated Reddit comments. Text and source-ID duplicates
are removed with test-first precedence before subset selection. The official
training and test partitions remain separate. The official development partition
is shuffled deterministically and divided into validation and calibration halves.
Source revision, removals, source IDs, and sampled rows are saved in `data.json`.

## Output and evaluation contract

A multi-label question uses `type: "multilabel"`, 2–10 distinct `options`, and an
optional threshold strictly between zero and one. The dedicated linear scorer
consumes each `[OPT]` representation. Sigmoid produces separate probabilities;
they do not need to sum to one. Masked binary cross-entropy trains observed labels,
plus mean per-label Brier loss using the configured Brier weight. Unknown targets
are `-1` and contribute neither loss nor evaluation decisions.

Temperature scaling minimizes binary NLL on calibration data. After choosing the
checkpoint using validation loss, a shared threshold is selected by candidate
micro-F1 on validation data over 0.05–0.95 in increments of 0.01. Test examples do
not choose model weights, temperature, or threshold. Individual label decisions
are separate sigmoid outputs; the shared encoder can still learn correlations.

Evaluation measures micro-F1, macro-F1, per-label precision/recall/F1, exact
candidate-set accuracy, Hamming loss, binary NLL, mean Brier, and label-probability
ECE. Partially annotated examples are excluded from exact-set accuracy. Candidate
recall reports the fraction of true labels present in the supplied shortlist.
The deterministic candidate recipe includes all-positive, partially omitted, and
no-positive shortlists; original human annotations are retained unchanged.

All these results concern supplied candidates. They do not establish full
28-label retrieval, unseen-label transfer, natural multi-intent banking
performance, or a calibrated probability that an entire label set is correct.
Existing single-choice metrics are reported separately. The baseline trains one
binary logistic classifier per label using TF-IDF text features, then calibrates
and selects its own threshold using the same held-out partitions. A smoothed
prior is used when a label has insufficient class variation in training.

Artifacts:

- `checkpoints/<run>/model.safetensors` and `jevlet.json`: version-2 checkpoint,
  including the new head, fourth temperature, and selected threshold.
- `checkpoints/<run>/calibration.json`: fitted temperatures and threshold evidence.
- `checkpoints/<run>/multilabel_results.json`: model and baseline metrics.
- `checkpoints/<run>/multilabel_baseline.json`: baseline metrics and predictions.
- `results/<run>_jevlet.json` and `results/predictions/<run>_jevlet.json`: full
  measured evaluation and prediction records.
- `checkpoints/<run>/run_status.json`: completion status and elapsed time.

Version-1 checkpoints load with their original three heads. Multi-label requests
to them fail explicitly. They are never relabeled as multi-label checkpoints.

## API and CLI

```sh
curl http://127.0.0.1:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"case":"emotions","text":"Thank you! I am so happy you helped.","options":["gratitude","joy","anger","neutral"]}'

.venv/bin/python -m jevlet.cli predict \
  --checkpoint checkpoints/multilabel-emotions --request examples/emotions.json
```

The multi-label answer contains a list of selected labels, the candidate list,
independent probabilities, the threshold, and `probability_semantics`. It does not
assign one maximum-probability confidence value to the whole set. `/api/info`
reports supported emotion labels, the saved threshold, dataset provenance, and
matching measured results. Unsupported labels and checkpoints fail visibly.

## Reviewed banking annotations

For a future banking extension, `multilabel --data annotations.jsonl` accepts
reviewed JSONL records using the existing Banking77 catalogue names:

```json
{"id":"example-1","split":"train","text":"...","labels":["card_arrival","change_pin"],"annotation_complete":true,"source_ids":["original-1"]}
```

Supply nonempty train, validation, calibration, and test partitions. For partial
annotations, set `annotation_complete` to false and list known absent labels in
`negative_labels`; unlisted labels are unknown. Text/source overlaps across
partitions are rejected. This import path does not supply banking annotations or
claim that a reviewed multi-label banking model already exists. The emotion demo
is available for checkpoints trained with the GoEmotions catalogue, including the
combined banking/emotion checkpoint.

## Verification

```sh
.venv/bin/python -m pytest tests/test_core.py tests/test_server.py tests/test_multilabel.py -q
PLAYWRIGHT_BROWSERS_PATH=.cache/ms-playwright .venv/bin/python scripts/check_multilabel_dashboard.py --url http://127.0.0.1:8000
```

Offline tests exercise real tiny encoder forward/backward passes, unknown-label
and padding masks, checkpoint compatibility/reload, variable candidate sets,
calibration, threshold selection, source leakage checks, API validation, and a
training-to-baseline integration test. The browser script calls the real loaded
checkpoint and saves screenshots and response JSON. CUDA-only comparison tests
require a CUDA device and are separate from these CPU checks.
