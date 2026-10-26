# Multi-label banking

The combined checkpoint selects all applicable banking intent components from
supplied candidates. It shares one DistilBERT encoder with the existing emotion,
single-choice, sentiment, and yes/no heads. The banking and emotion multi-label
heads have separate weights, temperatures, and selection thresholds.

## Use the demo

```sh
make demo
# Or explicitly:
.venv/bin/python -m jevlet.cli serve --checkpoint checkpoints/multilabel-banking
```

Open http://127.0.0.1:8000/cases/banking_multi. Choose 2–10 candidate components,
enter a banking message, and run **Find banking intents**. The threshold slider
changes the selected set without another encoder pass. Hover over a candidate
chip to read its definition. Short display titles preserve the underlying
catalogue values in requests and response JSON.

For example, “I am trying to make a transfer but it is not working” can receive
an action component, a transfer/payment component, and a not-working component.
The labels are NLU++ intent modules, combining actions, objects, and states.
They differ from Banking77's 77 mutually exclusive intent classes. The original
Banking77 case remains available at `/cases/banking`. Slots are not extracted,
and the demo does not carry out any account changes or transactions.

## Real data and reproducibility

The official [NLU++ banking dataset](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/nlupp)
provides banking text with multiple annotated intent components. This extension
uses its 48 banking/general intent modules, excluding hotel-only modules.
The source revision is `57ec275d8078af65b7731c2a98be812d844a6d6b`, under CC-BY-4.0.
The [paper](https://arxiv.org/abs/2204.13021) describes the dataset;
this implementation uses one fixed split and does not reproduce its repeated
cross-validation experiment.

Source folds are assigned before training:

| Partition | Source folds | Banking examples after duplicate removal |
|---|---|---:|
| Training | 0–13 | 1,422 |
| Validation | 14–15 | 206 |
| Calibration | 16–17 | 207 |
| Test | 18–19 | 204 |

Text duplicates are removed with test, calibration, validation, then training
precedence. Candidate shortlists include all-positive, partially omitted, and
no-positive cases; original message text and intent annotations are unchanged.
Source-fold IDs and per-file SHA-256 hashes are saved in `data.json`.

Training starts from `checkpoints/multilabel-emotions` and adds a fifth head.
Each training epoch uses the 1,422 banking texts plus up to 64 Banking77, 64
SST-5, and 256 emotion replay examples. Evaluation splits include up to 128
emotion replay examples each. Replay text that belongs to a different NLU++
partition is excluded, preventing cross-domain split leakage. One overlapping
emotion calibration replay example was removed in the completed run; model
weights and training, validation, and test manifests were unchanged during the
subsequent calibration refit.

```sh
.venv/bin/python -m jevlet.cli multilabel --nlupp \
  --checkpoint checkpoints/multilabel-emotions \
  --run multilabel-banking --epochs 3
```

Use a new run name if that checkpoint already exists. The source archive is
cached locally; `--nlupp-archive path/to/archive.zip` supports an existing pinned
archive. `--device cuda` explicitly selects a CUDA device when available.

Masked binary cross-entropy and mean per-label Brier loss train the banking
head. Temperature scaling uses only the calibration partition. A shared banking
selection threshold is fitted by candidate micro-F1 on validation data. The
emotion head has independent calibration and threshold fitting. The TF-IDF
logistic baseline uses the same banking source partitions and candidate lists.

## API and CLI

```sh
curl http://127.0.0.1:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"case":"banking_multi","text":"I lost my card and want to freeze it.","options":["lost stolen","card","cancel close leave freeze","pin"]}'

.venv/bin/python -m jevlet.cli predict \
  --checkpoint checkpoints/multilabel-banking \
  --request examples/banking-multilabel.json
```

The typed question is `banking_multilabel`. Responses include the selected label
list, each independent probability, and the applied threshold. They have no
single confidence value for the entire set. An optional API `threshold` must be
strictly between zero and one. `/api/info` exposes banking labels, definitions,
threshold, source provenance, and matching held-out metrics. Checkpoints lacking
the banking head reject this case explicitly.

Version-3 checkpoints include both multi-label heads and both thresholds.
Version-1 and version-2 checkpoints remain loadable with their existing behavior.

## Evidence and limits

See [BANKING_MULTILABEL_RESULTS.md](BANKING_MULTILABEL_RESULTS.md) for measured
scores and an actual model response. These scores concern 2–10 supplied
candidates on one seeded source-fold split, not unrestricted classification
across all 48 modules or all 77 Banking77 classes. Candidate recall measures
how many true components are even present in the shortlist. Empty selection
means no supplied candidate passed the threshold; it is not validated
out-of-domain detection. Rare labels have limited held-out support.

```sh
.venv/bin/python -m pytest tests/test_core.py tests/test_server.py tests/test_multilabel.py -q
PLAYWRIGHT_BROWSERS_PATH=.cache/ms-playwright .venv/bin/python scripts/check_multilabel_dashboard.py --url http://127.0.0.1:8000 --case banking_multi
```

The browser script saves actual response JSON and desktop/mobile screenshots in
`.cache/banking-multilabel-dashboard/`. Existing emotion and original-task
browser scripts also run against the combined checkpoint.
