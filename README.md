# Jevlet

A small encoder that answers typed questions about text: choose an option, select
multiple labels, rate sentiment from 1 to 5, or answer yes/no. Up to three questions share one sequence
and one DistilBERT encoder pass. Outputs are probabilities, not generated text.

Inspired by [TypeSafe's Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
[OpenAI Decisions](https://developers.openai.com/api/reference/resources/decisions/methods/create),
and [Laya](https://github.com/NandhaKishorM/laya). This project trains on labeled
public data; it does not reproduce those systems' capabilities or training.

## Multi-label banking

The combined checkpoint also selects **multiple banking intent components** from
real NLU++ annotations. It has a shared encoder with separate banking and emotion
multi-label heads, each with its own calibration and selection threshold.

```sh
.venv/bin/python -m jevlet.cli multilabel --nlupp --run multilabel-banking
make demo
```

Open **http://127.0.0.1:8000/cases/banking_multi**. NLU++ uses 48 modular actions,
objects, and states, distinct from the 77 Banking77 choice classes. The completed
run achieved **86.5% candidate micro-F1** on 204 held-out messages, versus **81.3%**
for TF-IDF logistic, on one fixed source-fold split with 2–10 supplied candidates.
See [the banking guide](docs/BANKING_MULTILABEL.md) and
[measured results](docs/BANKING_MULTILABEL_RESULTS.md).

## Multi-label emotion selection

Jevlet now supports **zero, one, or several emotion labels per text**, using real
human annotations from GoEmotions and a dedicated sigmoid option head. Training,
calibration, validation-based threshold selection, held-out set metrics, and a
TF-IDF logistic baseline are included. The completed CPU run reached **70.4%
candidate micro-F1** on 768 held-out texts, versus **60.9%** for the baseline;
see [measured results and live example](docs/MULTILABEL_RESULTS.md).

```sh
# Train a new multi-label checkpoint from the existing CPU checkpoint:
.venv/bin/python -m jevlet.cli multilabel --go-emotions --run multilabel-emotions
# Launch the resulting checkpoint:
make demo
```

Open **http://127.0.0.1:8000/cases/emotions**. Choose 2–10 candidate emotions and
inspect independent probabilities; the threshold slider updates the selected set
without another model call. The checkpoint includes the original task heads.
Existing checkpoints remain loadable with their original behavior.

See [the multi-label guide](docs/MULTILABEL.md) for reproducible data splits,
measured artifacts, API examples, limitations, and verification. These are
candidate-based emotion predictions, not a multi-intent banking benchmark.

## Explore the local dashboard

After training, launch **Jevlet Lab**, a custom browser dashboard with five live
case studies. It loads your checkpoint once and runs real, calibrated inference
on CPU. It does not retrain, call a hosted API, or save submitted messages.

```sh
make setup-demo                # add the web dependencies to your existing .venv
make demo                      # defaults to checkpoints/multilabel-banking, CPU, port 8000
```

Open **http://127.0.0.1:8000**. Each case includes editable examples, a probability
display, usage instructions, an explanation, a Python integration snippet, and
evaluation notes drawn from the matching full results file when available.

| Case study | What to try |
| --- | --- |
| Multiple banking intents | Select all applicable NLU++ banking components, with independent probabilities and a threshold. |
| Multiple emotions | Select several emotion labels from GoEmotions; inspect independent probabilities and adjust the selection threshold. |
| Banking support | Select 2–10 Banking77 intents and up to two topic checks. All questions share one encoder pass. |
| Review sentiment | Score a short English movie review on the fixed SST-5 five-level rubric; inspect the expected score and class distribution. |
| Confidence routing | Preview a support queue or human review using the banking choice confidence. Move the threshold without rerunning the model. |

The routing rule is an application policy demonstration, not an additional
learned capability; it does not send tickets or take financial actions. The 90%
default is illustrative. Choice confidence is relative to the supplied candidates,
and the true intent must be included for a valid candidate-based test. Sentiment
is the weaker head; the evaluation notes show its measured limitations.

To select another complete checkpoint or port:

```sh
make demo CHECKPOINT=checkpoints/friend-full PORT=8001
# Or, directly (also works without Make):
.venv/bin/python -m jevlet.cli serve --checkpoint checkpoints/cpu-full --port 8000
```

Windows PowerShell, using an already-created environment and trained checkpoint:

```powershell
uv pip install --python .venv\Scripts\python.exe -e ".[demo,demo-test]"
.\.venv\Scripts\python.exe -m jevlet.cli serve --checkpoint checkpoints/full
```

The server binds to `127.0.0.1` by default. Interactive API documentation is at
`/docs`: `GET /api/info` reports the loaded model and label catalogue, and
`POST /api/predict` accepts a `case` (`banking`, `reviews`, or `routing`), `text`,
and (for banking/routing) `options` and optional `checks`. Questions are built
from the trained rubrics. Unknown intents and invalid candidate counts are
rejected. Oversized context is truncated with a visible notice. A missing or
uncalibrated checkpoint fails at startup; there is no synthetic-response fallback.

Dashboard contract tests run with `make test` after `make setup-demo`. They are
skipped when the optional web dependencies are absent; the core training project
does not require a web server. Static assets ship with the Python package, with
no frontend build step, external fonts, or CDN dependencies.

An optional live browser check covers the real model path, desktop/mobile pages,
threshold changes, truncation, and visible error recovery. With the server running:

```sh
uv pip install --python .venv/bin/python playwright
PLAYWRIGHT_BROWSERS_PATH=.cache/ms-playwright .venv/bin/python -m playwright install chromium
PLAYWRIGHT_BROWSERS_PATH=.cache/ms-playwright .venv/bin/python scripts/check_dashboard.py
```

This stores screenshots and actual response JSON in `.cache/dashboard-checks/`.
Browser tooling is not needed to run the dashboard.

## Setup and CPU smoke check

Requirements: Python 3.10+, [uv](https://docs.astral.sh/uv/), and Make. Python 3.11
is the default tested interpreter. All caches and downloaded assets live under
`.cache/`; initial setup and downloads require internet, without API keys.

```sh
make setup                     # CPU PyTorch
make smoke                     # tests + real tiny training/calibration/evaluation
```

If your Python executable differs: `make setup PYTHON=/path/to/python3.11`.
The smoke run uses pretrained DistilBERT and small real Banking77/SST-5 subsets.
Its five-minute CPU target excludes first-time installation and downloads.
Smoke results demonstrate pipeline execution, not useful model accuracy.

## Train on your GPU

Copy this repository to the GPU machine, then:

```sh
make setup TORCH_BACKEND=auto
make train DEVICE=cuda
make eval DEVICE=cuda
```

To train, calibrate, evaluate, and update reports automatically in one command:
`.venv/bin/python -m jevlet.cli run --run cpu-full --device cpu`.
The run writes `checkpoints/cpu-full/progress.json`, `run_status.json`, and the
usual checkpoint/results artifacts. Progress estimates exclude validation,
calibration, and evaluation and may change as the CPU's speed changes.

New runs show live training, validation, calibration, and evaluation progress
bars with batch counts, loss, speed, and ETA. To monitor a run that was started
before the progress bars were added, open another terminal and run:

```sh
.venv/bin/python -m jevlet.cli watch --run cpu-full
```

This reads the existing progress files and does not restart training. Ctrl+C
in the monitor stops only the monitor. Its ETA excludes validation, calibration,
and evaluation; the JSON progress file updates every 25 training batches.
Terminal output changes apply to newly started processes. Training currently
starts from pretrained weights when invoked again, so restarting a running job
does not resume its previous batch or saved epoch.

`TORCH_BACKEND=auto` asks uv to select PyTorch for the detected hardware.
Install the appropriate NVIDIA driver first. Check CUDA with
`.venv/bin/python -c 'import torch; print(torch.cuda.is_available())'`.
Default full training uses batch size 8, accumulation 2, length 256, up to five
epochs, and CUDA mixed precision. Reduce batch size in `config.yaml` if necessary;
increase accumulation to retain effective batch size. No GPU memory or runtime
claim is made before measuring your hardware.

Copy `checkpoints/full/` back for local prediction. It contains weights,
tokenizer, encoder configuration, temperatures, source revisions, data manifest,
and training history. Prediction does not need the original model download.

### Windows (PowerShell, NVIDIA GPU)

Native Windows can invoke the Python CLI directly, without Make or WSL.
Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Reopen PowerShell after installation, open the project folder, then run:

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe --torch-backend=auto -e .

# Must print True before proceeding with CUDA training.
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"

.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m jevlet.cli smoke
.\.venv\Scripts\python.exe -m jevlet.cli train --device cuda
.\.venv\Scripts\python.exe -m jevlet.cli eval --device cuda
```

For the research suite:

```powershell
.\.venv\Scripts\python.exe -m jevlet.cli baselines --device cuda
.\.venv\Scripts\python.exe -m jevlet.cli ablate --device cuda
.\.venv\Scripts\python.exe -m jevlet.cli report
```

No environment activation is required. uv can download Python 3.11 when it is
missing. CUDA requires a supported NVIDIA GPU and driver; an AMD/Intel GPU needs
a different training setup. Native Windows execution has not been tested in
this workspace; the pipeline checks were performed on Linux CPU.

## Local typed prediction

```sh
.venv/bin/jevlet predict --checkpoint checkpoints/full --request examples/banking.json
```

```python
from jevlet import Jevlet

model = Jevlet.load("checkpoints/full", device="cpu")
result = model.predict("I was charged twice for my card payment.", [
    {"id": "intent", "type": "choice", "text": "What is the customer's intent?",
     "options": ["card arrival", "transaction charged twice"]},
    {"id": "duplicate", "type": "yesno",
     "text": "Is this about transaction charged twice?"},
])
print(result)
```

Requests contain nonempty `text` and a `questions` list of 1–3 questions. Each
question has a unique `id`, `type` (`choice`, `score`, `yesno`), and `text`.
Choice requires 2–10 distinct string `options`. Score always uses the learned
five-level sentiment scale: very negative, negative, neutral, positive, very
positive. A score request is included in `examples/sentiment.json`.

Answers retain question order. `answer` is an option string, an integer 1–5, or
a boolean. `probabilities` follow supplied option order, sentiment levels 1–5,
or `[false, true]`. `confidence` is maximum class probability. Score also returns
`expected_score`. Context truncation is reported as `context_tokens_dropped`.
Questions are never truncated; oversized question blocks are rejected.

This model learns banking and sentiment templates. Accepting question text in
the interface does not imply general instruction following, unseen-label
transfer, arbitrary scoring rubrics, or reliable out-of-domain confidence.

## Research workflow

### CUDA-only classifier comparison

After a full run has produced `checkpoints/full/data.json` and `training.json`,
the dedicated comparison suite freezes those source splits and retrains two
Jevlet sentiment variants plus five external classifier configurations:

```powershell
python -m pytest tests/test_gpu_comparison.py -q
python -m jevlet.gpu_compare
```

The suite requires CUDA and never falls back to CPU model fitting or inference.
CPU tokenization, TF-IDF extraction, and metric summaries remain necessary.
It uses seed 42, validation-based checkpoint/variant selection, and independent
calibration data. Linear classifiers use custom PyTorch logistic or squared-hinge
objectives on CUDA; transformer comparators encode context text and classify
known label IDs. These differ from Jevlet's marker-conditioned interface.

New artifacts live in `checkpoints/gpu_comparison/`, with a ten-second device
monitor in `vram.jsonl`, per-model progress, predictions, and the consolidated
`results/gpu_comparison_summary.json`. Completed models are reused when the
command is rerun. A smaller selection can be run with `--models`, for example
`--models jevlet_softmax bert_classifier`.

Every successful comparison run prints a saved test-metric table, including the
selected refined Jevlet when its separate results exist. Completed models are
skipped rather than trained again. To inspect results and recorded epoch losses
without starting training, evaluation, or CUDA work, use:

```powershell
python -m jevlet.gpu_compare --show-results --history
```

The history shows losses from previous runs, plus their recorded training time
and peak allocated GPU memory. Omit `--history` for just the comparison table.

The current Jevlet-focused IEEE manuscript uses the refinement workflow
below and places external comparisons in its results and conclusion.

### Expanded question validation and Jevlet refinement

```powershell
python -m jevlet.refine
python scripts/rewrite_jevlet_paper.py
python scripts/collect_jevlet_examples.py
```

The paper generator defaults to IEEE conference format. Use
`--format elsevier` to regenerate the author-year lab-template version.

This continues from `checkpoints/gpu_comparison/jevlet_softmax` on CUDA, using
five fixed question views for each of the 1,549 validation contexts. The
resulting 7,745 requests and 17,735 decisions are correlated views, not a larger
independent dataset or five-fold cross-validation. Training varies the same
five question templates. Three predefined seed/learning-rate settings each
receive at most three additional epochs. Expanded-validation sentiment accuracy
selects the checkpoint, with one-percentage-point choice/binary regression
guardrails and the initial checkpoint as a fallback. Test scores do not rank
the candidates. Refinements share their initial checkpoint; they do not measure
independent from-scratch seed variability.

Artifacts are saved under `checkpoints/jevlet_refinement/` and
`results/jevlet_refinement.json`. A ten-second GPU monitor continues throughout
the search. The separate illustrated model guide is generated by
`scripts/create_jevlet_explainer.py` using ReportLab and the recorded CUDA
examples, and written to `output/pdf/jevlet_model_explained.pdf`.

All settings are in `config.yaml`; command flags select run, device, smoke,
baseline, and seeds. Explicit stages avoid launching expensive experiments
accidentally:

```sh
make baselines DEVICE=cuda          # TF-IDF, three separate encoders, FLAN-T5
make ablate DEVICE=cuda             # packing, Brier, temperature, option count
make report                        # regenerate README and LaTeX tables
make paper                         # also compile PDF; requires latexmk + IEEEtran
make all DEVICE=cuda               # smoke, core training/evaluation, paper
```

For a three-seed experiment suite:

```sh
make baselines DEVICE=cuda SEEDS="42 43 44"
make ablate DEVICE=cuda SEEDS="42 43 44"
```

Run one baseline explicitly:

```sh
.venv/bin/jevlet baselines --run full --device cpu --only tfidf
```

Baselines and ablations reuse completed checkpoints. The FLAN baseline uses a
fixed, source-disjoint test subset controlled by `eval.flan_limit`; its results
include the subset size and must be compared with Jevlet on that same subset.
It scores candidate answer strings by length-normalized log-likelihood, with
no free-form generation or teacher-generated training data.

Evaluation writes `results/*.json`, predictions under `results/predictions/`,
and reliability/risk–coverage plots under `results/figs/`. Each run records
configuration, package versions, dataset/model revisions, and sample counts.
Paper tables exclude smoke runs. Missing results stay marked as not run.

### Data and evaluation boundaries

Banking77 contributes intent choices and balanced derived yes/no pairs. A real
three-question banking request contains choice + two independently sampled
yes/no questions. SST-5 contributes sentiment scores; it does not label banking
sentiment. Training includes standalone questions and packs of 2–3 questions.

Use the Parquet files in `legacy-datasets/banking77`, avoiding deprecated remote
dataset scripts, and `SetFit/sst5` JSONL. Split original contexts before deriving
questions. Remove normalized duplicate texts across splits with test taking
precedence; record removals. Banking train is split 80/10/10; SST development
is split equally into validation/calibration. Test labels never select a model
or fit temperatures. Training choices resample per epoch; test requests are fixed.

Candidate sets always include the true intent: this is a shortlist benchmark,
not all-77 intent detection. Both positive and negative yes/no queried labels
appear among packed choice options, preventing an option-presence shortcut.
Yes/no is balanced for this experiment; confidence is not calibrated for the
natural prevalence of arbitrary intent queries. Results should be read per
head and option count, not as a universal confidence guarantee.

Choice macro-F1 maps shuffled positions to original intent IDs, over all 77
classes. Score MAE uses the discrete predicted level; expected-score MAE is
reported separately. Brier sums squared class-probability errors (including
both classes for yes/no). ECE uses 15 equal-width bins and top-label confidence.
Risk–coverage groups tied confidence values; sentiment risk is MAE divided by 4.

Latency compares exactly the same retained context tokens: packed sequences,
three sequential passes, and separate question sequences batched in one call.
Report forward-only and packing + forward time separately; neither includes
network, initial checkpoint loading, or JSON serialization. Batch 32 reports
amortized ms/request, not individual request response latency. GPU timings use
synchronization; unavailable GPU measurements are explicitly marked.

### Measured results

Auto-generated calibrated test metrics. F1 applies to choice, MAE to score,
and AUROC to yes/no. **Smoke rows are pipeline checks only.**

<!-- RESULTS:START -->

| Run | Model | Head | Accuracy | F1 / MAE / AUROC | ECE | NLL | Brier |
|---|---|---|---:|---:|---:|---:|---:|
| cpu-full | jevlet | choice | 0.9627 | 0.9625 | 0.0062 | 0.1092 | 0.0546 |
| cpu-full | jevlet | score | 0.3624 | 0.8462 | 0.1891 | 1.3977 | 0.7423 |
| cpu-full | jevlet | yesno | 0.9768 | 0.9962 | 0.0048 | 0.0758 | 0.0383 |
| full | jevlet | choice | 0.9701 | 0.9700 | 0.0063 | 0.0923 | 0.0459 |
| full | jevlet | score | 0.3520 | 0.8593 | 0.2122 | 1.4426 | 0.7502 |
| full | jevlet | yesno | 0.9799 | 0.9969 | 0.0089 | 0.0686 | 0.0333 |
| gpu_compare | bert_classifier | choice | 0.9708 | 0.9707 | 0.0087 | 0.0790 | 0.0411 |
| gpu_compare | bert_classifier | score | 0.5312 | 0.5443 | 0.0212 | 1.0793 | 0.5921 |
| gpu_compare | bert_classifier | yesno | 0.9040 | 0.9911 | 0.0315 | 0.2556 | 0.1476 |
| gpu_compare | distilbert_classifier | choice | 0.9721 | 0.9720 | 0.0062 | 0.0880 | 0.0453 |
| gpu_compare | distilbert_classifier | score | 0.4923 | 0.5837 | 0.0272 | 1.1337 | 0.6172 |
| gpu_compare | distilbert_classifier | yesno | 0.8948 | 0.9884 | 0.0255 | 0.2695 | 0.1601 |
| gpu_compare | jevlet_ordinal | choice | 0.9704 | 0.9703 | 0.0067 | 0.0924 | 0.0459 |
| gpu_compare | jevlet_ordinal | score | 0.3520 | 0.8584 | 0.2115 | 1.4426 | 0.7501 |
| gpu_compare | jevlet_ordinal | yesno | 0.9799 | 0.9969 | 0.0091 | 0.0687 | 0.0334 |
| gpu_compare | jevlet_softmax | choice | 0.9545 | 0.9546 | 0.0101 | 0.1390 | 0.0707 |
| gpu_compare | jevlet_softmax | score | 0.5222 | 0.5724 | 0.0221 | 1.1062 | 0.6042 |
| gpu_compare | jevlet_softmax | yesno | 0.9683 | 0.9931 | 0.0064 | 0.0973 | 0.0506 |
| gpu_compare | roberta_classifier | choice | 0.9773 | 0.9772 | 0.0038 | 0.0700 | 0.0348 |
| gpu_compare | roberta_classifier | score | 0.5412 | 0.5059 | 0.0307 | 1.0251 | 0.5731 |
| gpu_compare | roberta_classifier | yesno | 0.9328 | 0.9927 | 0.0265 | 0.2001 | 0.1086 |
| gpu_compare | tfidf_logistic | choice | 0.9786 | 0.9786 | 0.0086 | 0.0666 | 0.0328 |
| gpu_compare | tfidf_logistic | score | 0.3986 | 0.8394 | 0.0386 | 1.3972 | 0.7170 |
| gpu_compare | tfidf_logistic | yesno | 0.9596 | 0.9946 | 0.0078 | 0.1058 | 0.0613 |
| gpu_compare | tfidf_svm | choice | 0.9776 | 0.9776 | 0.0084 | 0.0662 | 0.0328 |
| gpu_compare | tfidf_svm | score | 0.3977 | 0.8516 | 0.0266 | 1.4111 | 0.7185 |
| gpu_compare | tfidf_svm | yesno | 0.9573 | 0.9939 | 0.0045 | 0.1145 | 0.0658 |
| refinement | jevlet | choice | 0.9597 | 0.9597 | 0.0073 | 0.1198 | 0.0614 |
| refinement | jevlet | score | 0.5172 | 0.5674 | 0.0344 | 1.1039 | 0.5997 |
| refinement | jevlet | yesno | 0.9708 | 0.9937 | 0.0063 | 0.0924 | 0.0473 |
| smoke | jevlet | choice | 0.2500 | 0.0303 | 0.2782 | 0.7614 | 0.5285 |
| smoke | jevlet | score | 0.3333 | 1.6667 | 0.0897 | 1.6077 | 0.7994 |
| smoke | jevlet | yesno | 0.4167 | 0.5347 | 0.0835 | 0.6932 | 0.5000 |

<!-- RESULTS:END -->

## Attribution and paper

Laya's marker scoring, option masks, held-out temperature fitting, and confidence
definition informed this implementation. Jevlet's heads and multi-question
sequence packing are implemented independently; it has no Laya runtime
dependency or checkpoint compatibility. Laya's runtime builds a separate row
per question, which is included as a benchmark comparison here.

References and limitations live in `paper/paper.tex` and `paper/references.bib`.
The paper makes no fabricated result or speedup claims. Banking77 is CC BY 4.0;
the SST-5 mirror has limited licensing metadata, so consult the original
Stanford dataset terms before redistributing data. Downloaded data and model
weights are excluded from version control.
