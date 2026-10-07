# Jevlet

A small encoder that answers typed questions about text: choose an option, rate
sentiment from 1 to 5, or answer yes/no. Up to three questions share one sequence
and one DistilBERT encoder pass. Outputs are probabilities, not generated text.

Inspired by [TypeSafe's Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
[OpenAI Decisions](https://developers.openai.com/api/reference/resources/decisions/methods/create),
and [Laya](https://github.com/NandhaKishorM/laya). This project trains on labeled
public data; it does not reproduce those systems' capabilities or training.

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

## Train on your friend's GPU

Copy this repository to the GPU machine, then:

```sh
make setup TORCH_BACKEND=auto
make train DEVICE=cuda
make eval DEVICE=cuda
```

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
| smoke | flan_subset | choice | 0.6667 | 0.0909 | 0.1766 | 0.7198 | 0.4636 |
| smoke | flan_subset | score | 0.5833 | 0.7500 | 0.2809 | 1.0740 | 0.5761 |
| smoke | flan_subset | yesno | 0.7500 | 0.8819 | 0.1967 | 0.6003 | 0.3842 |
| smoke | jevlet | choice | 0.1667 | 0.0260 | 0.3061 | 0.7614 | 0.5284 |
| smoke | jevlet | score | 0.0000 | 1.5000 | 0.2521 | 1.6476 | 0.8188 |
| smoke | jevlet | yesno | 0.5000 | 0.4028 | 0.0034 | 0.6932 | 0.5001 |
| smoke | jevlet_flan_subset | choice | 0.1667 | 0.0260 | 0.3061 | 0.7614 | 0.5284 |
| smoke | jevlet_flan_subset | score | 0.0000 | 1.5000 | 0.2521 | 1.6476 | 0.8188 |
| smoke | jevlet_flan_subset | yesno | 0.5000 | 0.4028 | 0.0034 | 0.6932 | 0.5001 |
| smoke | no_brier | choice | 0.1667 | 0.0260 | 0.3071 | 0.7625 | 0.5295 |
| smoke | no_brier | score | 0.0000 | 1.5000 | 0.2535 | 1.6511 | 0.8204 |
| smoke | no_brier | yesno | 0.5000 | 0.4028 | 0.0034 | 0.6932 | 0.5001 |
| smoke | options10 | choice | 0.0833 | 0.0130 | 0.0169 | 2.3022 | 0.8999 |
| smoke | options10 | score | 0.0000 | 1.5000 | 0.2521 | 1.6476 | 0.8188 |
| smoke | options10 | yesno | 0.5417 | 0.4825 | 0.0384 | 0.6926 | 0.4995 |
| smoke | options2 | choice | 0.6667 | 0.0952 | 0.1655 | 0.6920 | 0.4989 |
| smoke | options2 | score | 0.0000 | 1.5000 | 0.2521 | 1.6476 | 0.8188 |
| smoke | options2 | yesno | 0.3750 | 0.4370 | 0.1284 | 0.6949 | 0.5018 |
| smoke | options5 | choice | 0.0833 | 0.0130 | 0.1171 | 1.6092 | 0.7999 |
| smoke | options5 | score | 0.0000 | 1.5000 | 0.2521 | 1.6476 | 0.8188 |
| smoke | options5 | yesno | 0.4583 | 0.5524 | 0.0450 | 0.6937 | 0.5005 |
| smoke | separate | choice | 0.2500 | 0.0303 | 0.2784 | 0.7616 | 0.5288 |
| smoke | separate | score | 0.0000 | 1.5000 | 0.2489 | 1.6486 | 0.8183 |
| smoke | separate | yesno | 0.5000 | 0.6389 | 0.0474 | 0.6964 | 0.5033 |
| smoke | single_question | choice | 0.2500 | 0.0346 | 0.2783 | 0.7614 | 0.5284 |
| smoke | single_question | score | 0.0000 | 1.5000 | 0.2521 | 1.6476 | 0.8188 |
| smoke | single_question | yesno | 0.5000 | 0.5833 | 0.0046 | 0.6932 | 0.5001 |
| smoke | tfidf | choice | 0.3333 | 0.0476 | 0.2503 | 0.7482 | 0.5154 |
| smoke | tfidf | score | 0.5000 | 1.2500 | 0.2495 | 1.5311 | 0.7695 |
| smoke | tfidf | yesno | 0.5417 | 0.5486 | 0.0407 | 0.6923 | 0.4992 |

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
