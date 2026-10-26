PYTHON ?= python3.11
UV ?= uv
CONFIG ?= config.yaml
DEVICE ?= auto
TORCH_BACKEND ?= cpu
RUN ?= full
SEEDS ?= 42
CHECKPOINT ?= checkpoints/multilabel-banking
PORT ?= 8000
TECTONIC ?= tectonic
export HF_HOME := $(CURDIR)/.cache/huggingface
export UV_CACHE_DIR := $(CURDIR)/.cache/uv
export MPLCONFIGDIR := $(CURDIR)/.cache/matplotlib
export TOKENIZERS_PARALLELISM := false
export USE_TF := 0
JEVLET = .venv/bin/python -m jevlet.cli --config $(CONFIG)

.PHONY: setup setup-demo demo test smoke train eval baselines ablate report paper paper-source all multilabel banking-multilabel
setup-demo:
	$(UV) pip install --python .venv/bin/python -e '.[demo,demo-test]'
demo:
	$(JEVLET) serve --checkpoint $(CHECKPOINT) --port $(PORT)
setup:
	$(UV) venv --allow-existing --python $(PYTHON) .venv
	$(UV) pip install --python .venv/bin/python --torch-backend $(TORCH_BACKEND) -e .
test:
	.venv/bin/python -m pytest -q
smoke: test
	$(JEVLET) smoke
train:
	$(JEVLET) train --run $(RUN) --device $(DEVICE)
eval:
	$(JEVLET) eval --run $(RUN) --device $(DEVICE)
baselines:
	$(JEVLET) baselines --run $(RUN) --device $(DEVICE) --seeds $(SEEDS)
ablate:
	$(JEVLET) ablate --run $(RUN) --device $(DEVICE) --seeds $(SEEDS)
report:
	$(JEVLET) report
paper-source:
	.venv/bin/python scripts/rewrite_jevlet_paper.py
paper: paper-source
	@if command -v latexmk >/dev/null; then cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex; \
	elif command -v "$(TECTONIC)" >/dev/null; then cd paper && "$(TECTONIC)" --keep-logs paper.tex; \
	else printf '%s\n' 'Install latexmk or Tectonic to compile the regenerated manuscript.'; exit 1; fi
	mkdir -p output/pdf
	cp paper/paper.pdf output/pdf/paper.pdf
	cp paper/paper.log output/pdf/paper.log
all: smoke
	$(MAKE) train
	$(MAKE) eval
	$(MAKE) paper

multilabel:
	$(JEVLET) multilabel --go-emotions --run multilabel-emotions --device $(DEVICE)

banking-multilabel:
	$(JEVLET) multilabel --nlupp --run multilabel-banking --device $(DEVICE)
