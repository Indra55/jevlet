PYTHON ?= python3.11
UV ?= uv
CONFIG ?= config.yaml
DEVICE ?= auto
TORCH_BACKEND ?= cpu
RUN ?= full
SEEDS ?= 42
export HF_HOME := $(CURDIR)/.cache/huggingface
export UV_CACHE_DIR := $(CURDIR)/.cache/uv
export MPLCONFIGDIR := $(CURDIR)/.cache/matplotlib
export TOKENIZERS_PARALLELISM := false
export USE_TF := 0
JEVLET = .venv/bin/python -m jevlet.cli --config $(CONFIG)

.PHONY: setup test smoke train eval baselines ablate report paper all
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
paper: report
	@command -v latexmk >/dev/null || { printf '%s\n' 'PDF compilation requires latexmk, a LaTeX distribution, and IEEEtran. Generated tables are in paper/generated/.'; exit 1; }
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
all: smoke
	$(MAKE) train
	$(MAKE) eval
	$(MAKE) paper
