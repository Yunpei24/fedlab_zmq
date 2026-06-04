# FedLab ZMQ — common tasks. Run `make help` for the list.
.DEFAULT_GOAL := help
PY ?= python

.PHONY: help install install-dev test smoke format lint clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime deps + the package (editable)
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

install-dev:  ## Install dev/eval deps (pytest, black, isort, ruff) + package
	$(PY) -m pip install -r requirements-dev.txt
	$(PY) -m pip install -e .

test:  ## Run the test suite (canonical guardrail: tests/test_flop_cost.py)
	$(PY) -m pytest

smoke:  ## Fast end-to-end sanity run (< 1 min, CPU)
	$(PY) run_experiment.py --config configs/smoke.yaml

format:  ## Apply isort + black
	isort .
	black .

lint:  ## Run ruff (no changes)
	ruff check .

clean:  ## Remove caches and __pycache__
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
