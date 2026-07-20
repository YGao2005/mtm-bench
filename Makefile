.PHONY: install lint test reproduce run-all splits all help

install:  ## Install the package in editable mode with dev deps
	pip install -e ".[dev]"

lint:  ## Run ruff linter
	ruff check

test:  ## Run the test suite
	python -m pytest -q

reproduce:  ## Reproduce all paper numbers (offline, self-checking)
	python scripts/reproduce_paper.py

run-all:  ## Score all shipped cells (test split, text output)
	python -m mtm_bench run-all

splits:  ## Show the frozen dev/test split table
	python -m mtm_bench splits

all: lint test reproduce  ## Full CI gate (lint + test + reproduce)

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
