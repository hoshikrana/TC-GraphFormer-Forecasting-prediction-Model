.PHONY: help install install-dev test lint format pipeline pretrain train eval clean

PYTHON ?= python

help:
	@echo "TC-GraphFormer Build and Execution Targets:"
	@echo "  make install       Install production dependencies"
	@echo "  make install-dev   Install development and testing dependencies"
	@echo "  make test          Run complete unit and integration test suite"
	@echo "  make lint          Run flake8 and code checks"
	@echo "  make pipeline      Run tcprimed_pipeline data processing on pilot dataset"
	@echo "  make pretrain      Run self-supervised Masked Feature Modeling pretraining"
	@echo "  make train         Run multi-task training on TC-GraphFormer"
	@echo "  make eval          Evaluate models and baselines on validation and test sets"
	@echo "  make clean         Remove build, cache, and temporary files"

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m unittest discover tests

lint:
	flake8 tcprimed_pipeline src tests --max-line-length=120 --ignore=E203,W503

pipeline:
	$(PYTHON) -m tcprimed_pipeline --config configs/pilot.yaml

pretrain:
	$(PYTHON) src/training/pretrain.py --config configs/pretraining.yaml

train:
	$(PYTHON) src/training/train_multitask.py --config configs/multitask_training.yaml

eval:
	$(PYTHON) src/evaluation/evaluate_baselines.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache build dist *.egg-info

