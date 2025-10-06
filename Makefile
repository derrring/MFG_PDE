.PHONY: help test test-fast lint type-check coverage clean install format

help:
	@echo "MFG_PDE Development Commands"
	@echo "============================"
	@echo "test          - Run all tests"
	@echo "test-fast     - Run fast tests only"
	@echo "lint          - Run linting checks"
	@echo "type-check    - Run mypy type checking"
	@echo "coverage      - Run tests with coverage report"
	@echo "format        - Format code with ruff"
	@echo "clean         - Remove cache and build files"
	@echo "install       - Install package in development mode"

test:
	pytest

test-fast:
	pytest -m "not slow" -x

lint:
	ruff check mfg_pde/

type-check:
	mypy mfg_pde/ --show-error-codes

coverage:
	pytest --cov=mfg_pde --cov-report=html --cov-report=term

format:
	ruff format mfg_pde/ tests/ examples/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage

install:
	pip install -e ".[dev]"

.DEFAULT_GOAL := help
