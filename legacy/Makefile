# Latent Space Reasoning Engine - Development Makefile
# Works on Windows (with Git Bash/MSYS2), macOS, and Linux

.PHONY: install install-dev test test-fast test-cov lint format check clean check-gpu demo help

# Default target
.DEFAULT_GOAL := help

# ============================================================================
# Installation
# ============================================================================

install:  ## Install package in editable mode
	pip install -e .

install-dev:  ## Install package with dev dependencies
	pip install -e ".[dev]"

install-all:  ## Install package with all optional dependencies
	pip install -e ".[all]"

# ============================================================================
# Testing
# ============================================================================

test:  ## Run all tests
	pytest tests/ -v

test-fast:  ## Run unit tests only (skip integration)
	pytest tests/ -v --ignore=tests/integration

test-cov:  ## Run tests with coverage report
	pytest tests/ -v --cov=src/latent_reasoning --cov-report=term-missing --cov-report=html

# ============================================================================
# Code Quality
# ============================================================================

lint:  ## Run linter (ruff)
	ruff check src/ tests/

format:  ## Format code (ruff)
	ruff format src/ tests/
	ruff check --fix src/ tests/

check:  ## Run all code quality checks
	ruff check src/ tests/
	ruff format --check src/ tests/

typecheck:  ## Run type checker (mypy)
	mypy src/latent_reasoning

# ============================================================================
# Development
# ============================================================================

check-gpu:  ## Check GPU availability
	python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

demo:  ## Run the demo script
	python demo.py

run:  ## Run a quick test query
	latent-reason run "How to implement a cache?" --encoder Qwen/Qwen3-0.6B -v

compare:  ## Run a comparison test
	latent-reason compare "Design a REST API" --encoder Qwen/Qwen3-0.6B

models:  ## List recommended models
	latent-reason models

# ============================================================================
# Cleanup
# ============================================================================

clean:  ## Remove build artifacts and caches
ifeq ($(OS),Windows_NT)
	if exist build rmdir /s /q build
	if exist dist rmdir /s /q dist
	if exist *.egg-info rmdir /s /q *.egg-info
	if exist .pytest_cache rmdir /s /q .pytest_cache
	if exist .ruff_cache rmdir /s /q .ruff_cache
	if exist .mypy_cache rmdir /s /q .mypy_cache
	if exist htmlcov rmdir /s /q htmlcov
	for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
	del /s /q *.pyc 2>nul
else
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache/ .ruff_cache/ .mypy_cache/ htmlcov/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
endif

clean-models:  ## Remove downloaded model caches (careful!)
ifeq ($(OS),Windows_NT)
	@echo "Model cache is at: %USERPROFILE%\.cache\huggingface"
	@echo "Run manually: rmdir /s /q %USERPROFILE%\.cache\huggingface"
else
	@echo "Model cache is at: ~/.cache/huggingface"
	@echo "Run manually: rm -rf ~/.cache/huggingface"
endif

# ============================================================================
# Help
# ============================================================================

help:  ## Show this help message
	@echo.
	@echo Latent Space Reasoning Engine - Development Commands
	@echo =====================================================
	@echo.
	@echo Usage: make [target]
	@echo.
	@echo Targets:
ifeq ($(OS),Windows_NT)
	@findstr /R "^[a-zA-Z_-]*:.*##" $(MAKEFILE_LIST) | findstr /V "findstr" | for /F "tokens=1,2 delims=:#" %%a in ('more') do @echo   %%a	%%b
else
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'
endif
