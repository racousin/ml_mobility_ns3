.PHONY: help install test format lint clean train generate

help:
	@echo "Available commands:"
	@echo "  install    Install dependencies with poetry"
	@echo "  test       Run unit tests"
	@echo "  format     Format code with black and isort"
	@echo "  lint       Run linting with flake8"
	@echo "  clean      Clean up generated files"
	@echo "  train      Train model with example data"
	@echo "  generate   Generate trajectories from trained model"

install:
	poetry install

test:
	poetry run pytest tests/

format:
	poetry run black ml_mobility_ns3/ tests/
	poetry run isort ml_mobility_ns3/ tests/

lint:
	poetry run flake8 ml_mobility_ns3/ tests/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf output/ generated_trajectories.*

train:
	poetry run python example.py

generate:
	@if [ -d "output" ]; then \
		poetry run python generate.py --model-dir output --n-samples 10 --plot; \
	else \
		echo "No trained model found. Run 'make train' first."; \
	fi