# VLSP2025 Legal Q&A System Makefile
# This Makefile provides convenient commands for common project tasks

.PHONY: help install setup venv venv-create venv-activate venv-remove lawdb task1 task2 clean docker-up docker-down test baseline neuro-symbolic baseline-compare baseline-vlm-only test-four-pipelines format lint ui-freeform-demo ui-facts

# Default target
help:
	@echo "VLSP2025 Legal Q&A System - Available Commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  venv-create  Create virtual environment using uv"
	@echo "  venv-activate Show activation command for virtual environment"
	@echo "  venv-remove  Remove virtual environment"
	@echo "  install      Install project dependencies"
	@echo "  setup        Complete project setup (venv + install + docker)"
	@echo ""
	@echo "Docker Services:"
	@echo "  docker-up   Start Qdrant vector database"
	@echo "  docker-down Stop Qdrant vector database"
	@echo ""
	@echo "Data Processing:"
	@echo "  lawdb       Process legal database (preprocess, extract, parse, ingest)"
	@echo "  task1       Run Task 1 (Article Retrieval) on training data"
	@echo "  task2       Run Task 2 (Answer Generation) on training data"
	@echo "  task1-test  Run Task 1 on public test data"
	@echo "  task2-test  Run Task 2 on public test data"
	@echo "  task1-private Run Task 1 on private test data"
	@echo "  task2-private Run Task 2 on private test data"
	@echo ""
	@echo "Development:"
	@echo "  test        Run the complete unit test suite"
	@echo "  baseline    Run the locked 20-sample baseline evaluation"
	@echo "  test-four-pipelines Capture answers and latency for the shared 20-image set"
	@echo "  format      Format code with black"
	@echo "  clean       Clean temporary files and caches"
	@echo ""
	@echo "UI Applications:"
	@echo "  ui-label    Start labeling application"
	@echo "  ui-inspect  Start law database inspection tool"
	@echo "  ui-freeform-demo Start free-form cited QA demo"
	@echo "  ui-facts    Upload an image and inspect extracted traffic-sign facts"
	@echo ""

# Virtual environment management
venv-create:
	@echo "Creating virtual environment using uv..."
	uv venv
	@echo "Virtual environment created successfully!"
	@echo "To activate it, run: source .venv/bin/activate"

venv-activate:
	@echo "To activate the virtual environment, run:"
	@echo "source .venv/bin/activate"

venv-remove:
	@echo "Removing virtual environment..."
	rm -rf .venv
	@echo "Virtual environment removed!"

# Installation and setup
install:
	uv pip install -e ".[dev]"

setup: venv-create install docker-up
	@echo "Project setup complete!"
	@echo "Virtual environment created and dependencies installed."
	@echo "Don't forget to create a .env file with your API keys"
	@echo "To activate the virtual environment, run: source .venv/bin/activate"

# Docker services
docker-up:
	docker-compose up -d
	@echo "Qdrant vector database started on http://localhost:6333"

docker-down:
	docker-compose down
	@echo "Qdrant vector database stopped"

# Legal database processing pipeline
lawdb:
	@echo "Starting legal database processing pipeline..."
	python -m src.core.lawdb.preprocess_lawdb_infos
	python -m src.core.lawdb.extract_lawdb_sign_images
	python -m src.core.lawdb.parse_lawdb_sign_infos
	python -m src.core.lawdb.ingest_lawdb_signs
	@echo "Legal database processing complete!"

# Task 1: Article Retrieval
task1-train:
	@echo "Running Task 1 (Article Retrieval) on training data..."
	python -m src.core.sub_task_1 --set_name train

task1-public-test:
	@echo "Running Task 1 (Article Retrieval) on public test data..."
	python -m src.core.sub_task_1 --set_name public_test

task1-private-test:
	@echo "Running Task 1 (Article Retrieval) on private test data..."
	python -m src.core.sub_task_1 --set_name private_test

# Task 2: Answer Generation
task2-train:
	@echo "Running Task 2 (Answer Generation) on training data..."
	python -m src.core.sub_task_2 --set_name train

task2-public-test:
	@echo "Running Task 2 (Answer Generation) on public test data..."
	python -m src.core.sub_task_2 --set_name public_test

task2-private-test:
	@echo "Running Task 2 (Answer Generation) on private test data..."
	python -m src.core.sub_task_2 --set_name private_test

# Complete pipeline
pipeline-train: lawdb task1-train task2-train
	@echo "Complete pipeline (lawdb + task1 + task2) on train set finished!"

pipeline-public-test: lawdb task1-public-test task2-public-test
	@echo "Complete pipeline on public test data finished!"

pipeline-private-test: lawdb task1-private-test task2-private-test
	@echo "Complete pipeline on private test data finished!"

test:
	pytest -q

baseline:
	python -m src.eval.freeform_qa_metrics --config configs/baseline_eval.json

neuro-symbolic:
	python -m src.eval.freeform_qa_metrics --config configs/neuro_symbolic_eval.json

baseline-compare:
	@test -n "$(CANDIDATE)" || (echo "Usage: make baseline-compare CANDIDATE=outputs/candidate_metrics.json" && exit 2)
	python -m src.eval.compare_baseline --candidate "$(CANDIDATE)"


lint:
	@echo "Running linting checks..."
	flake8 src/ tests/
	mypy src/

clean:
	@echo "Cleaning temporary files and caches..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf build/
	rm -rf dist/

# UI Applications
ui-label:
	@echo "Starting labeling application..."
	streamlit run src/ui/label_app.py

ui-inspect:
	@echo "Starting law database inspection tool..."
	PYTHONPATH=. streamlit run src/ui/inspect_lawdb.py

ui-inspect-task1:
	@echo "Starting Task 1 inspection tool..."
	PYTHONPATH=. streamlit run src/ui/inspect_subtask1.py

ui-freeform-demo:
	@echo "Starting free-form cited QA demo..."
	PYTHONPATH=. streamlit run src/ui/freeform_qa_demo.py

ui-facts:
	@echo "Starting traffic-sign facts viewer..."
	PYTHONPATH=. streamlit run src/ui/facts_viewer.py

ui-sign-template-review:
	@echo "Starting traffic-sign fact-template review..."
	PYTHONPATH=. streamlit run src/ui/sign_template_review.py --server.port 8502

# Data management
data-clean:
	@echo "Cleaning processed data files..."
	rm -f data/train_data/*_preprocessed*.json
	rm -f data/public_test/*_preprocessed*.json
	rm -f data/private_test/*_preprocessed*.json
	rm -f data/law_db/*_preprocessed*.json

# Quick development workflow
dev: clean venv-create install docker-up
	@echo "Development environment ready!"
	@echo "To activate the virtual environment, run: source .venv/bin/activate"

# Production deployment preparation
build: clean install
	@echo "Project built for production!"

# Show project status
status:
	@echo "Project Status:"
	@echo "==============="
	@echo "Python version: $(shell python --version)"
	@echo "Docker status: $(shell docker-compose ps --services --filter status=running)"
	@echo "Qdrant status: $(shell curl -s http://localhost:6333/collections 2>/dev/null && echo 'Running' || echo 'Not running')"
	@echo "Environment file: $(shell test -f .env && echo 'Present' || echo 'Missing')"

baseline-vlm-only:
	python -m src.eval.vlm_only --config configs/baseline2_vlm_only.json

test-four-pipelines:
	python -m src.eval.four_pipeline_test
