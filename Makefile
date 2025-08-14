# Makefile for Telco Customer Intelligence Platform

.PHONY: help install clean test run-pipeline train-model api dashboard docker-build docker-up format lint

# Variables
PYTHON := python
PIP := pip
PROJECT_NAME := telco-customer-intelligence
DOCKER_IMAGE := $(PROJECT_NAME):latest

# Default target
help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make clean        - Clean temporary files and caches"
	@echo "  make test         - Run tests"
	@echo "  make run-pipeline - Run data pipeline"
	@echo "  make train-model  - Train ML model"
	@echo "  make api          - Start FastAPI server"
	@echo "  make dashboard    - Start Streamlit dashboard"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-up    - Run Docker containers"
	@echo "  make format       - Format code with black"
	@echo "  make lint         - Lint code with flake8"
	@echo "  make all          - Run complete pipeline"

# Installation
install:
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	$(PYTHON) -m pip install -e .
	@echo "Creating necessary directories..."
	@mkdir -p data/{raw,processed,features}
	@mkdir -p logs
	@mkdir -p models
	@mkdir -p mlruns
	@echo "Installation complete!"

# Clean temporary files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name ".DS_Store" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	@echo "Cleaned temporary files"

# Run tests
test:
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=html

test-unit:
	$(PYTHON) -m pytest tests/unit/ -v

test-integration:
	$(PYTHON) -m pytest tests/integration/ -v

# Run data pipeline
run-pipeline:
	$(PYTHON) src/data_pipeline/pipeline.py --config configs/pipeline_config.yaml

# Train model
train-model:
	$(PYTHON) src/models/train.py --config configs/model_config.yaml

# Start API server
api:
	uvicorn src.api.main:app --reload --port 8000

# Start dashboard
dashboard:
	streamlit run src/dashboard/app.py

# Docker commands
docker-build:
	docker build -t $(DOCKER_IMAGE) .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Code quality
format:
	black src/
	black tests/
	isort src/
	isort tests/

lint:
	flake8 src/ --max-line-length=100 --ignore=E203,W503
	mypy src/ --ignore-missing-imports

# Run quality checks
quality: format lint test
	@echo "All quality checks passed!"

# Complete pipeline
all: clean install run-pipeline train-model
	@echo "Complete pipeline executed successfully!"

# Development setup
dev-setup: install
	pre-commit install
	@echo "Development environment ready!"

# Create directories
setup-dirs:
	mkdir -p data/{raw,processed,features}
	mkdir -p src/{data_pipeline,feature_engineering,models,api,monitoring,utils}
	mkdir -p notebooks
	mkdir -p tests/{unit,integration}
	mkdir -p deployment/{docker,kubernetes}
	mkdir -p docs/{api,models}
	mkdir -p configs
	mkdir -p scripts
	mkdir -p logs
	mkdir -p models
	touch data/raw/.gitkeep
	touch data/processed/.gitkeep
	touch data/features/.gitkeep

# Download data (if needed)
download-data:
	@echo "Place your Telco_Customer_Churn.csv file in data/raw/"
	@echo "You can download it from: https://www.kaggle.com/blastchar/telco-customer-churn"

# Generate documentation
docs:
	pdoc --html --output-dir docs src

# Run notebook
notebook:
	jupyter notebook notebooks/

# Check environment
check-env:
	@echo "Python version:"
	@$(PYTHON) --version
	@echo "\nInstalled packages:"
	@$(PIP) list | grep -E "pandas|scikit-learn|xgboost|fastapi|streamlit"

# Initialize database (if using)
init-db:
	$(PYTHON) scripts/init_db.py

# Run MLflow UI
mlflow-ui:
	mlflow ui --backend-store-uri file:./mlruns

# Profile code
profile:
	$(PYTHON) -m cProfile -o profile.stats src/data_pipeline/pipeline.py
	@echo "Profile saved to profile.stats"
	@echo "View with: python -m pstats profile.stats"
