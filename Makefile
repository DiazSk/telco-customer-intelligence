# Telco Customer Intelligence Platform - Makefile
# Development and deployment automation

.PHONY: help install dev-install clean test lint format build up down logs restart health deploy-prod backup

# Default target
help:
	@echo "Telco Customer Intelligence Platform - Available commands:"
	@echo ""
	@echo "Development:"
	@echo "  install          Install production dependencies"
	@echo "  dev-install      Install development dependencies"
	@echo "  clean           Clean up Python cache and temp files"
	@echo "  test            Run all tests"
	@echo "  lint            Run linting checks"
	@echo "  format          Format code with black and isort"
	@echo ""
	@echo "Docker:"
	@echo "  build           Build Docker images"
	@echo "  up              Start development environment"
	@echo "  down            Stop development environment"
	@echo "  logs            Show container logs"
	@echo "  restart         Restart all containers"
	@echo "  health          Check container health"
	@echo "  test-docker     Test Docker containers"
	@echo ""
	@echo "Production:"
	@echo "  deploy-prod     Deploy to production"
	@echo "  backup          Backup production data"
	@echo ""

# Development setup
install:
	pip install -r requirements.txt

dev-install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/

# Testing
test:
	python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

test-unit:
	python -m pytest tests/unit/ -v

test-integration:
	python -m pytest tests/integration/ -v

test-performance:
	python -m pytest tests/performance/ -v

# Code quality
lint:
	flake8 src/ --max-line-length=88 --extend-ignore=E203,W503
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/
	isort src/ tests/

format-check:
	black --check src/ tests/
	isort --check-only src/ tests/

# Docker development
build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

restart:
	docker compose restart

health:
	docker compose ps
	@echo "\nChecking service health..."
	@curl -f http://localhost:8000/health || echo "API not ready"
	@curl -f http://localhost:8501/_stcore/health || echo "Dashboard not ready"
	@curl -f http://localhost:5000/health || echo "MLflow not ready"

test-docker:
	python scripts/test_docker.py

# Production deployment
deploy-prod:
	@echo "Deploying to production..."
	docker compose -f docker-compose.prod.yml build
	docker compose -f docker-compose.prod.yml up -d
	@echo "Production deployment complete!"

backup:
	@echo "Creating backup..."
	docker compose exec postgres pg_dump -U telco_user telco_db > backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "Backup complete!"

# Database operations
db-shell:
	docker compose exec postgres psql -U telco_user -d telco_db

db-reset:
	docker compose down -v
	docker compose up -d postgres
	sleep 10
	docker compose up -d

# Monitoring
monitor:
	@echo "Opening monitoring dashboards..."
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana: http://localhost:3000"
	@echo "API Docs: http://localhost:8000/docs"
	@echo "Dashboard: http://localhost:8501"
	@echo "MLflow: http://localhost:5000"

# CI/CD helpers
ci-test:
	python -m pytest tests/ -v --cov=src --cov-report=xml

ci-build:
	docker build -t telco-api .
	docker build -f Dockerfile.dashboard -t telco-dashboard .

ci-security:
	bandit -r src/ -f json -o bandit-report.json
	safety check --json --output safety-report.json

# Development server
run-api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-dashboard:
	streamlit run src/dashboard/app.py --server.port 8501

run-pipeline:
	python src/data_pipeline/pipeline.py

run-notebook:
	jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser