# FloodRisk Development Makefile

# Variables
PYTHON := python
PIP := pip
DOCKER := docker
DOCKER_COMPOSE := docker-compose
PROJECT_NAME := floodrisk

# Colors for output
BLUE := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m # No Color

.PHONY: help install dev-install test test-cov lint format type-check security-check clean build dev-up dev-down logs db-migrate db-reset backup restore docs docs-serve jupyter notebook pre-commit-install pre-commit-run ci-test release version-patch version-minor version-major

# Default target
help: ## Show this help message
	@echo "$(BLUE)FloodRisk Development Commands$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Installation and Setup
install: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

dev-install: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	$(MAKE) pre-commit-install

# Testing
test: ## Run tests
	@echo "$(BLUE)Running tests...$(NC)"
	pytest

test-cov: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	pytest --cov=src --cov-report=html --cov-report=term --cov-report=xml

test-fast: ## Run fast tests (exclude slow markers)
	@echo "$(BLUE)Running fast tests...$(NC)"
	pytest -m "not slow"

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	pytest tests/integration/

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	pytest tests/unit/

# Code Quality
lint: ## Run linting
	@echo "$(BLUE)Running linting...$(NC)"
	flake8 src tests
	isort --check-only src tests
	black --check src tests

format: ## Format code
	@echo "$(BLUE)Formatting code...$(NC)"
	isort src tests
	black src tests

type-check: ## Run type checking
	@echo "$(BLUE)Running type checking...$(NC)"
	mypy src

security-check: ## Run security checks
	@echo "$(BLUE)Running security checks...$(NC)"
	bandit -r src
	safety check

# Docker Development
build: ## Build Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	$(DOCKER_COMPOSE) build

dev-up: ## Start development environment
	@echo "$(BLUE)Starting development environment...$(NC)"
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)Development environment started!$(NC)"
	@echo "API: http://localhost:8000"
	@echo "API Docs: http://localhost:8000/docs"
	@echo "Grafana: http://localhost:3000 (admin/admin123)"
	@echo "Jupyter: http://localhost:8888"

dev-down: ## Stop development environment
	@echo "$(BLUE)Stopping development environment...$(NC)"
	$(DOCKER_COMPOSE) down

dev-restart: ## Restart development environment
	@echo "$(BLUE)Restarting development environment...$(NC)"
	$(DOCKER_COMPOSE) restart

logs: ## View logs
	$(DOCKER_COMPOSE) logs -f

logs-app: ## View app logs only
	$(DOCKER_COMPOSE) logs -f app

logs-db: ## View database logs only
	$(DOCKER_COMPOSE) logs -f postgres

# Database Management
db-shell: ## Access database shell
	$(DOCKER_COMPOSE) exec postgres psql -U flooduser -d floodrisk

db-migrate: ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(NC)"
	$(DOCKER_COMPOSE) exec app alembic upgrade head

db-migration: ## Create new migration
	@echo "$(BLUE)Creating new migration...$(NC)"
	@read -p "Migration message: " msg; \
	$(DOCKER_COMPOSE) exec app alembic revision --autogenerate -m "$$msg"

db-reset: ## Reset database (WARNING: destroys all data)
	@echo "$(RED)WARNING: This will destroy all data!$(NC)"
	@read -p "Are you sure? (y/N): " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		$(DOCKER_COMPOSE) down -v; \
		$(DOCKER_COMPOSE) up -d postgres redis; \
		sleep 5; \
		$(MAKE) db-migrate; \
	fi

db-backup: ## Create database backup
	@echo "$(BLUE)Creating database backup...$(NC)"
	mkdir -p backups
	$(DOCKER_COMPOSE) exec postgres pg_dump -U flooduser floodrisk > backups/backup_$(shell date +%Y%m%d_%H%M%S).sql

db-restore: ## Restore database from backup
	@echo "$(BLUE)Restoring database...$(NC)"
	@read -p "Backup file path: " backup_file; \
	$(DOCKER_COMPOSE) exec -T postgres psql -U flooduser floodrisk < $$backup_file

# Development Tools
jupyter: ## Start Jupyter notebook server
	@echo "$(BLUE)Starting Jupyter notebook...$(NC)"
	$(DOCKER_COMPOSE) exec app jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

notebook: jupyter ## Alias for jupyter

shell: ## Access application shell
	$(DOCKER_COMPOSE) exec app bash

python-shell: ## Access Python shell with app context
	$(DOCKER_COMPOSE) exec app python

redis-cli: ## Access Redis CLI
	$(DOCKER_COMPOSE) exec redis redis-cli

# Pre-commit and CI
pre-commit-install: ## Install pre-commit hooks
	@echo "$(BLUE)Installing pre-commit hooks...$(NC)"
	pre-commit install

pre-commit-run: ## Run pre-commit on all files
	@echo "$(BLUE)Running pre-commit checks...$(NC)"
	pre-commit run --all-files

ci-test: ## Run CI test suite
	@echo "$(BLUE)Running CI test suite...$(NC)"
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) security-check
	$(MAKE) test-cov

# Documentation
docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(NC)"
	cd docs && make html

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation...$(NC)"
	cd docs/_build/html && python -m http.server 8080

docs-clean: ## Clean documentation build
	cd docs && make clean

# Monitoring and Metrics
prometheus: ## Open Prometheus UI
	@echo "Opening Prometheus at http://localhost:9090"
	open http://localhost:9090

grafana: ## Open Grafana dashboard
	@echo "Opening Grafana at http://localhost:3000 (admin/admin123)"
	open http://localhost:3000

metrics: ## Show application metrics
	curl -s http://localhost:8000/metrics | grep -v "^#"

health: ## Check application health
	@echo "$(BLUE)Checking application health...$(NC)"
	curl -s http://localhost:8000/health | jq .

# Cleanup
clean: ## Clean up temporary files and caches
	@echo "$(BLUE)Cleaning up...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true

clean-docker: ## Remove all Docker containers, images and volumes
	@echo "$(RED)WARNING: This will remove all Docker containers, images and volumes!$(NC)"
	@read -p "Are you sure? (y/N): " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		$(DOCKER_COMPOSE) down -v --remove-orphans; \
		$(DOCKER) system prune -af --volumes; \
	fi

# Deployment and Release
version-patch: ## Bump patch version
	@echo "$(BLUE)Bumping patch version...$(NC)"
	bump2version patch

version-minor: ## Bump minor version
	@echo "$(BLUE)Bumping minor version...$(NC)"
	bump2version minor

version-major: ## Bump major version
	@echo "$(BLUE)Bumping major version...$(NC)"
	bump2version major

release: ## Create release (run tests, build, tag)
	@echo "$(BLUE)Creating release...$(NC)"
	$(MAKE) ci-test
	$(MAKE) build
	@echo "$(GREEN)Release ready!$(NC)"

# Production
prod-build: ## Build production Docker image
	@echo "$(BLUE)Building production image...$(NC)"
	$(DOCKER) build --target production -t $(PROJECT_NAME):prod .

prod-up: ## Start production environment
	@echo "$(BLUE)Starting production environment...$(NC)"
	$(DOCKER_COMPOSE) -f docker-compose.prod.yml up -d

prod-down: ## Stop production environment
	$(DOCKER_COMPOSE) -f docker-compose.prod.yml down

# Data and Model Management
download-data: ## Download sample datasets
	@echo "$(BLUE)Downloading sample datasets...$(NC)"
	mkdir -p data/raw
	# Add dataset download commands here

train-model: ## Train the ML model
	@echo "$(BLUE)Training model...$(NC)"
	$(DOCKER_COMPOSE) exec app python -m src.training.train

evaluate-model: ## Evaluate trained model
	@echo "$(BLUE)Evaluating model...$(NC)"
	$(DOCKER_COMPOSE) exec app python -m src.evaluation.evaluate

# Utilities
install-system-deps: ## Install system dependencies (Ubuntu/Debian)
	@echo "$(BLUE)Installing system dependencies...$(NC)"
	sudo apt-get update
	sudo apt-get install -y gdal-bin libgdal-dev libspatialindex-dev libproj-dev libgeos-dev

check-env: ## Check environment configuration
	@echo "$(BLUE)Environment Configuration:$(NC)"
	@echo "Python: $$(python --version)"
	@echo "Docker: $$(docker --version)"
	@echo "Docker Compose: $$(docker-compose --version)"
	@echo "Git: $$(git --version)"

info: ## Show project information
	@echo "$(BLUE)FloodRisk Project Information:$(NC)"
	@echo "Project: $(PROJECT_NAME)"
	@echo "Directory: $$(pwd)"
	@echo "Git Branch: $$(git branch --show-current 2>/dev/null || echo 'No git repo')"
	@echo "Docker Status: $$(docker-compose ps 2>/dev/null | wc -l) containers"