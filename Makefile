.PHONY: setup build run test clean

# Default shell
SHELL := /bin/bash

# Configuration
PYTHON := python3
PIP := pip
DOCKER_COMPOSE := docker-compose

setup:
	$(PYTHON) -m venv .venv
	source .venv/bin/activate && $(PIP) install -r requirements.txt

build:
	$(DOCKER_COMPOSE) build

run-docker:
	$(DOCKER_COMPOSE) up

run-local:
	chmod +x run_local.sh
	./run_local.sh

test:
	source .venv/bin/activate && export PYTHONPATH=$$(pwd)/backend/src && pytest tests/

kill:
	lsof -ti:8000 | xargs kill -9 2>/dev/null || true
	lsof -ti:8501 | xargs kill -9 2>/dev/null || true

clean:
	$(DOCKER_COMPOSE) down -v
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov

run-standalone:
	streamlit run streamlit_app.py
