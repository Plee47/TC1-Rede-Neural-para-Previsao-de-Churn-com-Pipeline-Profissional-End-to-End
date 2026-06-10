.PHONY: install lint format test test-all api mlflow docker-build docker-up

install:
	pip install -e ".[all]"

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

test:
	pytest tests/ -m "not slow" --cov=src --cov-fail-under=85

test-all:
	pytest tests/ --cov=src

api:
	uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

mlflow:
	mlflow ui --backend-store-uri mlruns

docker-build:
	docker build -t tc1-churn-api:latest .

docker-up:
	docker compose up --build
