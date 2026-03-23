.PHONY: install dev test lint format run clean docker

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

lint:
	ruff check src/ tests/
	mypy src/

format:
	ruff format src/ tests/

run:
	streamlit run src/viz/app.py

cli-price:
	python -m src.cli price --spot 100 --strike 105 --expiry 0.5 --rate 0.05 --vol 0.2 --type call

cli-greeks:
	python -m src.cli greeks --spot 100 --strike 105 --expiry 0.5 --rate 0.05 --vol 0.2 --type call

docker:
	docker build -t options-pricing-trainer .
	docker run -p 8501:8501 options-pricing-trainer

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache htmlcov .ruff_cache
	find . -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
