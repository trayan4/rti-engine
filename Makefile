.PHONY: install lint format typecheck test check reqs data ingest

install:
	uv sync

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy src

test:
	uv run pytest

check: lint typecheck test

reqs:
	uv export --format requirements-txt > requirements.txt

data:
	uv run python -m rti_engine.analytics.generate

ingest:
	uv run python -m rti_engine.knowledge.vectorstore
