.PHONY: install lint format typecheck test check reqs data ingest graph

install:
	uv sync
	uv run python -m spacy download en_core_web_lg

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

graph:
	uv run python -m rti_engine.knowledge.graph_ingest
