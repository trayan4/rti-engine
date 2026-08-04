.PHONY: install lint format typecheck test check reqs data ingest graph eval-routing eval-scenarios migrate api ui

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

eval-routing:
	uv run python -m rti_engine.evals routing

eval-scenarios:
	uv run python -m rti_engine.evals scenarios

eval-quality:
	uv run python -m rti_engine.evals quality

migrate:
	uv run alembic upgrade head
	uv run python -c "import asyncio; from rti_engine.agents.checkpointing import setup_checkpointer; asyncio.run(setup_checkpointer())"

api:
	uv run uvicorn rti_engine.api.app:app --reload

ui:
	uv run streamlit run src/rti_engine/ui/app.py
