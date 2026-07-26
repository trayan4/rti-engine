# 0001 — Engineering baseline

- Status: accepted
- Date: 2026-07-26

## Context

This system produces statutory documents. Non-determinism is unavoidable at the model layer, so it must be eliminated everywhere else: the same source must produce the same environment, the same behaviour, and the same checks on any machine, at any time.

## Decision

- **Python 3.13** — current stable release; no dependency in the planned stack requires an older version.
- **uv** for environment and dependency management, with `uv.lock` and `.python-version` committed. The lockfile pins exact resolved versions, so the environment is reproducible rather than approximately similar. `pyproject.toml` is the single source of truth; dependencies are added with `uv add`, never pip. `requirements.txt`, if ever needed, is a generated artifact via `uv export`.
- **ruff** for linting and formatting, configured once in `pyproject.toml` and consumed by both the editor and CI.
- **mypy in strict mode** over `src/`. Type annotations here are a reliability control, not decoration: agent state, tool inputs and tool outputs are contracts, and a violated contract should fail at check time rather than mid-run in a live request.
- **pre-commit** hooks for fast, mechanical checks (whitespace, end-of-file, YAML validity, large files, private keys, ruff). mypy is deliberately excluded — it is slow enough that developers start bypassing hooks. Fast checks at commit, thorough checks in CI.
- **Make** as the command surface. `make check` runs lint, typecheck and tests, and CI invokes the same targets, so local and CI behaviour cannot drift.

## Consequences

- Any contributor gets an identical environment from `make install`.
- Type and lint failures surface before runtime.
- Two check layers to maintain, with the split documented above so the distinction stays intentional.
- Tool versions need periodic bumping (`pre-commit autoupdate`, `uv lock --upgrade`).
