# rti-engine

An agentic compliance platform for the EU Pay Transparency Directive
((EU) 2023/970) — turns a free-text employee pay-information request into
a compliant written response, with a full audit trail and a human always
in the loop for anything involving other people's pay.

## The core idea

Not every pay question carries the same risk. Asking how pay is set is
different from asking your own salary, which is different from asking
how your pay compares to colleagues'. This system treats those as three
distinct autonomy tiers, and the boundary between them is enforced in
code — not by an instruction a model could be talked out of.

- **T0 — informational.** No employee data involved. Answered
  autonomously from the regulatory and policy corpus.
- **T1 — own data.** The requester's own record only. Answered
  autonomously; the authorization layer makes nothing else reachable.
- **T2 — comparator disclosure.** Touches pay data about other people.
  Drafted, validated, and reviewed automatically — but every response
  waits for a human decision before it is ever sent. This is structural,
  not advisory: there is no code path from this tier to completion
  without a person.

A fourth path exists for input that was never a pay request at all (a
greeting, small talk) — answered immediately with no model call and no
pipeline run.

For reference, please check the "rti_engine_flow.png" file stored in the root directory.

## Design principles

- **LLMs never calculate numbers.** Every statistic in a response comes
  from deterministic Python (`statsmodels`, `PuLP`) working over the
  underlying data. Models draft prose around figures; they never produce
  the figures.
- **Every figure is checked against its source before a human ever sees
  it.** A number that doesn't trace back to the fact sheet or the legal
  position is a blocking finding, not a stylistic note.
- **Authorization lives in code, not in a prompt.** What each tier can
  reach is a fixed mapping enforced by the analytics server itself.
  Agents hold no database credentials.
- **Statistical findings account for multiple comparisons and
  distinguish "explained," "inconclusive," and "unexplained."** A modest
  gap that can't be distinguished from chance is never reported as
  established.
- **Evaluation is eval-first, and mostly deterministic.** Routing,
  figure-grounding, and trajectory checks are pure code. Only two things
  are judged by a model — groundedness and retrieval relevance — scored
  by a different vendor from the one being judged.

See `docs/adr/` for the full reasoning behind each of these, written at
the time the decision was made.

## Architecture

A LangGraph state machine orchestrates six agents (intake, analyst,
regulatory, drafter, reviewer, plus the fixed not-applicable response)
across two MCP servers — one for analytics (pay data, statistics), one
for knowledge (a hybrid vector + graph retrieval layer over the
directive, national implementation status, and company policy).

- **Models:** role-based access (`REASONING`, `CLASSIFICATION`,
  `REVIEW`), Azure OpenAI as primary, cross-vendor fallback to Anthropic
  or Groq. The reviewer deliberately runs on a different vendor from the
  drafter.
- **Persistence:** PostgreSQL for request records and LangGraph
  checkpointing (a T2 request can pause for human approval across
  process restarts). Neo4j for the citation graph between the
  directive's articles, national provisions, and policy sections —
  reached over its HTTP Cypher endpoint rather than the Bolt driver
  (see `docs/adr/0008-neo4j-http-transport.md`).
- **Retrieval:** Pinecone, with balanced per-document-kind quotas so the
  thin national-law layer isn't drowned out by the much larger directive
  text.
- **Observability:** OpenTelemetry traces to Jaeger locally and
  Application Insights when deployed; LangSmith for LLM-specific tracing.
  Every settled request is archived as a self-contained audit bundle in
  Blob Storage, independent of the database.

The detailed tech stack & their individual usage is mentioned in the
"tech-stack-diagram.png" file in the root directory.

### MCP (Model Context Protocol)

The two data-holding systems — pay analytics and the regulatory
knowledge graph — are exposed as separate MCP servers, each its own
process, each running as its own container in the deployed environment.

- **Two servers, split by data domain, not by convenience.** The
  analytics server holds employee pay data and the authorization rules
  governing it; the knowledge server holds the directive, national
  status notes, and company policy. An agent that only needs regulatory
  text is never handed a connection that could reach pay data, and vice
  versa.
- **A fixed catalog of named tools, not open queries.** Every tool an
  agent can call is a specific, narrow operation — `get_own_pay_record`,
  `describe_comparator_group`, `get_jurisdiction_status` — never a
  general "run this SQL" or "run this Cypher" tool. An agent that can
  compose its own query can read the whole graph regardless of what its
  tier permits; a prompt-injected one would. Adding a capability means
  adding a named tool and a test, which is friction by design.
- **Identity is bound server-side, not supplied by the agent.** The
  requester's employee ID and tier are injected into every tool call by
  the application layer before an agent ever sees the tool's schema —
  an agent cannot query as someone else because it never has the chance
  to supply an identity at all. This is the same enforcement boundary
  described in `docs/adr/0004-authorization-in-code.md`, applied at the
  MCP layer specifically.
- **Two transports for two environments.** Locally, both servers run
  as subprocesses reached over stdio — no ports, no networking, the
  server dies with its parent. Deployed, they run as separate container
  apps reached over HTTP. Nothing in the application code needs to know
  which environment it's in; `MCP_TRANSPORT` and the presence or
  absence of a configured URL decide it.
- **`langchain-mcp-adapters` and `FastMCP`** bridge the two sides —
  FastMCP implements each server, `langchain-mcp-adapters` lets
  LangGraph agents discover and call their tools through the standard
  MCP session lifecycle, with tracing propagated across the process
  boundary so a request's full path — API to graph to MCP server to
  database — shows up as one connected trace, not three unrelated ones.

## Running locally

```bash
make install     # uv sync, pre-commit hooks
docker compose up -d   # Postgres, Neo4j, Pinecone-compatible local index
make data         # generate the synthetic workforce dataset
make graph         # seed the knowledge graph
make api           # in one terminal
make ui            # in another — opens http://localhost:8501
```

`make check` runs lint, typecheck, and the full test suite (464 tests as
of the last count) — the same targets CI runs, so local and CI can't
drift.

## Deployment

Deploys to Azure Container Apps via Bicep (`infra/`) and a two-phase
GitHub Actions pipeline (`.github/workflows/cd.yml`): bootstrap the
registry and core infrastructure, build and push the image, deploy the
five container apps, then seed the database and graph. See the ADRs for
the reasoning behind specific infrastructure choices, several of which
were shaped by real platform constraints hit during deployment.

## Testing and evaluation

- `make check` — lint, mypy (strict), pytest
- `make eval-routing` — tier classification scored against a fixed,
  asymmetrically-weighted catalog (under-routing is disqualifying;
  over-routing is recorded)
- `make eval-scenarios` — end-to-end scenario suite with grounding checks
- `make eval-quality` — the two model-judged metrics (groundedness,
  retrieval relevance)


## Tools and technologies

### Orchestration

- **LangGraph** drives the whole request lifecycle as an explicit state
  machine — six agent nodes, conditional routing by tier, a bounded
  revision loop between drafter and reviewer, and an `interrupt()` that
  pauses a T2 request for human approval and resumes it later, in a
  different process, via PostgreSQL-backed checkpointing. Routing is a
  pure function of state with no model in the loop — the tier was
  already decided and floored in code, so re-deciding it here would
  just be a second chance to get it wrong.
- **LangChain** supplies the model client abstractions
  (`langchain-openai`, `langchain-anthropic`, `langchain-groq`) and the
  MCP tool-adapter layer (`langchain-mcp-adapters`) that lets LangGraph
  agents call tools exposed by the two MCP servers.
- **LangSmith** traces every LLM call — prompt, response, token usage,
  latency — for both local development and the deployed environment.
  Kept genuinely optional: a missing API key means tracing is off, not
  a startup failure.
- **FastMCP** implements the two MCP servers themselves (analytics,
  knowledge), each its own process, each holding only the credentials
  and tools its tier of work actually needs.

### Models, and why each one has that job

| Role | Model | Why |
|---|---|---|
| Reasoning / drafting | `gpt-5.6-terra` (Azure) | Regulatory analysis and letter drafting — the heaviest reasoning load |
| Classification | `gpt-5.6-luna` (Azure) | Cheap, fast tier classification; correctness matters more than depth here |
| Review | `claude-sonnet-5` (Anthropic, direct) | Deliberately a **different vendor** from the drafter — a model reviewing its own family's output tends to find it reasonable |
| Fallback | `llama-3.3-70b-versatile` (Groq) | Cross-vendor fallback reached over entirely separate infrastructure, so an Azure outage doesn't take down the fallback too |
| Embeddings | `text-embedding-3-small` (Azure) | Pinecone retrieval |

Models are requested **by role**, never by name — one module decides
what `REASONING`/`CLASSIFICATION`/`REVIEW` resolves to, with what
fallback chain, at what timeout. Temperature is pinned to zero wherever
the provider accepts it, since this system produces statutory
documents and identical input should yield identical output.

### Prompting

Every prompt is a versioned `Prompt` object (`agents/prompts.py`) —
name, version, description, declared inputs, and the template itself —
so a prompt identifier (`intake_classification@v1`,
`compliance_review@v1`) is recorded in the audit trail for every
decision a model made. Changing a prompt's wording without bumping its
version is a lint-checkable mistake, not a silent drift.

Strategies used deliberately, by task:

- **Structured output**, not free text, for every model call that
  produces something downstream code consumes — a Pydantic schema is
  passed alongside the prompt, so a classification or a review verdict
  is a typed object, not a string to parse.
- **Escalate-when-uncertain framing** for tier classification: the
  prompt explicitly tells the model that treating an ambiguous request
  as broader-than-needed costs a human a few minutes, while treating it
  as narrower-than-needed can release data without review — asymmetric
  stakes stated directly in the prompt, then enforced again afterward
  in code regardless of what the model actually returned.
- **Revision-feedback loops**: a rejected draft goes back to the
  drafter with the reviewer's specific findings appended to the prompt,
  not just a "try again" — bounded to a fixed number of revisions, after
  which the draft goes to a human with the findings attached rather
  than looping forever.
- **No free-text tool queries**: agents select from a fixed catalog of
  named query templates (both for the graph and the analytics server)
  rather than being able to write their own Cypher or SQL — a prompt
  cannot expand what a model is allowed to ask for.

### Responsible AI

- **PII redaction before anything reaches a model** — Presidio + spaCy
  strip names from a request before classification or drafting ever
  sees it, so a request naming a colleague is still evaluated as a
  comparator request without the name ever entering a prompt or a
  stored transcript.
- **Every figure in a response is validated against its source**
  before a human sees it — a number that doesn't trace back to the fact
  sheet or the legal position is a blocking finding, not a style note.
- **Authorization is a function, not an instruction** — what each tier
  can access is enforced by the analytics server itself, server-side;
  agents hold no database credentials, so a prompt-injected agent has
  nothing to compromise with.
- **Statistical safeguards on findings** — Benjamini-Hochberg correction
  across all tested groups, and three verdicts (explained / inconclusive
  / unexplained) rather than two, so a gap that can't be distinguished
  from chance is never reported as established.
- **Cross-vendor review** — the reviewer runs on a different model
  family from the drafter specifically to avoid shared blind spots.
- **Structural human-in-the-loop** — a T2 (comparator disclosure)
  request has no code path to completion without a person deciding.
- **Full audit archival** — every settled request's complete trail
  (every classification, revision, review finding, and human decision)
  is written as a self-contained JSON bundle to Blob Storage,
  independent of the database.

See `docs/adr/` for the full reasoning behind each of these,
`docs/threat-model.md` for what this system defends against end to end,
and `docs/patterns.md` for the recurring implementation shapes these
principles produced across the codebase.

### Data and knowledge

- **PostgreSQL** — request records and LangGraph's checkpoint store.
- **Neo4j** — the citation graph linking directive articles, national
  legal provisions, and company policy sections. Reached over its HTTP
  Cypher endpoint rather than the Bolt driver — a deliberate choice
  forced by a platform-level limitation in this deployment environment
  (see `docs/adr/0008-neo4j-http-transport.md`).
- **Pinecone** — vector retrieval over the same three-layer corpus,
  with per-document-kind quotas so the much larger directive text
  doesn't drown out the thin national-law layer.
- **statsmodels / PuLP** — every statistic and every remediation cost in
  the system comes from deterministic Python here, never from a model.

### Azure infrastructure

Deployed via Bicep (`infra/`), one module per resource:

- **Container Apps** — five apps from one image (API, UI, two MCP
  servers, Neo4j), differing only in the command each runs.
- **Container Apps Environment** — the shared environment all five run
  in. Internal-only ingress for the two MCP servers; external ingress
  for the API, UI, and Neo4j — Neo4j's is external only because a
  platform-level limitation ruled out internal TCP ingress for this
  workload (see `docs/adr/0008-neo4j-http-transport.md`); real access
  still requires the database credential.
- **Container Registry** — holds the built image; pulled via managed
  identity, no admin credentials.
- **PostgreSQL Flexible Server**.
- **Key Vault** — every secret (API keys, database password, Neo4j
  auth) is stored here and referenced by the container apps via managed
  identity; nothing is passed as a plain environment variable.
- **Storage Account** — audit bundle archival.
- **Log Analytics + Application Insights** — OpenTelemetry traces land
  here in the deployed environment (Jaeger locally).

### Engineering baseline

- **uv** for environment and dependency management — `uv.lock` pins
  exact resolved versions, so the environment is reproducible rather
  than approximately similar.
- **ruff** for linting and formatting, one config, used identically by
  the editor and CI.
- **mypy in strict mode** over `src/` — agent state and tool
  inputs/outputs are treated as contracts that should fail at check
  time, not mid-request.
- **pre-commit** for fast mechanical checks; mypy deliberately excluded
  from the hook (too slow, encourages bypassing) and run in CI instead.
- **Make** as the single command surface — `make check` runs the same
  lint/typecheck/test targets locally and in CI, so the two can't drift.
- **Docker** — a multi-stage build producing one image for all five
  deployed apps, plus `docker-compose.yml` for local Postgres, Neo4j,
  and a Pinecone-compatible local index.
- **GitHub Actions** — CI on every push (lint, typecheck, test, tier
  routing eval); a two-phase CD pipeline (bootstrap infrastructure,
  build and push, deploy apps, seed database and graph) on merge to
  main.

## Status

Deployed and working end to end across all three tiers, with human
approval, audit archival, and tracing confirmed in the live environment.
Portfolio project — not in production use.
