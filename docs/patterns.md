# Engineering patterns

Recurring shapes in this codebase, named once here rather than
re-explained at every occurrence. Where a pattern exists because of a
specific decision, the relevant ADR is linked — this document describes
the *shape*, the ADR explains the *reasoning*.

## Advisory model, deterministic floor

A model classifies or judges something; code decides what actually
happens, and the model's answer only ever narrows what's possible, never
widens it.

- Intake's tier classification is advisory; `_apply_floor` forces T2 for
  anything touching comparator data or marked ambiguous, regardless of
  what the model returned (`docs/adr/0004-authorization-in-code.md`).
- The reviewer's approval is advisory for T2; a `revising` draft still
  goes back for another pass, and an approved draft still waits for a
  human — the model's "approved" never itself completes a request.

The tell: a model call's output feeds into a separate, unconditional
check afterward, not directly into a decision.

## Fixed catalog instead of free-form generation

Anywhere a model or an agent needs to reach data or perform an
operation, the set of things it can ask for is a closed, named list —
never a query it composes itself.

- `graph_queries.py`'s six named Cypher templates.
- The MCP analytics server's named tools (`get_own_pay_record`,
  `describe_comparator_group`, ...) — no general query tool exists.
- Intake's `RequestCategory` is a `Literal` of five fixed values, not a
  free-text field.

The tell: a `Literal[...]` type, a `TEMPLATES_BY_NAME` dict, or a
docstring saying "adding a capability means adding a [template/tool]
and a test."

## Identity supplied by the caller, never by the model

Anywhere an operation needs to know *who* is asking, that identity is
bound by application code before an agent's context exists — an agent
is never in a position to state or restate it.

- `bind_principal` in `mcp/client.py` strips any identity fields a tool
  schema exposes and replaces them with the authenticated caller's
  values before the agent ever sees the schema.
- `_apply_floor` and the analytics server's authorization module both
  receive identity as a plain function argument from the API layer, not
  from parsed model output.

The tell: a function that strips or overwrites specific field names
before invoking something, rather than trusting what it was given.

## Guarded edges: every path can degrade

Every conditional edge in the request graph checks for failure or
budget exhaustion first, before its normal routing logic runs — a
request can leave any node into the degraded response, not just the
ones that obviously might fail.

- `_guarded(next_node)` in `agents/graph.py` wraps a plain routing
  function with an unconditional `if state.get("errors"): return
  DEGRADED` check first.
- `route_after_validation`, `route_after_review`, and
  `route_after_approval` all repeat this same check independently,
  rather than relying on one shared gate — a request cannot silently
  fall through a node that forgot to add it.

The tell: near-identical `if state.get("errors"): return DEGRADED` at
the top of several otherwise-different functions. The repetition is
deliberate, not an oversight to consolidate.

## Configuration absence is a valid state, not an error

Anything that depends on optional external infrastructure (tracing,
archival) checks for its own configuration and quietly does nothing if
absent — it never raises just because a feature wasn't set up.

- `archive_bundle` returns `None` immediately if `archive_account_name`
  isn't set — normal locally, not an error.
- `enable_tracing` exports LangSmith config only if a key is present;
  its absence means tracing is off, not a startup failure.

The tell: an early `if not settings.X: return None` guard, paired with
a docstring explicitly stating that the absence is expected, not
exceptional.

## Failure is logged, never silently swallowed — and this was violated once

The intended pattern: an operation that can fail without blocking the
user-facing outcome (archival, tracing) catches its exception, logs it,
and returns a null result — the caller proceeds regardless, but the
failure is never invisible.

This was violated in `archive_settled_request`'s first version — an
`except ArchiveError: return None` with no logging at all, despite the
function's own docstring claiming otherwise. It went undetected because
every symptom of it looked like success: no error, no crash, just a
number that never appeared. Fixed by adding the logging the docstring
already promised. Worth remembering as the concrete cost of this pattern
being followed in word but not in code (`docs/threat-model.md`,
"infrastructure failure" section).

The tell now, post-fix: every `except` block in an optional/best-effort
path has a paired `logger.exception` or `logger.info`, not just a
`return None`.

## Roles, not names

Anywhere this system depends on an external, swappable resource (a
model, a transport), code asks for a role or a capability, and a single
module resolves that to a concrete choice — nothing downstream names a
vendor or a specific endpoint.

- `get_structured_model(ModelRole.CLASSIFICATION, ...)` — no agent
  names `gpt-5.6-luna` directly (`docs/adr/0005-model-access-layer.md`).
- `server_connections()` decides stdio vs. HTTP transport from whether
  a URL is configured; nothing else in the codebase checks which
  environment it's running in.

The tell: an enum or role name passed to a factory function, with the
actual resolution logic confined to one file.

## State updates as small, named, auditable dicts

Every graph node returns a plain dict of the state fields it's
changing, built with the `audited(...)` helper so every node's effect
on state produces a matching entry in the request's audit trail by
construction — not as a separate step someone could forget.

- Every `*_node` function in `agents/graph.py` ends with `return {...,
  **audited(Actor.X, "action_name", ...)}`.

The tell: a node function that returns state changes without a call to
`audited(...)` is the exception, not the norm, and should be treated as
a gap.

## Structured output over parsed prose

Any model call whose result feeds downstream code returns a typed
Pydantic object, never a string the caller then parses.

- `IntakeClassification`, `ReviewResult`, `DraftLetter` — all
  `BaseModel` subclasses with `extra="forbid"`, passed as the expected
  schema alongside the prompt.

The tell: `get_structured_model(role, SomeSchema)` rather than a plain
chat call whose `.content` gets regex'd or JSON-parsed by hand.

## Versioned, identified prompts

Every prompt is a `Prompt` object with a name and version, not an
inline string — so the audit trail can record exactly which prompt
version produced a given decision, and changing wording without
bumping the version is a mistake the linter or a reviewer can catch.

- `INTAKE_PROMPT = Prompt(name="intake_classification", version=1, ...)`
- Every audit entry recording a model decision includes
  `prompt=result.prompt_identifier`.

The tell: a prompt string that lives as a bare Python literal inside a
function, rather than a named `Prompt` object at module scope.
