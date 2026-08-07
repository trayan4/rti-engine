# Threat model

This system handles two categories of sensitive material: individual pay
data, and statutory disclosure obligations with legal weight. This
document states what it defends against, and how — deliberately at the
level of mechanism, not narrative, since the value of a threat model is
in what it lets a reader verify.

## In scope

- An employee submitting a request through the UI or API, in good or bad
  faith.
- A request crafted to manipulate a model's behavior (prompt injection),
  whether embedded in the request text itself or, in principle, in any
  retrieved document.
- A model behaving unreliably on its own — hallucinating a figure,
  misclassifying a request, drifting mid-response — with no adversarial
  intent involved at all.
- Failures in the surrounding infrastructure (a database outage, a
  timeout, a partial deployment) that could cause an incorrect or
  incomplete response to reach an employee.

## Out of scope

- Compromise of the underlying Azure platform, the model providers'
  infrastructure, or the developer's own machine and credentials.
- Denial of service via request volume — no rate limiting is implemented;
  this is a portfolio deployment, not a production service sized for
  adversarial load.
- Social engineering of a human reviewer. The system assumes a reviewer
  acts in good faith once a request reaches them; nothing here defends
  against a reviewer who chooses to approve something they shouldn't.

## Threats and mitigations

### An agent is manipulated into disclosing data outside its tier

**Threat.** A request — through its own text, or in principle through a
retrieved document — persuades a model to attempt an action beyond what
its assigned tier permits: querying another employee's record, treating
a comparator request as an own-data one, or supplying a fabricated
identity to a tool call.

**Mitigation.** Authorization is enforced in code, not in a prompt
(`docs/adr/0004-authorization-in-code.md`). The requester's identity and
tier are bound to every tool call by the application layer before an
agent's context is ever constructed — an agent is never given the
opportunity to supply an identity, so there is no parameter for an
injected instruction to override. Permitted scopes per tier are a fixed
mapping in one module, enforced server-side, in the MCP analytics server
itself. A prompt can ask a model to attempt anything; what actually
executes is decided somewhere the prompt has no reach into.

This is checked by unit tests asserting each tier's actual reachable
scope, independent of any model in the loop, and by an adversarial test
suite (`tests/test_attacks.py`) exercising injection attempts directly.

### A request is misclassified into a lower-risk tier than it warrants

**Threat.** Intake's classification is itself a model call, and a model
call can be wrong — through injection, through ambiguity, or through
ordinary model error — potentially routing a comparator-disclosure
request down to the autonomous own-data path.

**Mitigation.** The model's classification is advisory; a deterministic
floor is applied afterward in code (`agents/intake.py`,
`_apply_floor`). A request touching comparator data, or one the model
itself flags as ambiguous, is forced to T2 regardless of what category
the model returned. The floor cannot be weakened by anything in the
model's output, because it runs after that output and does not consult
it for permission.

Routing correctness is measured asymmetrically, not as a single accuracy
figure: under-routing (treating a request as lower-risk than it is) is
disqualifying; over-routing (treating it as higher-risk) is recorded but
tolerated, since the two errors have different real consequences
(`docs/adr/0007-measurement-boundary.md`).

### A response contains a fabricated or incorrect figure

**Threat.** A model asked to draft a letter referencing pay figures or
statistical findings could state a number that doesn't match its actual
source — through hallucination, through misreading the fact sheet, or
through carrying an earlier draft's figure into a revision where it no
longer applies.

**Mitigation.** No figure in any response is trusted from generation.
Every number in a drafted letter is parsed out and checked against the
fact sheet and legal position it should have come from
(`guardrails/numbers.py`) before a human ever sees the letter. An
ungrounded figure is a blocking finding, not a stylistic note, and — for
autonomous tiers — a request whose figures don't validate does not
complete; it degrades to a response saying a person will follow up. This
check is deterministic Python, not a model judging another model's
output.

### A genuine but statistically weak pay gap is reported as an established finding

**Threat.** With many groups tested, some will show an apparent gap by
chance alone. Reporting all of them as findings would produce
confidently wrong statutory documents and materially wrong remediation
costs.

**Mitigation.** Benjamini-Hochberg correction is applied across every
tested group before any verdict is reached, and findings use three
verdicts — explained, inconclusive, unexplained — rather than a binary
one, so a gap that cannot be distinguished from chance is never reported
as established (`docs/adr/0003-statistical-safeguards.md`). This runs in
the analytics layer itself; a prompt cannot switch it off.

### A drafted response is subtly wrong in a way no deterministic check catches

**Threat.** Some failures aren't a wrong number — they're a legal claim
stated with more certainty than the evidence supports, or a citation
that doesn't actually cover the claim it's attached to. Neither has a
mechanical test.

**Mitigation.** A second model, deliberately a different vendor from the
one that drafted the response, reviews every T2 draft against the
analysis and legal position it was built from
(`docs/adr/0005-model-access-layer.md`). Its findings are structured
(blocking vs. advisory), and a draft with unresolved blocking findings
is either sent back for revision (up to a fixed limit) or forwarded to a
human with the findings attached — never silently approved. This is the
one place in the system where a model's judgment is load-bearing rather
than advisory, and it is scoped as narrowly as possible: two judged
metrics, one agent, everything else deterministic.

### A statutory disclosure is sent without a person reviewing it

**Threat.** Any request touching comparator pay data is legally
consequential and should never be sent on a model's judgment alone —
including if every deterministic and model-based check above passes.

**Mitigation.** This is structural, not a check that could be skipped.
A T2 request has no code path to a completed status without passing
through `approval_node`, which calls `interrupt()` and genuinely stops
graph execution until a human supplies a decision — not a flag that
could be set programmatically, an actual pause in execution persisted
via LangGraph's checkpointer, resumable only by a caller outside the
graph. There is no tier-2 code path that reaches `END` without first
reaching `APPROVAL`.

### PII reaches a model or a stored transcript

**Threat.** A request naming a colleague ("how much does Maria earn")
could leak that name into a prompt, a trace, or a stored record, beyond
what's actually needed to serve the request.

**Mitigation.** Every request is redacted (Presidio + spaCy) before
intake classification or any downstream model call — the redacted text
is what agents and stored records see, not the original. A comparator
request remains identifiable as one without the name ever entering a
model's context.

### A response is lost, delayed, or silently incomplete due to infrastructure failure

**Threat.** A database outage, a network partition between components,
or a slow dependency could leave a request in an inconsistent state, or
cause the pipeline to produce a degraded or incorrect response without
anyone noticing.

**Mitigation.** Every node in the request graph runs with a retry policy
and timeout; a failure that survives retries routes to a fixed,
model-free degraded response rather than an error page — the employee is
told a person will follow up, and the request is never silently
dropped. Every settled request, complete or degraded, is durably
recorded twice: once in PostgreSQL for querying, and once as a
self-contained JSON audit bundle in Blob Storage, independent of the
database, so the record survives even if the database that produced it
does not.

This system's own deployment history is itself a case study in
infrastructure failures surfacing as silent gaps rather than loud
errors — see `docs/adr/0008-neo4j-http-transport.md` for one, and the
archival logging fix (an exception handler with no logging at all,
despite a docstring claiming otherwise) for another. Both are recorded
because a threat model that only lists intended defenses, and not the
real gaps found while building it, is less useful than one that's
honest about what actually broke.

## What this threat model does not claim

This is a portfolio project, evaluated against a synthetic dataset, not
an audited production system. The mitigations above are real and tested,
but "tested by this project's own suite" is a different claim from
"independently verified." Where that distinction matters, it's noted
above rather than glossed over.
