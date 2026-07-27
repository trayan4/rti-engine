# 0004 — Authorization enforced in code, not in prompts

- Status: accepted
- Date: 2026-07-27

## Context

The system operates at three levels of autonomy. Informational requests
reach no employee data. Own-data requests reach exactly one person's
record. Statutory disclosure requests may reach comparator groups, but
only in aggregate.

The obvious place to express those rules is the agent's instructions:
describe the tiers, describe what each may access, and rely on the model
to comply. That approach fails under exactly the conditions where it
matters most. A model that has been prompt-injected, or that has drifted
mid-conversation, will not report that it is exceeding its scope — it will
produce a fluent, confident answer containing another employee's salary.
Access to individual pay data is also the one thing in this domain that
cannot be undone once disclosed.

A second, quieter risk: if an agent supplies the employee identifier it is
querying for, then substituting a different identifier is a single-token
change to a tool call.

## Decision

Authorization is a function, not an instruction.

**Scope is decided by code.** An agent states what it wants; the
authorization module decides what it gets. Permitted scopes per tier are a
single mapping in one module, and a scope not in that mapping has no code
path that returns data.

**Requester identity comes from the authenticated principal**, established
by the application from the session, never from a request body or anything
an agent produced. An agent cannot query as someone else because it never
supplies the identity at all.

**Aggregates are stripped and size-checked.** Direct identifiers are
removed from every group result, and a group below the minimum reportable
size is refused, so a filter narrowed to one person cannot be used to read
that person's pay.

**Filterable columns are allow-listed.** Only declared group selectors may
be filtered on. Anything else is refused rather than ignored.

**Enforcement lives server-side.** The MCP analytics server calls this
module before returning data, and agents hold no database credentials. A
compromised agent has nothing to compromise with.

**A refusal is terminal.** Callers must not retry with a narrower query,
so that every refusal appears in the audit trail rather than being
silently worked around.

## Consequences

- The tier guarantees hold regardless of model behaviour, prompt content,
  or injected instructions.
- The rules are testable without a model in the loop, and are covered by
  unit tests asserting that each tier cannot exceed its scope.
- Prompts still describe the tiers, but only to help the agent behave
  sensibly. They are not the enforcement mechanism, and the distinction is
  documented so no future change mistakes one for the other.
- New data access paths must be added to the permitted-scope mapping
  deliberately, which is friction — intentionally.

## Alternatives considered

**Describe the rules in the system prompt and rely on compliance.**
Rejected: unenforceable, and undetectable when it fails.

**Filter results after the agent has retrieved them.** Rejected: the data
has already left the database and entered the model's context, so the
disclosure has effectively occurred.

**Row-level security in PostgreSQL.** A reasonable complement and worth
revisiting for the Azure deployment, but it protects only the SQL path.
Aggregate suppression and minimum group size are application concerns that
sit above it.
