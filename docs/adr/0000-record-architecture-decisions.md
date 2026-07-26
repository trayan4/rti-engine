# 0000 — Record architecture decisions

- Status: accepted
- Date: 2026-07-26

## Context

This system makes non-obvious technical choices in a regulated domain, where the reasoning matters as much as the outcome. Without a record, the reasoning lives only in memory and is lost — leaving future readers unable to tell a deliberate trade-off from an accident.

## Decision

Record every significant architecture decision as a Markdown file in `docs/adr/`, using MADR format: sequential number, title, status, date, Context, Decision, Consequences, and where relevant the alternatives considered and why they were rejected.

Once an ADR is accepted it is immutable. A decision that changes is superseded by a new ADR that links back to the old one; the old file stays, with its status updated.

## Consequences

- The reasoning behind the architecture is auditable, which matters for a system whose whole premise is traceability.
- Rejected options are documented, so settled debates stay settled.
- Small cost per decision: one file, written at the time.
