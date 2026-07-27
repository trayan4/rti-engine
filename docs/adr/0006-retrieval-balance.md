# 0006 — Balanced retrieval across document kinds

- Status: accepted
- Date: 2026-07-27

## Context

The knowledge corpus has three layers, and answering a question correctly
usually requires all three: the directive states what EU law requires, a
national status note states what is actually in force in the requester's
country, and the company policy states what the employer has committed to.

These layers are not the same size. The directive contributes 106 of 138
chunks. The three national notes contribute nineteen between them.

They also differ in vocabulary. The directive is drafted in the register a
legal question tends to be phrased in; the national notes are shorter,
plainer, and describe status rather than obligation.

The result, on the first real queries run against the ingested corpus, was
that plain top-k similarity search returned nothing but directive and
policy chunks. Not one national chunk surfaced, across five different
queries — including one asking which pay gap percentage requires
justification, which returned the directive's 5% assessment trigger and
the company's 5% threshold while missing the 25% justification threshold
that actually applies under current Spanish law.

Two consequences followed. The system would have answered as though the
directive were in force in Germany, France and Spain, which it is not. And
the test asserting that one country's law never reaches another country's
requester was passing vacuously: no national chunks were being retrieved
at all, so no leakage was possible.

## Decision

Retrieve with **per-document-kind quotas**. A query runs once per kind
with its own limit, and the results are merged and re-sorted by score.
Default quotas are four chunks of legislation, three of national status,
and three of company policy.

Each layer of the legal picture is therefore represented in every result
set by construction, rather than by hoping that cosine similarity balances
a corpus it has no knowledge of.

Unbalanced search remains available for cases that genuinely want pure
ranking, and one test exercises it specifically to record the failure mode
this decision exists to prevent.

The isolation test was rewritten to query in the national notes' own
vocabulary, and to assert both that the requester's own country appears
and that the other two do not. A test that can pass while retrieving
nothing relevant is not a test.

## Consequences

- The national status notes are reachable, so the system can distinguish
  what the directive requires from what a country currently compels.
- Retrieval costs three queries rather than one, adding roughly a hundred
  milliseconds. Immaterial against a model call.
- Ranking across the merged set is less pure: a low-scoring national chunk
  may appear above a higher-scoring directive chunk that fell outside its
  kind's quota. That is the intended trade, since a complete picture beats
  an optimally ordered partial one.
- Quotas are a fixed policy rather than an adaptive one. If the corpus
  grows substantially they will need revisiting, and that is a deliberate
  point of friction.

## Alternatives considered

**Rely on the reranker planned for a later step.** Rejected: a reranker
reorders candidates, so it cannot surface a chunk that never entered the
candidate set. It remains valuable on top of quotas, not instead of them.

**Increase top-k until national chunks appear.** Rejected: it would take a
large k to reach them, filling the model's context with weakly relevant
legislation to retrieve three short chunks, and it would still be a
coincidence rather than a guarantee.

**Split the corpus into separate indexes per kind.** Rejected: it achieves
the same balance at the cost of managing three indexes and three
ingestion paths, for no gain over filtering metadata within one.
