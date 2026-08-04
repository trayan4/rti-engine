# 0007 — Where measurement stops and judgment begins

- Status: accepted
- Date: 2026-08-03

## Context

The system had to be evaluated, and the obvious approach is to ask a
model whether the output was good. That approach would have measured
almost nothing here.

Most of what this system must get right is not a matter of degree. A
figure in a letter either appears in the fact sheet or it does not. A
request either reached the analytics server or it did not. A tier either
permits a scope or it does not. Asking a model to judge those introduces
a source of error into checks that have none, and the error is
asymmetric: a judge that occasionally accepts a fabricated figure is
worse than no check, because it produces a passing report.

There is a residue that genuinely is a matter of degree. Whether the
legal reasoning stayed inside the passages it was given. Whether those
passages were relevant to the question. Neither has a mechanical test,
and both matter.

## Decision

**Everything checkable is checked deterministically.**

- Figures are compared against the fact sheet and the legal position by
  parsing the letter and matching values. No model is involved.
- The path a request took is checked against its audit trail: that
  classification preceded everything, that no autonomous tier reached the
  disclosure pipeline, that every draft was validated before review, that
  no approval followed an absent review.
- Tier routing is scored against a fixed catalog, and scored
  **asymmetrically**. A request handled below its required tier released
  data without the review it needed; one handled above cost a person a
  short look. A single accuracy figure would average those together, so
  under-routing is disqualifying and over-routing is recorded.

**Only two things are judged by a model**, and both are questions of
degree over prose: whether the regulatory position is grounded in its
retrieved evidence, and whether that evidence was relevant to the legal
question asked.

**The judge is a different vendor from the model it scores**, using the
review role. A model asked to assess output from its own family tends to
find it reasonable. The judge is also built without fallbacks: a judge
that quietly degraded to the drafter's vendor would be scoring its own
family's work while reporting a score, and a score is not urgent enough
to be worth that.

**A judge that fails to produce a score is recorded as unscored, not as
zero.** The judge occasionally writes its verdict as prose rather than
returning it as a field, and a parse failure is a different finding from
a low score.

## Consequences

- The claims the system makes about itself are checkable. "No figure in
  any letter came from outside its sources" is a measurement, not an
  impression.
- The expensive, variable part of evaluation is confined to two metrics
  over one agent, so a full evaluation costs a few dollars rather than
  tens.
- Deterministic checks caught defects the LLM judge did not, and one that
  the reviewer agent had missed entirely: a structural exemption in the
  figure validator that let every whole-number percentage under a hundred
  pass unsourced.
- The judged metrics carry a threshold that is a judgment call. It is
  written down so that changing it is a visible decision rather than a
  moved goalpost.
- Adding a check means deciding which side of this line it falls on,
  which is friction, and is the point.

## Alternatives considered

**Judge the whole output with an LLM.** Rejected: it would replace exact
checks with approximate ones and report confidently either way.

**Skip LLM judgment entirely.** Rejected: it would leave the legal
reasoning — the one part with no deterministic test — unmeasured, and
that is where a wrong answer does the most damage.

**Use RAGAS for the judged metrics.** Not available: the package does not
import against this project's LangChain version, because it reaches for a
module that langchain-community has removed. openevals is used instead,
which is maintained alongside LangChain and provides equivalent
groundedness and retrieval-relevance prompts.
