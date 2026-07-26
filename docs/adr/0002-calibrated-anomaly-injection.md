# 0002 — Calibrated anomaly injection

- Status: accepted
- Date: 2026-07-26

## Context

The synthetic dataset plants known pay anomalies whose correct analysis is
declared in advance, so that the system's output can be graded rather than
merely inspected. This only works if the dataset actually contains the gap
the catalog specifies.

A naive injection — multiply women's salaries in the affected group by
0.93 to create a 7% gap — does not achieve that. Salaries carry realistic
lognormal variation, so within a group of roughly 120 employees the
sampling difference between genders has a standard deviation of about
3.4% before any effect is planted. The realised gap is therefore the
specified gap plus noise, landing anywhere from 0% to 14% depending on the
seed.

Measured across the eight scenarios, seven of ten declared ground-truth
bands were missed. Two sub-populations of the same scenario, sharing an
identical mechanism, produced errors in opposite directions.

## Decision

Calibrate every injection. Before an effect is applied, the affected
group's female mean is scaled to match its male mean; the specified
multiplier is then applied to that equalised baseline. Where an effect is
not a simple multiplier — a bonus rate, a working-pattern skew — the
required parameter is solved for directly against the intended outcome.

Individual salaries retain their full natural variation. Only the group
mean is pinned.

Scenario group sizes were separately raised so that planted effects are
statistically detectable: a 7% gap in a 120-person group reaches
significance only about 73% of the time, which is not an acceptable basis
for a build-gating evaluation.

## Consequences

- Every scenario lands inside its declared band on every seed, so golden
  tests can assert exact figures rather than wide tolerances.
- A failing test indicates a real regression rather than an unlucky draw.
- The injection is no longer a naive multiplier, and the distinction
  between specified and realised effect must be understood by anyone
  reading the generator.
- Adjusted gaps estimated by regression still carry their own estimation
  error and cannot be pinned this way; their ground truth is expressed as
  a significance expectation rather than a narrow band.

## Alternatives considered

**Widen the ground-truth bands to about plus or minus four percentage
points.** Rejected: the threshold-straddle scenario exists to distinguish a
6.5% gap from a 3.5% one, and bands that wide would overlap, leaving the
scenario testing nothing.

**Reduce salary variance so planted effects dominate.** Rejected: real pay
distributions are dispersed, and an artificially tight one would make the
analytics look more reliable than they are.
