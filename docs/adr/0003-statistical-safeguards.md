# 0003 — Statistical safeguards on findings and remediation

- Status: accepted
- Date: 2026-07-26

## Context

The system produces statutory documents and costed remediation plans. Two
failure modes surfaced once the analytics ran against the full dataset,
both of which would have produced confidently wrong output.

**A single verdict for two different situations.** The remediation layer
initially skipped any group whose adjusted gap was not statistically
significant, describing all of them as "explained by controls". That was
true of one group, where a raw gap of 8.6% fell to -0.02% once tenure was
accounted for. It was false of the rest, where gaps of 6% to 8.5% survived
the controls almost intact and merely could not be distinguished from
chance in a group of fifty-odd people. Telling an employer that such a gap
is explained is a misstatement in a document with legal weight.

**Multiple comparisons.** Seventy-four groups were tested at a
significance level of 0.05, so three or four false positives were
statistically guaranteed. Three background groups with no planted anomaly
were flagged as unexplained gaps, attracting roughly EUR 239,000 of
recommended raises on the strength of random variation.

## Decision

**Three verdicts, not two.** Every assessed group is classified as
*explained* (adjusted gap within two percentage points of zero),
*inconclusive* (gap survives the controls but cannot be distinguished from
chance), or *unexplained* (gap survives the controls and is statistically
established). Each carries its own wording. Inconclusive groups are
reported for monitoring and explicitly described as not constituting
evidence of equal pay.

**Benjamini-Hochberg correction across the family of comparisons.** Every
assessable group is tested, and the resulting p-values are corrected for
the number of tests before any verdict is reached. Only findings that
clear the corrected threshold are treated as established.

**Both safeguards are enforced in the analytics layer,** not by the
calling agent. A prompt cannot switch them off.

## Consequences

- Remediation fell from thirteen groups and EUR 1.21m to two groups and
  EUR 281k. The two remaining are the deliberately planted anomalies.
- The system will not describe an unproven gap as explained.
- Genuine but modest gaps become harder to establish. A real 3.5% gap
  across six hundred employees no longer clears the corrected threshold
  and is reported as inconclusive rather than actionable. In a statutory
  context, wrongly asserting unlawful pay disparity is the worse error,
  but the trade-off is real and is stated in the output rather than
  hidden.
- Reported findings are conservative by construction, which is the correct
  default for a system whose output an employer may be required to act on.

## Alternatives considered

**Filter at the calling agent instead.** Rejected: it places a
six-figure error one prompt regression away. Correctness properties of
this weight belong in code.

**No multiple-comparison correction, relying on the analyst to judge.**
Rejected: the false positives are indistinguishable from real findings by
inspection, which is precisely why the correction exists.
