"""Multiple-comparison correction and pay-gap verdict classification.

Two distinct problems live here.

The first is that testing many groups at a fixed significance level
guarantees false positives: seventy-four groups at alpha 0.05 will throw
up three or four apparently real gaps by chance alone. The
Benjamini-Hochberg procedure raises the bar in proportion to the number
of tests, so a finding has to be strong enough to stand out from the
whole family of comparisons rather than just from its own null.

The second is that "not significant" is not one verdict but two. A gap
that vanishes under controls is *explained*. A gap that survives the
controls almost intact but sits in a group too small to prove it is
*inconclusive*. Collapsing those into a single message would state
something false in a document with legal weight.
"""

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

EXPLAINED_RESIDUAL_PCT = 2.0
"""An adjusted gap smaller than this is treated as fully explained."""

GapVerdict = Literal["unexplained", "explained", "inconclusive"]


class GapClassification(BaseModel):
    """The verdict on one group's pay gap, and whether it warrants action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    verdict: GapVerdict
    q_value: float
    actionable: bool
    note: str


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Adjust p-values for multiple comparisons, preserving input order.

    Returns q-values: the smallest false-discovery rate at which each
    test would be declared significant. Compared against the same alpha
    as an ordinary p-value, but far harder to clear when many tests were
    run.
    """
    count = len(p_values)
    if count == 0:
        return []

    ranked = sorted(enumerate(p_values), key=lambda pair: pair[1])
    q_values = [0.0] * count
    running_minimum = 1.0

    # Walk from the largest p-value down, enforcing monotonicity: a q-value
    # can never exceed that of a less significant test.
    for rank in range(count, 0, -1):
        original_index, p_value = ranked[rank - 1]
        candidate = min(1.0, p_value * count / rank)
        running_minimum = min(running_minimum, candidate)
        q_values[original_index] = round(running_minimum, 6)

    return q_values


def classify_gap(
    raw_gap_pct: float,
    adjusted_gap_pct: float,
    q_value: float,
    alpha: float,
    explained_residual_pct: float = EXPLAINED_RESIDUAL_PCT,
) -> GapClassification:
    """Decide what a group's gap actually means, and how to describe it."""
    if q_value < alpha:
        return GapClassification(
            verdict="unexplained",
            q_value=q_value,
            actionable=True,
            note=(
                f"raw gap {raw_gap_pct:.2f}% remains at {adjusted_gap_pct:.2f}% after "
                f"controlling for level, job family, country and tenure, and is "
                f"statistically significant after correction for multiple comparisons "
                f"(q={q_value:.4f})"
            ),
        )

    if abs(adjusted_gap_pct) <= explained_residual_pct:
        return GapClassification(
            verdict="explained",
            q_value=q_value,
            actionable=False,
            note=(
                f"raw gap {raw_gap_pct:.2f}% falls to {adjusted_gap_pct:.2f}% once "
                f"level, job family, country and tenure are accounted for; the "
                f"difference is attributable to those factors"
            ),
        )

    return GapClassification(
        verdict="inconclusive",
        q_value=q_value,
        actionable=False,
        note=(
            f"raw gap {raw_gap_pct:.2f}% remains at {adjusted_gap_pct:.2f}% after "
            f"controls, but cannot be distinguished from chance in a group this "
            f"size (q={q_value:.4f}); this is not evidence of equal pay and the "
            f"group should be monitored"
        ),
    )
