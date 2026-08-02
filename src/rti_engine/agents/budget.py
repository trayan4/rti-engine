"""What one request may spend, and what happens when it cannot continue.

Two limits with the same consequence. A request that exceeds its token or
cost ceiling stops, and a request whose pipeline failed stops — in both
cases by routing to a degraded response rather than ending silently.

The ceiling exists because the cost of a request is not bounded by its
size. A Tier 2 request makes eight model calls, each node retries three
times, and the drafting loop runs twice — so a bad afternoon at a
provider multiplies into something worth capping.

The degraded response is deterministic. No model is called to produce it,
because the situation it exists for is one where model calls are already
failing.
"""

from typing import Any

from rti_engine.agents.drafter import DraftLetter, LetterSection

MAX_TOKENS_PER_REQUEST = 250_000
"""A Tier 2 request with two revisions uses roughly 60,000. This allows
several times that before intervening, so the ceiling catches a runaway
rather than a busy request."""

MAX_COST_USD_PER_REQUEST = 2.00
"""A Tier 2 request costs roughly fifteen cents. Same reasoning."""

DEGRADED_SUBJECT = "We have received your pay information request"


class BudgetExceededError(RuntimeError):
    """Raised when a request has spent more than it is allowed."""


def over_budget(tokens_used: int, cost_usd: float) -> str | None:
    """Return why a request must stop, or None if it may continue."""
    if tokens_used > MAX_TOKENS_PER_REQUEST:
        return (
            f"token budget exceeded: {tokens_used} used against a limit of {MAX_TOKENS_PER_REQUEST}"
        )
    if cost_usd > MAX_COST_USD_PER_REQUEST:
        return (
            f"cost budget exceeded: {cost_usd:.2f} USD against a limit of "
            f"{MAX_COST_USD_PER_REQUEST:.2f}"
        )
    return None


def degraded_letter(reason: str) -> DraftLetter:
    """Build the response sent when the pipeline could not complete.

    Deterministic by construction. The circumstances that produce it are
    ones where a model call is the thing that failed, so composing this
    with a model would be asking the broken part to explain itself.

    It states what happened without excusing it, and commits to a person
    picking the request up. It does not apologise at length or speculate
    about when — neither is information.
    """
    return DraftLetter(
        subject=DEGRADED_SUBJECT,
        salutation="Dear colleague,",
        sections=[
            LetterSection(
                heading="Your request has been received",
                body=(
                    "We have your request for pay information and it has been "
                    "recorded. Your right to receive this information is not "
                    "affected by this message."
                ),
            ),
            LetterSection(
                heading="Why this is not the full response",
                body=(
                    "Our system was unable to complete the analysis needed to "
                    "answer you. Rather than send you a partial or unverified "
                    "answer, we have passed your request to a member of the "
                    "People Operations team, who will respond to you directly."
                ),
            ),
            LetterSection(
                heading="What happens next",
                body=(
                    "A person will review your request and reply within the "
                    "period set out in the compensation policy. You do not need "
                    "to submit your request again."
                ),
            ),
        ],
        closing="Yours sincerely,",
        figures_used=[],
        citations=[],
    )


def degraded_detail(reason: str, errors: list[str]) -> dict[str, Any]:
    """Describe a degraded outcome for the audit trail.

    The reason is recorded for the operator, not for the employee: the
    letter says a person will follow up, and says nothing about token
    budgets or provider failures.
    """
    return {
        "reason": reason,
        "errors": errors[:5],
        "queued_for_manual_handling": True,
    }
