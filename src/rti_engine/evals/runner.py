"""Run the evaluation catalog and score what came back.

Two runners, because the two kinds of case differ in cost by two orders
of magnitude. Routing runs intake alone and can be run against every case
cheaply. Pipeline cases run the whole graph, so they are bounded by a
concurrency limit and worth running in a subset while iterating.

Routing failures are scored asymmetrically. A request that should have
been held for a human and was not released other people's pay without
review; one held unnecessarily cost a human five minutes. Reporting a
single accuracy figure would hide the difference between them.
"""

import asyncio
from typing import Any, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict

from rti_engine.agents.graph import build_graph
from rti_engine.agents.intake import classify_request
from rti_engine.agents.state import (
    RequestState,
    current_status,
    current_tier,
    initial_state,
)
from rti_engine.db.models import AutonomyTier, RequestStatus
from rti_engine.evals.cases import ScenarioCase, TierCase, scenario_cases, tier_cases
from rti_engine.evals.trajectory import check_trajectory
from rti_engine.mcp.analytics_server import DATASET_PATH

TIER_CONCURRENCY = 6
"""Routing calls are small; this is about provider rate limits."""

SCENARIO_CONCURRENCY = 3
"""Each of these is eight model calls and three minutes. More would
finish no sooner and would collide on rate limits."""

RECURSION_LIMIT = 40

TIER_ORDER = {AutonomyTier.T0: 0, AutonomyTier.T1: 1, AutonomyTier.T2: 2}


class TierOutcome(BaseModel):
    """What happened to one routing case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    expected: str
    actual: str
    passed: bool
    under_routed: bool
    """True where the request was handled at a lower tier than required.

    The failure that matters: data released without the review it needed.
    """

    over_routed: bool
    escalated_by_floor: bool
    error: str | None = None


class ScenarioOutcome(BaseModel):
    """What happened to one pipeline case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    scenario_id: str
    passed: bool
    completed: bool
    status: str
    tier: str | None
    verdict_expected: str
    verdict_actual: str | None
    figures_grounded: bool | None
    reviewer_approved: bool | None
    blocking_findings: list[dict[str, str]] = []
    """What the reviewer objected to on the final draft.

    Recorded because "the reviewer did not approve" is not actionable on
    its own: whether that reflects a defect in the drafting or a
    miscalibrated reviewer can only be told from what it said.
    """

    trajectory_valid: bool
    revisions: int
    tokens: int
    cost_usd: float
    failures: list[str]


class TierReport(BaseModel):
    """The routing suite's result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    outcomes: list[TierOutcome]

    @property
    def passed(self) -> bool:
        """Under-routing is disqualifying; over-routing is not.

        A request held for a human that need not have been is a cost. A
        request released that should have been held is a disclosure.
        """
        return not any(outcome.under_routed for outcome in self.outcomes)

    def summary(self) -> dict[str, Any]:
        return {
            "cases": len(self.outcomes),
            "correct": sum(outcome.passed for outcome in self.outcomes),
            "under_routed": sum(outcome.under_routed for outcome in self.outcomes),
            "over_routed": sum(outcome.over_routed for outcome in self.outcomes),
            "errors": sum(outcome.error is not None for outcome in self.outcomes),
            "passed": self.passed,
        }


class ScenarioReport(BaseModel):
    """The pipeline suite's result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    outcomes: list[ScenarioOutcome]

    @property
    def passed(self) -> bool:
        return all(outcome.passed for outcome in self.outcomes)

    def summary(self) -> dict[str, Any]:
        return {
            "cases": len(self.outcomes),
            "passed_cases": sum(outcome.passed for outcome in self.outcomes),
            "ungrounded": sum(outcome.figures_grounded is False for outcome in self.outcomes),
            "total_tokens": sum(outcome.tokens for outcome in self.outcomes),
            "total_cost_usd": round(sum(outcome.cost_usd for outcome in self.outcomes), 4),
            "passed": self.passed,
        }


async def _run_tier_case(case: TierCase, limit: asyncio.Semaphore) -> TierOutcome:
    """Classify one request and compare the tier against the catalog."""
    async with limit:
        try:
            result = await classify_request(case.request_text)
        except Exception as error:
            return TierOutcome(
                name=case.name,
                expected=case.expected_tier.value,
                actual="error",
                passed=False,
                under_routed=False,
                over_routed=False,
                escalated_by_floor=False,
                error=f"{type(error).__name__}: {error}",
            )

    if result.tier is None:
        # Every case in this catalog is a genuine pay request, so a case
        # classified as not one is a real failure to report, not a type
        # mismatch to route around.
        return TierOutcome(
            name=case.name,
            expected=case.expected_tier.value,
            actual="not_a_pay_request",
            passed=False,
            under_routed=False,
            over_routed=False,
            escalated_by_floor=False,
            error="classified as not a pay request",
        )

    actual = result.tier
    return TierOutcome(
        name=case.name,
        expected=case.expected_tier.value,
        actual=actual.value,
        passed=actual is case.expected_tier,
        under_routed=TIER_ORDER[actual] < TIER_ORDER[case.expected_tier],
        over_routed=TIER_ORDER[actual] > TIER_ORDER[case.expected_tier],
        escalated_by_floor=result.escalated,
    )


async def run_tier_suite(names: list[str] | None = None) -> TierReport:
    """Run the routing cases concurrently."""
    limit = asyncio.Semaphore(TIER_CONCURRENCY)
    cases = tier_cases(names)

    outcomes = await asyncio.gather(*(_run_tier_case(case, limit) for case in cases))
    return TierReport(outcomes=list(outcomes))


def _requester_for(case: ScenarioCase) -> str:
    """Find an employee whose group carries the case's planted anomaly."""
    frame = pd.read_parquet(DATASET_PATH)
    rows = frame[
        (frame.country == case.country)
        & (frame.job_family == case.job_family)
        & (frame.level == case.level)
    ]
    if rows.empty:
        raise LookupError(f"{case.name}: no employee in {case.country}/{case.job_family}")
    return str(rows.iloc[0]["employee_id"])


def _score_scenario(case: ScenarioCase, state: RequestState) -> list[str]:
    """Return every way this run failed its expectations."""
    failures: list[str] = []
    status = current_status(state)

    # The path is checked whatever the outcome: a request that ended in
    # the right place by the wrong route is still a defect.
    for violation in check_trajectory(state.get("audit", []), current_tier(state)):
        failures.append(f"trajectory: {violation.rule} — {violation.detail}")

    if not case.must_complete:
        if status is not RequestStatus.FAILED:
            failures.append(f"expected a refusal but the request reached {status.value}")
        return failures

    if status is not RequestStatus.AWAITING_APPROVAL:
        failures.append(f"expected awaiting_approval, reached {status.value}")

    check = state.get("number_check")
    if check is None:
        failures.append("figures were never validated")
    elif not check.grounded:
        values = ", ".join(figure.value for figure in check.ungrounded)
        failures.append(f"ungrounded figures in the letter: {values}")

    draft = state.get("draft")
    if draft is None:
        failures.append("no letter was produced")
    elif not draft.citations:
        failures.append("the letter cites no sources")

    return failures


async def _run_scenario_case(case: ScenarioCase, limit: asyncio.Semaphore) -> ScenarioOutcome:
    """Run one requester through the whole graph and score the result."""
    async with limit:
        employee_id = _requester_for(case)
        graph = build_graph()

        state = cast(
            RequestState,
            await graph.ainvoke(
                initial_state(f"eval-{case.name}", employee_id, case.request_text, case.country),
                config={"recursion_limit": RECURSION_LIMIT},
            ),
        )

    failures = _score_scenario(case, state)
    check = state.get("number_check")
    review = state.get("review")
    analysis = state.get("analysis")

    verdict: str | None = None
    if analysis is not None:
        from rti_engine.agents.drafter import base_salary_verdict

        verdict, _ = base_salary_verdict(analysis)
        if case.must_complete and verdict != case.expected_verdict:
            failures.append(f"verdict was {verdict}, the catalog expects {case.expected_verdict}")

    return ScenarioOutcome(
        name=case.name,
        scenario_id=case.scenario_id,
        passed=not failures,
        completed=state.get("draft") is not None,
        status=current_status(state).value,
        tier=tier.value if (tier := current_tier(state)) else None,
        verdict_expected=case.expected_verdict,
        verdict_actual=verdict,
        figures_grounded=check.grounded if check else None,
        reviewer_approved=review.approved if review else None,
        trajectory_valid=not check_trajectory(state.get("audit", []), current_tier(state)),
        blocking_findings=[
            {"kind": finding.kind, "quote": finding.quote, "problem": finding.problem}
            for finding in review.blocking
        ]
        if review
        else [],
        revisions=state.get("revision_count", 0),
        tokens=state.get("tokens_used", 0),
        cost_usd=round(state.get("cost_usd", 0.0), 4),
        failures=failures,
    )


async def run_scenario_suite(names: list[str] | None = None) -> ScenarioReport:
    """Run the pipeline cases, bounded by the concurrency limit."""
    limit = asyncio.Semaphore(SCENARIO_CONCURRENCY)
    cases = scenario_cases(names)

    outcomes = await asyncio.gather(*(_run_scenario_case(case, limit) for case in cases))
    return ScenarioReport(outcomes=list(outcomes))
