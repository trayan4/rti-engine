"""Run the evaluation suites from the command line.

Two suites with very different costs, so they are separate commands. The
routing suite is cheap enough to run on every change; the scenario suite
costs about a pound and ten minutes, and takes a subset for iteration.

    python -m rti_engine.evals routing
    python -m rti_engine.evals scenarios
    python -m rti_engine.evals scenarios s1_unexplained_gap
"""

import asyncio
import json
import sys

from rti_engine.evals.runner import run_scenario_suite, run_tier_suite


async def routing(names: list[str] | None) -> bool:
    """Run the tier-routing suite and print the outcome of each case."""
    report = await run_tier_suite(names)

    for outcome in report.outcomes:
        if outcome.under_routed:
            mark = "UNDER"
        elif outcome.passed:
            mark = "pass"
        else:
            mark = "over"
        print(f"  {mark:5s} {outcome.name:34s} {outcome.expected} -> {outcome.actual}")
        if outcome.error:
            print(f"        {outcome.error[:120]}")

    print(f"\n{json.dumps(report.summary(), indent=2)}")
    return report.passed


async def scenarios(names: list[str] | None) -> bool:
    """Run the pipeline suite and print what each case produced."""
    report = await run_scenario_suite(names)

    for outcome in report.outcomes:
        mark = "pass" if outcome.passed else "FAIL"
        print(f"\n  {mark}  {outcome.name} ({outcome.scenario_id})")
        print(f"        status={outcome.status} verdict={outcome.verdict_actual}")
        print(
            f"        grounded={outcome.figures_grounded} "
            f"revisions={outcome.revisions} ${outcome.cost_usd}"
        )
        for failure in outcome.failures:
            print(f"        - {failure}")
        for finding in outcome.blocking_findings:
            print(f"        [{finding['kind']}] {finding['problem'][:90]}")

    print(f"\n{json.dumps(report.summary(), indent=2)}")
    return report.passed


SUITES = {"routing": routing, "scenarios": scenarios}


def main() -> None:
    """Run one suite, exiting non-zero if it failed.

    The exit code is what a CI gate reads, so a failing suite must not
    exit cleanly however much it printed.
    """
    suite = sys.argv[1] if len(sys.argv) > 1 else ""
    if suite not in SUITES:
        print(f"usage: python -m rti_engine.evals {{{'|'.join(SUITES)}}} [case ...]")
        raise SystemExit(2)

    names = sys.argv[2:] or None
    raise SystemExit(0 if asyncio.run(SUITES[suite](names)) else 1)


if __name__ == "__main__":
    main()
