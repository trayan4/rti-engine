"""Plant the catalogued pay anomalies into the baseline workforce.

Every mechanism is *calibrated*. Before an effect is applied, the affected
group's female mean is scaled to match its male mean, so the realised gap
equals the specified gap exactly rather than the specified gap plus
whatever sampling noise that particular draw produced.

This matters because the planted effects are small relative to natural
salary variation: a 7% effect in a 120-person group lands anywhere between
roughly 0% and 14% without calibration, which would make the catalog's
ground-truth bands unstatable. Individual salaries keep their full natural
spread; only the group mean is pinned.
"""

import numpy as np
import pandas as pd
from numpy.random import Generator

from rti_engine.analytics.catalog import Catalog, Defaults, Injection, Scenario
from rti_engine.analytics.workforce import (
    MAX_AGE,
    MIN_AGE,
    PART_TIME_FTE,
    TENURE_GAMMA_SHAPE,
    TENURE_PAY_COEFFICIENT,
    recompute_derived_pay,
)

BASE_FTE_COLUMN = "base_salary_fte_eur"
INTERSECTIONAL_AGE_CUTOFF = 45

Mask = "pd.Series[bool]"


def _group_mean(frame: pd.DataFrame, mask: pd.Series, column: str) -> float:
    """Mean of a column over the masked rows, as a plain float."""
    return float(frame.loc[mask, column].mean())


def _equalise(frame: pd.DataFrame, mask: pd.Series, column: str) -> None:
    """Scale women's values so their mean matches men's within the masked rows.

    This removes the sampling difference that exists before any effect is
    planted, so that whatever is applied next lands exactly.
    """
    female = mask & (frame["gender"] == "F")
    male = mask & (frame["gender"] == "M")
    if int(female.sum()) == 0 or int(male.sum()) == 0:
        return

    female_mean = _group_mean(frame, female, column)
    male_mean = _group_mean(frame, male, column)
    if female_mean <= 0.0:
        return

    frame.loc[female, column] = frame.loc[female, column] * (male_mean / female_mean)


def _assign_part_time(frame: pd.DataFrame, mask: pd.Series, share: float, rng: Generator) -> None:
    """Set an exact fraction of the masked rows to part time.

    An exact count is used rather than a per-person random draw, so the
    resulting working-pattern split is the specified one rather than an
    approximation of it.
    """
    index = frame.index[mask]
    count = len(index)
    if count == 0:
        return

    part_time_count = int(round(share * count))
    chosen = rng.permutation(count)[:part_time_count]
    frame.loc[index, "fte"] = 1.0
    frame.loc[index[chosen], "fte"] = PART_TIME_FTE


def _apply_direct_base_multiplier(
    frame: pd.DataFrame, mask: pd.Series, injection: Injection, scenario_id: str
) -> None:
    """Apply a flat pay penalty to women, calibrated to land exactly."""
    if injection.female_base_multiplier is None:
        raise ValueError(f"{scenario_id}: direct_base_multiplier needs female_base_multiplier")

    _equalise(frame, mask, BASE_FTE_COLUMN)
    female = mask & (frame["gender"] == "F")
    frame.loc[female, BASE_FTE_COLUMN] = (
        frame.loc[female, BASE_FTE_COLUMN] * injection.female_base_multiplier
    )


def _apply_tenure_skew(
    frame: pd.DataFrame,
    mask: pd.Series,
    injection: Injection,
    scenario_id: str,
    rng: Generator,
) -> None:
    """Redraw tenure by gender so the raw gap is real but fully explained.

    Salary is adjusted by the change in the tenure uplift only, and the
    tenure-free component of pay is then equalised across genders, so that
    every remaining difference is attributable to tenure and nothing else.
    """
    if injection.female_mean_tenure_years is None or injection.male_mean_tenure_years is None:
        raise ValueError(f"{scenario_id}: tenure_skew needs both mean tenure values")

    targets = {
        "F": injection.female_mean_tenure_years,
        "M": injection.male_mean_tenure_years,
    }

    for gender, mean_tenure in targets.items():
        selection = mask & (frame["gender"] == gender)
        count = int(selection.sum())
        if count == 0:
            continue

        old_tenure = frame.loc[selection, "tenure_years"].to_numpy()
        new_tenure = rng.gamma(
            shape=TENURE_GAMMA_SHAPE,
            scale=mean_tenure / TENURE_GAMMA_SHAPE,
            size=count,
        )
        uplift_ratio = (1.0 + TENURE_PAY_COEFFICIENT * new_tenure) / (
            1.0 + TENURE_PAY_COEFFICIENT * old_tenure
        )
        frame.loc[selection, BASE_FTE_COLUMN] = (
            frame.loc[selection, BASE_FTE_COLUMN].to_numpy() * uplift_ratio
        )
        shifted_age = frame.loc[selection, "age"].to_numpy() + (new_tenure - old_tenure)
        frame.loc[selection, "age"] = np.clip(np.round(shifted_age), MIN_AGE, MAX_AGE).astype(int)
        frame.loc[selection, "tenure_years"] = np.round(new_tenure, 2)

    tenure_uplift = 1.0 + TENURE_PAY_COEFFICIENT * frame.loc[mask, "tenure_years"]
    residual = frame.loc[mask, BASE_FTE_COLUMN] / tenure_uplift
    is_female = frame.loc[mask, "gender"] == "F"
    female_residual = float(residual[is_female].mean())
    male_residual = float(residual[~is_female].mean())
    if female_residual > 0.0:
        female = mask & (frame["gender"] == "F")
        frame.loc[female, BASE_FTE_COLUMN] = frame.loc[female, BASE_FTE_COLUMN] * (
            male_residual / female_residual
        )


def _apply_bonus_multiplier(
    frame: pd.DataFrame,
    mask: pd.Series,
    injection: Injection,
    defaults: Defaults,
    scenario_id: str,
    rng: Generator,
) -> None:
    """Move the gap entirely into variable pay, calibrated on total compensation.

    Working pattern is balanced across genders within this group first, so
    the total-compensation gap reflects bonus alone and is not confounded
    by part-time working. The bonus rate is then solved for directly, so
    the realised total-comp gap equals the intended one regardless of the
    group's own bonus level.
    """
    if injection.female_bonus_multiplier is None:
        raise ValueError(f"{scenario_id}: bonus_multiplier needs female_bonus_multiplier")

    for gender in ("F", "M"):
        _assign_part_time(frame, mask & (frame["gender"] == gender), defaults.part_time_share, rng)

    _equalise(frame, mask, BASE_FTE_COLUMN)
    _equalise(frame, mask, "bonus_pct")

    target_gap = (
        defaults.bonus_pct_mean
        * (1.0 - injection.female_bonus_multiplier)
        / (1.0 + defaults.bonus_pct_mean)
    )

    female = mask & (frame["gender"] == "F")
    male = mask & (frame["gender"] == "M")
    base_actual = frame[BASE_FTE_COLUMN] * frame["fte"]

    female_base = float(base_actual[female].mean())
    female_bonus_base = float((base_actual[female] * frame.loc[female, "bonus_pct"]).mean())
    male_total = float((base_actual[male] * (1.0 + frame.loc[male, "bonus_pct"])).mean())

    if female_bonus_base <= 0.0:
        raise ValueError(f"{scenario_id}: cannot calibrate bonus on a zero bonus base")

    solved_multiplier = (male_total * (1.0 - target_gap) - female_base) / female_bonus_base
    frame.loc[female, "bonus_pct"] = frame.loc[female, "bonus_pct"] * solved_multiplier


def _apply_interaction_multiplier(
    frame: pd.DataFrame, mask: pd.Series, injection: Injection, scenario_id: str
) -> None:
    """Penalise only women above the age cutoff.

    Each age band is equalised separately before the penalty is applied,
    so the effect is exact within the band that carries it and absent from
    the band that does not. A gender-only model sees a diluted gap; a model
    with a gender-by-age interaction recovers the full effect.
    """
    if (
        injection.female_age_45_plus_multiplier is None
        or injection.female_under_45_multiplier is None
    ):
        raise ValueError(f"{scenario_id}: interaction_multiplier needs both age-band multipliers")

    older = frame["age"] >= INTERSECTIONAL_AGE_CUTOFF
    _equalise(frame, mask & older, BASE_FTE_COLUMN)
    _equalise(frame, mask & ~older, BASE_FTE_COLUMN)

    female = mask & (frame["gender"] == "F")
    frame.loc[female & older, BASE_FTE_COLUMN] = (
        frame.loc[female & older, BASE_FTE_COLUMN] * injection.female_age_45_plus_multiplier
    )
    frame.loc[female & ~older, BASE_FTE_COLUMN] = (
        frame.loc[female & ~older, BASE_FTE_COLUMN] * injection.female_under_45_multiplier
    )


def _apply_part_time_skew(
    frame: pd.DataFrame,
    mask: pd.Series,
    injection: Injection,
    scenario_id: str,
    rng: Generator,
) -> None:
    """Concentrate part-time working among women without touching pay rates.

    Full-time-equivalent pay is equalised, so the gap exists only in actual
    amounts paid and disappears entirely once salaries are normalised to
    full-time.
    """
    if injection.female_part_time_share is None or injection.male_part_time_share is None:
        raise ValueError(f"{scenario_id}: part_time_skew needs both part-time shares")

    _equalise(frame, mask, BASE_FTE_COLUMN)
    _assign_part_time(frame, mask & (frame["gender"] == "F"), injection.female_part_time_share, rng)
    _assign_part_time(frame, mask & (frame["gender"] == "M"), injection.male_part_time_share, rng)


def _apply_injection(
    frame: pd.DataFrame,
    mask: pd.Series,
    injection: Injection,
    defaults: Defaults,
    scenario_id: str,
    rng: Generator,
) -> None:
    """Dispatch one injection onto the rows it applies to."""
    mechanism = injection.mechanism

    if mechanism == "direct_base_multiplier":
        _apply_direct_base_multiplier(frame, mask, injection, scenario_id)
    elif mechanism == "tenure_skew":
        _apply_tenure_skew(frame, mask, injection, scenario_id, rng)
    elif mechanism == "bonus_multiplier":
        _apply_bonus_multiplier(frame, mask, injection, defaults, scenario_id, rng)
    elif mechanism == "interaction_multiplier":
        _apply_interaction_multiplier(frame, mask, injection, scenario_id)
    elif mechanism == "part_time_skew":
        _apply_part_time_skew(frame, mask, injection, scenario_id, rng)
    elif mechanism == "data_quality_defects":
        return
    else:
        raise ValueError(f"{scenario_id}: unhandled mechanism {mechanism!r}")


def _apply_scenario(
    frame: pd.DataFrame, scenario: Scenario, defaults: Defaults, rng: Generator
) -> None:
    """Apply every injection belonging to one scenario."""
    if scenario.sub_populations is not None:
        for sub in scenario.sub_populations:
            mask = (frame["scenario_id"] == scenario.id) & (frame["sub_population"] == sub.label)
            _apply_injection(frame, mask, sub.injection, defaults, scenario.id, rng)
        return

    if scenario.injection is None:
        return

    mask = frame["scenario_id"] == scenario.id
    _apply_injection(frame, mask, scenario.injection, defaults, scenario.id, rng)


def apply_scenarios(workforce: pd.DataFrame, catalog: Catalog, rng: Generator) -> pd.DataFrame:
    """Plant every catalogued anomaly into a copy of the baseline workforce.

    Data-quality defects are not applied here; they operate on the finished
    table and are handled separately.
    """
    frame = workforce.copy()
    for scenario in catalog.scenarios:
        _apply_scenario(frame, scenario, defaults=catalog.defaults, rng=rng)
    return recompute_derived_pay(frame)
