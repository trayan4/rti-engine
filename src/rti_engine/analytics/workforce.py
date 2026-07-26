"""Baseline synthetic workforce generation.

Produces an employee population in which gender has no effect on pay.
Scenario anomalies are injected on top of this baseline by a separate
module, so that any pay gap measured in the final dataset is
attributable to a deliberately planted scenario rather than to an
artefact of the baseline draw.

All randomness flows from a single seeded NumPy generator, making the
output byte-for-byte reproducible.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from faker import Faker
from numpy.random import Generator

from rti_engine.analytics.catalog import Defaults

LEVEL_BASE_SALARY_EUR: dict[str, float] = {
    "L1": 40_000.0,
    "L2": 55_000.0,
    "L3": 72_000.0,
    "L4": 95_000.0,
    "L5": 130_000.0,
}
"""Full-time annual base salary anchor for each level, before adjustments."""

COUNTRY_PAY_FACTOR: dict[str, float] = {
    "DE": 1.00,
    "FR": 0.95,
    "ES": 0.80,
}
"""Country cost-of-labour multiplier applied to the level anchor."""

JOB_FAMILY_PAY_FACTOR: dict[str, float] = {
    "Sales": 1.00,
    "Engineering": 1.10,
    "Marketing": 0.95,
    "Legal": 1.05,
    "Operations": 0.90,
}
"""Job-family premium or discount applied to the level anchor."""

FAKER_LOCALE_BY_COUNTRY: dict[str, str] = {
    "DE": "de_DE",
    "FR": "fr_FR",
    "ES": "es_ES",
}

# LEVEL_SENIORITY_INDEX: dict[str, int] = {"L1": 0, "L2": 1, "L3": 2, "L4": 3, "L5": 4}

TENURE_PAY_COEFFICIENT = 0.015
"""Fractional base-pay uplift per year of tenure (1.5% per year)."""

PART_TIME_FTE = 0.6
"""Working fraction assigned to part-time employees."""

TENURE_GAMMA_SHAPE = 2.0
"""Shape of the gamma distribution used for tenure; produces a right skew."""

AGE_ENTRY_BASE = 22.0
PRIOR_EXPERIENCE_GAMMA_SHAPE = 3.0
PRIOR_EXPERIENCE_MEAN_YEARS_BY_LEVEL: dict[str, float] = {
    "L1": 2.5,
    "L2": 11.0,
    "L3": 17.0,
    "L4": 19.5,
    "L5": 21.5,
}
"""Mean years of prior career experience on joining, by level."""
MIN_AGE = 22
MAX_AGE = 67

PERFORMANCE_RATINGS: list[int] = [1, 2, 3, 4, 5]
PERFORMANCE_WEIGHTS: list[float] = [0.05, 0.15, 0.50, 0.22, 0.08]


@dataclass(frozen=True)
class GroupSpec:
    """One homogeneous block of employees to generate.

    A group is a single country / job family / level combination. Scenario
    populations and background filler are both expressed as groups, so the
    baseline builder needs no knowledge of scenarios.
    """

    country: str
    job_family: str
    level: str
    headcount: int
    female_share: float
    scenario_id: str | None = None
    sub_population: str | None = None


def _faker_for_country(country: str, seed: int) -> Faker:
    """Return a locale-appropriate, deterministically seeded Faker instance."""
    locale = FAKER_LOCALE_BY_COUNTRY.get(country, "en_US")
    faker = Faker(locale)
    faker.seed_instance(seed)
    return faker


def _draw_genders(headcount: int, female_share: float, rng: Generator) -> list[str]:
    """Assign genders by drawing against the group's female share.

    Drawn rather than allocated exactly, so group composition varies
    realistically instead of hitting the target share to the person.
    """
    draws = rng.random(headcount)
    return ["F" if draw < female_share else "M" for draw in draws]


def _draw_tenure_years(headcount: int, mean_tenure_years: float, rng: Generator) -> np.ndarray:
    """Draw tenure from a gamma distribution with the requested mean.

    Gamma is used because tenure is non-negative and right-skewed: most
    employees are relatively recent, a few have been there a long time.
    """
    scale = mean_tenure_years / TENURE_GAMMA_SHAPE
    return rng.gamma(shape=TENURE_GAMMA_SHAPE, scale=scale, size=headcount)


def _derive_ages(level: str, tenure_years: np.ndarray, rng: Generator) -> np.ndarray:
    """Derive ages from prior career experience plus tenure at this employer.

    Prior experience is drawn from a gamma distribution whose mean rises
    with level, so senior employees are older and the population contains
    enough people in each age band for age-based effects to be estimable.
    """
    mean_prior = PRIOR_EXPERIENCE_MEAN_YEARS_BY_LEVEL[level]
    prior_experience = rng.gamma(
        shape=PRIOR_EXPERIENCE_GAMMA_SHAPE,
        scale=mean_prior / PRIOR_EXPERIENCE_GAMMA_SHAPE,
        size=tenure_years.shape[0],
    )
    ages = AGE_ENTRY_BASE + prior_experience + tenure_years
    return np.clip(np.round(ages), MIN_AGE, MAX_AGE)


def _draw_fte(headcount: int, part_time_share: float, rng: Generator) -> np.ndarray:
    """Assign each employee a full-time-equivalent working fraction."""
    is_part_time = rng.random(headcount) < part_time_share
    return np.where(is_part_time, PART_TIME_FTE, 1.0)


def _compute_base_salary_fte(
    spec: GroupSpec,
    tenure_years: np.ndarray,
    defaults: Defaults,
    rng: Generator,
) -> np.ndarray:
    """Compute full-time-equivalent base salary.

    Gender is deliberately absent from this calculation. The multiplicative
    model is: level anchor x country factor x job family factor x tenure
    uplift x lognormal noise. Lognormal noise keeps salaries positive and
    right-skewed, matching real pay distributions.
    """
    anchor = LEVEL_BASE_SALARY_EUR[spec.level]
    country_factor = COUNTRY_PAY_FACTOR[spec.country]
    family_factor = JOB_FAMILY_PAY_FACTOR[spec.job_family]
    tenure_uplift = 1.0 + TENURE_PAY_COEFFICIENT * tenure_years
    noise = rng.lognormal(mean=0.0, sigma=defaults.base_salary_sigma, size=tenure_years.shape[0])
    return anchor * country_factor * family_factor * tenure_uplift * noise


def _draw_bonus_pct(headcount: int, defaults: Defaults, rng: Generator) -> np.ndarray:
    """Draw each employee's bonus as a fraction of base pay, floored at zero.

    Stored as a rate rather than a euro amount so that any later change to
    base pay flows through to bonus automatically.
    """
    drawn = rng.normal(
        loc=defaults.bonus_pct_mean,
        scale=defaults.bonus_pct_sigma,
        size=headcount,
    )
    bonus_pct: np.ndarray = np.maximum(drawn, 0.0)
    return bonus_pct


def build_group(
    spec: GroupSpec,
    defaults: Defaults,
    rng: Generator,
    seed: int,
    start_index: int,
) -> pd.DataFrame:
    """Generate one group of employees with no gender effect on pay."""
    faker = _faker_for_country(spec.country, seed + start_index)

    genders = _draw_genders(spec.headcount, spec.female_share, rng)
    tenure_years = _draw_tenure_years(spec.headcount, defaults.mean_tenure_years, rng)
    ages = _derive_ages(spec.level, tenure_years, rng)
    fte = _draw_fte(spec.headcount, defaults.part_time_share, rng)

    base_salary_fte = _compute_base_salary_fte(spec, tenure_years, defaults, rng)
    bonus_pct = _draw_bonus_pct(spec.headcount, defaults, rng)

    ratings = rng.choice(
        PERFORMANCE_RATINGS,
        size=spec.headcount,
        p=PERFORMANCE_WEIGHTS,
    )

    employee_ids = [f"EMP-{start_index + offset:05d}" for offset in range(spec.headcount)]
    full_names = [faker.name() for _ in range(spec.headcount)]

    return pd.DataFrame(
        {
            "employee_id": employee_ids,
            "full_name": full_names,
            "country": spec.country,
            "job_family": spec.job_family,
            "level": spec.level,
            "gender": genders,
            "age": ages.astype(int),
            "tenure_years": np.round(tenure_years, 2),
            "fte": fte,
            "working_pattern": np.where(fte < 1.0, "part_time", "full_time"),
            "base_salary_fte_eur": np.round(base_salary_fte, 2),
            "bonus_pct": np.round(bonus_pct, 4),
            "performance_rating": ratings.astype(int),
            "salary_period": "annual",
            "scenario_id": spec.scenario_id,
            "sub_population": spec.sub_population,
        }
    )


def recompute_derived_pay(workforce: pd.DataFrame) -> pd.DataFrame:
    """Recompute every pay column that depends on an underlying driver.

    The drivers are full-time-equivalent base salary, working fraction and
    bonus rate. Actual pay, bonus and total compensation are derived from
    them. Called once after generation and again after every scenario
    injection, so derived values can never drift out of step with the
    drivers a scenario has changed.
    """
    result = workforce.copy()
    result["working_pattern"] = np.where(result["fte"] < 1.0, "part_time", "full_time")
    result["base_salary_actual_eur"] = (result["base_salary_fte_eur"] * result["fte"]).round(2)
    result["bonus_actual_eur"] = (result["base_salary_actual_eur"] * result["bonus_pct"]).round(2)
    result["total_comp_actual_eur"] = (
        result["base_salary_actual_eur"] + result["bonus_actual_eur"]
    ).round(2)
    return result


def build_workforce(
    specs: list[GroupSpec],
    defaults: Defaults,
    rng: Generator,
    seed: int,
) -> pd.DataFrame:
    """Generate every group in order and concatenate into one workforce table."""
    frames: list[pd.DataFrame] = []
    next_index = 1

    for spec in specs:
        frames.append(build_group(spec, defaults, rng, seed, next_index))
        next_index += spec.headcount

    workforce = pd.concat(frames, ignore_index=True)
    return recompute_derived_pay(workforce)
