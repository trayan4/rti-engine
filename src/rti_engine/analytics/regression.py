"""Regression-based adjusted pay gap estimation.

The raw gap says women are paid less; it does not say why. This module
answers the question the directive actually asks: comparing employees at
the same level, in the same job family and country, with the same tenure,
does a pay difference by gender remain?

Pay is modelled on a log scale because pay effects are proportional
rather than absolute, which makes the gender coefficient directly
interpretable as a percentage. Men are the reference category, so a
negative coefficient means women earn less.

A second model adds a gender-by-age interaction. A gender-only model
averages an age-specific penalty across all women and understates it; the
interaction model estimates each age band separately and recovers it.

Every function here is deterministic and free of any model call.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pydantic import BaseModel, ConfigDict

from rti_engine.analytics.catalog import Thresholds

LOG_PAY_COLUMN = "log_pay"
AGE_FLAG_COLUMN = "age_45_plus"
INTERSECTIONAL_AGE_CUTOFF = 45

GENDER_TERM = 'C(gender, Treatment(reference="M"))'
"""Gender as a categorical with men as the baseline."""

FEMALE_PARAM = f"{GENDER_TERM}[T.F]"
"""Name of the fitted coefficient for women relative to men."""

INTERACTION_PARAM = f"{FEMALE_PARAM}:{AGE_FLAG_COLUMN}"

CATEGORICAL_CONTROLS: tuple[str, ...] = ("level", "job_family", "country")
NUMERIC_CONTROLS: tuple[str, ...] = ("tenure_years",)


class AdjustedGapResult(BaseModel):
    """A pay gap estimated after controlling for legitimate pay drivers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    column: str
    formula: str
    controls: list[str]
    n_observations: int
    coefficient: float
    adjusted_gap_pct: float
    p_value: float
    significant: bool
    alpha: float
    ci_low_pct: float
    ci_high_pct: float
    r_squared: float


class InteractionGapResult(BaseModel):
    """Pay gaps estimated separately for each age band."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    column: str
    formula: str
    n_observations: int
    age_cutoff: int
    younger_gap_pct: float
    younger_p_value: float
    older_gap_pct: float
    older_p_value: float
    older_significant: bool
    interaction_p_value: float
    interaction_significant: bool
    alpha: float
    r_squared: float


def _coefficient_to_gap_pct(coefficient: float) -> float:
    """Convert a log-scale coefficient into a percentage pay gap.

    A coefficient of -0.1054 means women earn exp(-0.1054) = 0.90 times
    what men earn, which is a 10% gap.
    """
    return float((1.0 - np.exp(coefficient)) * 100.0)


def _prepare(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Attach the log-pay response, dropping rows that cannot be logged."""
    usable = frame[frame[column].notna() & (frame[column] > 0)].copy()
    usable[LOG_PAY_COLUMN] = np.log(usable[column])
    return usable


def _select_controls(frame: pd.DataFrame) -> list[str]:
    """Keep only controls that actually vary within this population.

    A control with a single value across the whole group carries no
    information and would make the design matrix singular.
    """
    terms: list[str] = []
    for name in CATEGORICAL_CONTROLS:
        if name in frame.columns and int(frame[name].nunique(dropna=True)) > 1:
            terms.append(f"C({name})")
    for name in NUMERIC_CONTROLS:
        if name in frame.columns and float(frame[name].std()) > 0.0:
            terms.append(name)
    return terms


def estimate_adjusted_gap(
    frame: pd.DataFrame,
    column: str,
    thresholds: Thresholds,
) -> AdjustedGapResult:
    """Estimate the gender pay gap after controlling for pay drivers."""
    data = _prepare(frame, column)
    controls = _select_controls(data)
    formula = " + ".join([GENDER_TERM, *controls])
    formula = f"{LOG_PAY_COLUMN} ~ {formula}"

    fit = smf.ols(formula, data=data).fit()

    coefficient = float(fit.params[FEMALE_PARAM])
    p_value = float(fit.pvalues[FEMALE_PARAM])
    interval = fit.conf_int(alpha=thresholds.significance_alpha)
    lower_coef = float(interval.loc[FEMALE_PARAM, 0])
    upper_coef = float(interval.loc[FEMALE_PARAM, 1])

    return AdjustedGapResult(
        column=column,
        formula=formula,
        controls=controls,
        n_observations=int(fit.nobs),
        coefficient=round(coefficient, 6),
        adjusted_gap_pct=round(_coefficient_to_gap_pct(coefficient), 4),
        p_value=round(p_value, 6),
        significant=p_value < thresholds.significance_alpha,
        alpha=thresholds.significance_alpha,
        # A more negative coefficient is a larger gap, so the bounds swap.
        ci_low_pct=round(_coefficient_to_gap_pct(upper_coef), 4),
        ci_high_pct=round(_coefficient_to_gap_pct(lower_coef), 4),
        r_squared=round(float(fit.rsquared), 4),
    )


def estimate_interaction_gap(
    frame: pd.DataFrame,
    column: str,
    thresholds: Thresholds,
    age_cutoff: int = INTERSECTIONAL_AGE_CUTOFF,
) -> InteractionGapResult:
    """Estimate gender pay gaps separately above and below an age cutoff.

    The gap for the older band is the sum of the gender coefficient and
    the interaction coefficient, so its significance is tested as a linear
    combination of the two rather than read off either one.
    """
    data = _prepare(frame, column)
    data[AGE_FLAG_COLUMN] = (data["age"] >= age_cutoff).astype(int)

    controls = _select_controls(data)
    terms = [f"{GENDER_TERM} * {AGE_FLAG_COLUMN}", *controls]
    formula = f"{LOG_PAY_COLUMN} ~ " + " + ".join(terms)

    fit = smf.ols(formula, data=data).fit()

    younger_coef = float(fit.params[FEMALE_PARAM])
    younger_p = float(fit.pvalues[FEMALE_PARAM])
    interaction_p = float(fit.pvalues[INTERACTION_PARAM])

    combined = fit.t_test(f"{FEMALE_PARAM} + {INTERACTION_PARAM} = 0")
    older_coef = float(np.asarray(combined.effect).ravel()[0])
    older_p = float(np.asarray(combined.pvalue).ravel()[0])

    return InteractionGapResult(
        column=column,
        formula=formula,
        n_observations=int(fit.nobs),
        age_cutoff=age_cutoff,
        younger_gap_pct=round(_coefficient_to_gap_pct(younger_coef), 4),
        younger_p_value=round(younger_p, 6),
        older_gap_pct=round(_coefficient_to_gap_pct(older_coef), 4),
        older_p_value=round(older_p, 6),
        older_significant=older_p < thresholds.significance_alpha,
        interaction_p_value=round(interaction_p, 6),
        interaction_significant=interaction_p < thresholds.significance_alpha,
        alpha=thresholds.significance_alpha,
        r_squared=round(float(fit.rsquared), 4),
    )
