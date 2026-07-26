"""Typed loader and schema for the synthetic-data anomaly catalog.

The catalog YAML is the single source of truth for both data generation
and evaluation. Every consumer reads it through this module so that a
malformed catalog fails at load time with a clear error, rather than
silently producing wrong data or wrong test expectations.
"""

from datetime import date
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_CATALOG_PATH = Path("data/anomaly_catalog.yaml")

Range = tuple[float, float]

Mechanism = Literal[
    "direct_base_multiplier",
    "tenure_skew",
    "bonus_multiplier",
    "interaction_multiplier",
    "part_time_skew",
    "data_quality_defects",
]

Metric = Literal["base_salary", "total_compensation"]


class StrictModel(BaseModel):
    """Base model: unknown keys are rejected and instances are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class GenerationConfig(StrictModel):
    """Global parameters controlling the shape of the generated workforce."""

    seed: int
    total_employees: int = Field(gt=0)
    countries: list[str] = Field(min_length=1)
    currency: str
    reference_date: date
    job_families: list[str] = Field(min_length=1)
    levels: list[str] = Field(min_length=1)


class Defaults(StrictModel):
    """Baseline population characteristics applied where a scenario is silent."""

    female_share: float = Field(ge=0.0, le=1.0)
    base_salary_sigma: float = Field(gt=0.0)
    bonus_pct_mean: float = Field(ge=0.0)
    bonus_pct_sigma: float = Field(ge=0.0)
    part_time_share: float = Field(ge=0.0, le=1.0)
    mean_tenure_years: float = Field(gt=0.0)


class Thresholds(StrictModel):
    """Regulatory and statistical cut-offs used by the analytics layer."""

    jpa_trigger_pct: float = Field(gt=0.0)
    significance_alpha: float = Field(gt=0.0, lt=1.0)
    min_reportable_group_size: int = Field(gt=0)


class MonthlySalaryRows(StrictModel):
    """Rows to be expressed as monthly rather than annual salary (S8)."""

    country: str
    count: int = Field(gt=0)


class Injection(StrictModel):
    """How a scenario perturbs its population.

    Only the fields relevant to the declared mechanism are set; the rest
    stay None. The generator dispatches on `mechanism`.
    """

    mechanism: Mechanism
    female_base_multiplier: float | None = Field(default=None, gt=0.0)
    female_bonus_multiplier: float | None = Field(default=None, gt=0.0)
    female_mean_tenure_years: float | None = Field(default=None, gt=0.0)
    male_mean_tenure_years: float | None = Field(default=None, gt=0.0)
    interaction_on: list[str] | None = None
    female_age_45_plus_multiplier: float | None = Field(default=None, gt=0.0)
    female_under_45_multiplier: float | None = Field(default=None, gt=0.0)
    female_part_time_share: float | None = Field(default=None, ge=0.0, le=1.0)
    male_part_time_share: float | None = Field(default=None, ge=0.0, le=1.0)
    missing_performance_rating_share: float | None = Field(default=None, ge=0.0, le=1.0)
    duplicate_row_count: int | None = Field(default=None, ge=0)
    monthly_salary_rows: MonthlySalaryRows | None = None


class GroundTruth(StrictModel):
    """The correct verdict for a scenario, used to grade the system."""

    must_flag: bool
    metric: Metric | None = None
    statistically_significant: bool | None = None
    attribution: str | None = None
    exceeds_jpa_threshold: bool | None = None
    severity: str | None = None
    required_language: str | None = None
    requires_interaction_model: bool = False
    requires_fte_normalization: bool = False
    expected_gap_pct: Range | None = None
    expected_raw_gap_pct: Range | None = None
    expected_adjusted_gap_pct: Range | None = None
    expected_base_gap_pct: Range | None = None
    expected_total_comp_gap_pct: Range | None = None
    expected_naive_gap_pct: Range | None = None
    expected_interaction_gap_pct: Range | None = None
    expected_unnormalized_gap_pct: Range | None = None
    expected_fte_normalized_gap_pct: Range | None = None
    must_detect: list[str] | None = None
    must_normalize: list[str] | None = None
    must_log: bool | None = None
    must_never: str | None = None


class Population(StrictModel):
    """The slice of the workforce a scenario applies to."""

    country: str | list[str] | None = None
    job_family: str | None = None
    level: str | None = None
    levels: list[str] | None = None
    headcount: int | None = Field(default=None, gt=0)
    female_share: float | None = Field(default=None, ge=0.0, le=1.0)
    applies_to: Literal["existing_rows"] | None = None


class SubPopulation(StrictModel):
    """One arm of a multi-arm scenario, with its own injection and verdict."""

    label: str
    country: str
    job_family: str
    level: str
    headcount: int = Field(gt=0)
    female_share: float = Field(ge=0.0, le=1.0)
    injection: Injection
    ground_truth: GroundTruth


class Scenario(StrictModel):
    """A single planted pay situation and the answer the system must reach."""

    id: str
    name: str
    description: str
    population: Population | None = None
    injection: Injection | None = None
    ground_truth: GroundTruth | None = None
    sub_populations: list[SubPopulation] | None = None

    @model_validator(mode="after")
    def check_exactly_one_shape(self) -> Self:
        """Enforce single-population or multi-arm form, never both or neither."""
        single = self.population is not None
        multi = self.sub_populations is not None

        if single == multi:
            raise ValueError(
                f"scenario {self.id}: define either 'population' or "
                f"'sub_populations', not both and not neither"
            )

        if single and (self.injection is None or self.ground_truth is None):
            raise ValueError(
                f"scenario {self.id}: a single-population scenario requires "
                f"both 'injection' and 'ground_truth'"
            )

        if multi and (self.injection is not None or self.ground_truth is not None):
            raise ValueError(
                f"scenario {self.id}: a multi-arm scenario carries 'injection' "
                f"and 'ground_truth' on each sub-population, not at the top level"
            )

        return self


class Catalog(StrictModel):
    """The full anomaly catalog."""

    version: int = Field(gt=0)
    generation: GenerationConfig
    defaults: Defaults
    thresholds: Thresholds
    scenarios: list[Scenario] = Field(min_length=1)

    @model_validator(mode="after")
    def check_unique_scenario_ids(self) -> Self:
        """Scenario IDs are referenced by the eval harness and must be unique."""
        seen: set[str] = set()
        for scenario in self.scenarios:
            if scenario.id in seen:
                raise ValueError(f"duplicate scenario id: {scenario.id}")
            seen.add(scenario.id)
        return self

    def scenario(self, scenario_id: str) -> Scenario:
        """Return one scenario by ID, raising if it does not exist."""
        for scenario in self.scenarios:
            if scenario.id == scenario_id:
                return scenario
        raise KeyError(f"no scenario with id {scenario_id!r}")


def load_catalog(path: Path | None = None) -> Catalog:
    """Load and validate the anomaly catalog from disk.

    Uses ``yaml.safe_load``, which refuses to construct arbitrary Python
    objects from the file, then validates the result against the schema
    above.
    """
    catalog_path = path if path is not None else DEFAULT_CATALOG_PATH
    with catalog_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return Catalog.model_validate(raw)
