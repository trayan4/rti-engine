"""Translate catalog scenarios into concrete generation groups.

The catalog describes scenarios in whatever shape is clearest for a human
reader: one population, several sub-populations, or a set of levels sharing
a headcount. The workforce builder needs a flat list of homogeneous groups.
This module performs that translation, and fills the remainder of the
target headcount with background employees in cells no scenario occupies.
"""

from rti_engine.analytics.catalog import Catalog, Scenario
from rti_engine.analytics.workforce import GroupSpec

Cell = tuple[str, str, str]
"""A country / job family / level coordinate."""


def _split_headcount(total: int, parts: int) -> list[int]:
    """Divide a headcount as evenly as possible, giving remainders to the front."""
    if parts <= 0:
        raise ValueError("parts must be positive")
    base, remainder = divmod(total, parts)
    return [base + (1 if index < remainder else 0) for index in range(parts)]


def _scenario_groups(scenario: Scenario, default_female_share: float) -> list[GroupSpec]:
    """Expand one scenario into the groups it requires.

    Returns an empty list for scenarios that modify existing rows rather
    than creating their own population (S8).
    """
    if scenario.sub_populations is not None:
        return [
            GroupSpec(
                country=sub.country,
                job_family=sub.job_family,
                level=sub.level,
                headcount=sub.headcount,
                female_share=sub.female_share,
                scenario_id=scenario.id,
                sub_population=sub.label,
            )
            for sub in scenario.sub_populations
        ]

    population = scenario.population
    if population is None:
        raise ValueError(f"scenario {scenario.id}: no population and no sub_populations")

    if population.applies_to == "existing_rows":
        return []

    if not isinstance(population.country, str):
        raise ValueError(
            f"scenario {scenario.id}: a generating population needs exactly one country"
        )
    if population.job_family is None:
        raise ValueError(f"scenario {scenario.id}: a generating population needs a job_family")
    if population.headcount is None:
        raise ValueError(f"scenario {scenario.id}: a generating population needs a headcount")

    female_share = (
        population.female_share if population.female_share is not None else default_female_share
    )

    if population.levels is not None:
        counts = _split_headcount(population.headcount, len(population.levels))
        return [
            GroupSpec(
                country=population.country,
                job_family=population.job_family,
                level=level,
                headcount=count,
                female_share=female_share,
                scenario_id=scenario.id,
            )
            for level, count in zip(population.levels, counts, strict=True)
        ]

    if population.level is None:
        raise ValueError(f"scenario {scenario.id}: a generating population needs level or levels")

    return [
        GroupSpec(
            country=population.country,
            job_family=population.job_family,
            level=population.level,
            headcount=population.headcount,
            female_share=female_share,
            scenario_id=scenario.id,
        )
    ]


def _free_cells(catalog: Catalog, occupied: set[Cell]) -> list[Cell]:
    """Every country/family/level cell not claimed by a scenario, in stable order."""
    return [
        (country, job_family, level)
        for country in catalog.generation.countries
        for job_family in catalog.generation.job_families
        for level in catalog.generation.levels
        if (country, job_family, level) not in occupied
    ]


def _filler_groups(catalog: Catalog, occupied: set[Cell], headcount: int) -> list[GroupSpec]:
    """Distribute the remaining headcount across unclaimed cells."""
    if headcount <= 0:
        return []

    cells = _free_cells(catalog, occupied)
    if not cells:
        raise ValueError("no free cells available for background population")

    counts = _split_headcount(headcount, len(cells))
    return [
        GroupSpec(
            country=country,
            job_family=job_family,
            level=level,
            headcount=count,
            female_share=catalog.defaults.female_share,
        )
        for (country, job_family, level), count in zip(cells, counts, strict=True)
        if count > 0
    ]


def build_group_specs(catalog: Catalog) -> list[GroupSpec]:
    """Produce the full list of groups needed to generate the workforce.

    Scenario groups come first, in catalog order, followed by background
    filler groups sized to reach the target total headcount.
    """
    scenario_specs: list[GroupSpec] = []
    for scenario in catalog.scenarios:
        scenario_specs.extend(_scenario_groups(scenario, catalog.defaults.female_share))

    occupied: set[Cell] = {(spec.country, spec.job_family, spec.level) for spec in scenario_specs}

    planted = sum(spec.headcount for spec in scenario_specs)
    remaining = catalog.generation.total_employees - planted
    if remaining < 0:
        raise ValueError(
            f"scenario headcounts total {planted}, which exceeds the target "
            f"{catalog.generation.total_employees}"
        )

    return scenario_specs + _filler_groups(catalog, occupied, remaining)
