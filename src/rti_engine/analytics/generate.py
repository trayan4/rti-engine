"""End-to-end synthetic dataset generation.

Runs the full pipeline — gender-neutral baseline, calibrated scenario
injection, then data-quality defects — and writes the result to disk.

Each stage draws from its own random stream, derived from the single
catalog seed. Sharing one stream would mean that adding a draw in an
early stage silently shifted every later stage's numbers.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from rti_engine.analytics.catalog import Catalog, load_catalog
from rti_engine.analytics.data_quality import apply_data_quality_defects
from rti_engine.analytics.injection import apply_scenarios
from rti_engine.analytics.specs import build_group_specs
from rti_engine.analytics.workforce import build_workforce

DEFAULT_OUTPUT_PATH = Path("data/generated/workforce.parquet")

BASELINE_SEED_OFFSET = 0
INJECTION_SEED_OFFSET = 1
DEFECT_SEED_OFFSET = 2


def generate_workforce(catalog: Catalog) -> pd.DataFrame:
    """Build the complete synthetic workforce described by the catalog."""
    seed = catalog.generation.seed

    frame = build_workforce(
        build_group_specs(catalog),
        catalog.defaults,
        np.random.default_rng(seed + BASELINE_SEED_OFFSET),
        seed,
    )
    frame = apply_scenarios(frame, catalog, np.random.default_rng(seed + INJECTION_SEED_OFFSET))
    return apply_data_quality_defects(
        frame, catalog, np.random.default_rng(seed + DEFECT_SEED_OFFSET)
    )


def write_workforce(frame: pd.DataFrame, path: Path | None = None) -> Path:
    """Write the workforce to Parquet, creating the output directory if needed."""
    output_path = path if path is not None else DEFAULT_OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return output_path


def main() -> None:
    """Generate the dataset and write it to the default location."""
    catalog = load_catalog()
    frame = generate_workforce(catalog)
    path = write_workforce(frame)
    print(f"wrote {len(frame)} rows to {path}")


if __name__ == "__main__":
    main()
