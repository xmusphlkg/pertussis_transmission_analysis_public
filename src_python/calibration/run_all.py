from __future__ import annotations

import argparse

import pandas as pd

from src_python.calibration.calibrate_baseline import calibrate_country
from src_python.simulation.common import load_configs
from src_python.utils.io import ensure_output_dirs, project_path, write_dataframe
from src_python.utils.parallel import parallel_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Run country-level pertussis calibrations.")
    parser.add_argument(
        "--countries",
        nargs="*",
        default=None,
        help="Optional subset of country profile names to calibrate.",
    )
    parser.add_argument("--n-jobs", type=int, default=None, help="Number of countries to calibrate in parallel.")
    args = parser.parse_args()

    ensure_output_dirs()
    configs = load_configs()
    requested = args.countries or list(configs["countries"].keys())

    results = parallel_map(
        lambda country: calibrate_country(country),
        requested,
        desc="country_calibration",
        n_jobs=args.n_jobs,
    )
    summaries = [summary for _, summary in results]

    combined = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    write_dataframe(combined, project_path("outputs/tables/calibration_all_countries.csv"))


if __name__ == "__main__":
    main()
