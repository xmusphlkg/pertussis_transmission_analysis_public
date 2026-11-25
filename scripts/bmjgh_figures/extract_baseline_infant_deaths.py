"""Write the reference programme's mean absolute infant deaths for Figure 1C."""

import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analyse_review_revision_20261110 import COUNTRIES, clinical


def main():
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["country", "horizon", "baseline_infant_deaths"])
    for country in COUNTRIES:
        folder = ROOT / "outputs/core2026/current_preserving_strategies_20261103" / country
        info = json.loads((folder / "complete.json").read_text())
        with np.load(folder / "projection_arrays.npz") as archive:
            current = archive["onsets"][list(archive["strategy_ids"].astype(str)).index("current")]
            # The clinical accounting returns first-arm minus each arm.
            # A zero-event second arm gives absolute deaths under current provision.
            outcomes = clinical(country, np.stack([current, np.zeros_like(current)]),
                                archive["month_starts"], archive["month_ends"], info)
        for horizon in (20, 30):
            draws = outcomes["deaths"][:, 1, :horizon, 0].sum(axis=1)
            if draws.shape != (1000,) or not np.all(np.isfinite(draws) & (draws > 0)):
                raise ValueError(f"Invalid baseline deaths for {country} at {horizon} years")
            writer.writerow([country, horizon, draws.mean()])


if __name__ == "__main__":
    main()
