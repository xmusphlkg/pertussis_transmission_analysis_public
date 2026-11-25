"""Setting-specific contact matrices from Prem et al.

Sources (``data/raw/external``, exported from the contactdata R package by
``scripts_R/core2026/export_prem_contact_sources.R``):

* ``core2026_prem_contact_matrices_by_location_16.csv``: 16 five-year groups
  (0-4 ... 75+), settings home, school, work, other; rows are participants
  (``source_age_bin``), columns their contacts (``target_age_bin``).
* ``core2026_prem_reference_population_16.csv``: the population of each source
  country by the same groups (``contactdata::age_df_countries``).

Primary analysis: every profile uses the 2021 release (labelled 2020 in
contactdata).  It does not contain Australia or Japan, so their contact
structure is taken from New Zealand and South Korea respectively and projected
onto their own population.  The ``own_country`` variant (sensitivity) uses
Australia's and Japan's own 2017-release matrices.

Processing for each setting:

1. merge 70-74 and 75+ into 70+ (participant rows averaged with the source
   population as weights, contact columns summed), giving 15 contact groups;
2. convert to intrinsic connectivity ``G_ij = C_ij * N / N_j`` with the source
   country's reference population and symmetrise ``G`` (the usual reciprocity
   correction ``(C_ij N_i + C_ji N_j) / (2 N_i)``);
3. project to the profile's WPP population of each calendar year,
   ``C_ij(y) = G_ij * N_j(y) / N(y)`` (Arregui et al. 2018), so contacts follow
   demographic change and stay exactly reciprocal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..grid import N_CONTACT_GROUPS
from ..model.tables import Calendar
from .demography import RAW, Demography, contact_group_population

PREM_MATRICES = RAW / "core2026_prem_contact_matrices_by_location_16.csv"
PREM_POPULATION = RAW / "core2026_prem_reference_population_16.csv"
SETTINGS: tuple[str, ...] = ("home", "school", "work", "other")
NONHOME: tuple[str, ...] = ("school", "work", "other")
N_PREM = 16
PREM_2021 = 2020  # contactdata label of the Prem et al. 2021 release
PREM_2017 = 2017

# profile -> (source country, release) for each contact variant.
CONTACT_SOURCES: dict[str, dict[str, tuple[str, int]]] = {
    "primary": {
        "Australia": ("New_Zealand", PREM_2021),
        "Brazil": ("Brazil", PREM_2021),
        "China": ("China", PREM_2021),
        "Japan": ("South_Korea", PREM_2021),
        "New_Zealand": ("New_Zealand", PREM_2021),
        "Sweden": ("Sweden", PREM_2021),
        "Thailand": ("Thailand", PREM_2021),
        "United_Kingdom": ("United_Kingdom", PREM_2021),
    },
}
CONTACT_SOURCES["own_country"] = {
    **CONTACT_SOURCES["primary"],
    "Australia": ("Australia", PREM_2017),
    "Japan": ("Japan", PREM_2017),
}


@dataclass(frozen=True)
class ContactTables:
    country: str
    variant: str
    source_country: str
    release: int
    first_year: int
    by_setting: dict[str, np.ndarray]   # setting -> (n_years, G, G)

    @property
    def home(self) -> np.ndarray:
        return self.by_setting["home"]

    @property
    def nonhome(self) -> np.ndarray:
        return sum(self.by_setting[s] for s in NONHOME)


def _prem_bins() -> list[str]:
    return [f"[{5 * k:02d},{5 * k + 5:02d})" for k in range(N_PREM)]


def load_prem(source_country: str, release: int) -> dict[str, np.ndarray]:
    raw = pd.read_csv(PREM_MATRICES)
    rows = raw[(raw["country"] == source_country) & (raw["data_source"] == release)]
    if rows.empty:
        raise KeyError(f"no Prem {release} matrices for {source_country}")
    bins = _prem_bins()
    out: dict[str, np.ndarray] = {}
    for setting in (*SETTINGS, "all"):
        sub = rows[rows["location"] == setting]
        mat = sub.pivot(index="source_age_bin", columns="target_age_bin", values="contacts_per_day")
        mat = mat.reindex(index=bins, columns=bins)
        if mat.isna().any().any():
            raise ValueError(f"{source_country} {release} {setting} matrix is incomplete")
        out[setting] = mat.to_numpy(dtype=float)
    total = sum(out[s] for s in SETTINGS)
    if not np.allclose(total, out["all"], rtol=1e-8, atol=1e-10):
        raise ValueError(f"{source_country}: settings do not sum to the all-setting matrix")
    return out


def reference_population(source_country: str) -> np.ndarray:
    raw = pd.read_csv(PREM_POPULATION)
    rows = raw[raw["country"] == source_country].set_index("age_bin")["population"]
    rows = rows.reindex(_prem_bins())
    if rows.isna().any():
        raise ValueError(f"reference population incomplete for {source_country}")
    return rows.to_numpy(dtype=float)


def merge_open_group(matrix16: np.ndarray, population16: np.ndarray) -> np.ndarray:
    """Merge the last two Prem groups (70-74, 75+) into one 70+ group."""

    m = np.asarray(matrix16, dtype=float)
    cols = np.concatenate([m[:, :-2], m[:, -2:].sum(axis=1, keepdims=True)], axis=1)
    w = population16[-2:] / population16[-2:].sum()
    out = np.empty((N_CONTACT_GROUPS, N_CONTACT_GROUPS))
    out[:-1, :] = cols[:-2, :]
    out[-1, :] = w[0] * cols[-2, :] + w[1] * cols[-1, :]
    return out


def merge_population(population16: np.ndarray) -> np.ndarray:
    return np.concatenate([population16[:-2], [population16[-2:].sum()]])


def reciprocity_error(contacts: np.ndarray, population: np.ndarray) -> float:
    total = contacts * population[:, None]
    return float(np.abs(total - total.T).sum() / total.sum())


def intrinsic_connectivity(contacts: np.ndarray, population: np.ndarray) -> np.ndarray:
    gamma = contacts * population.sum() / population[None, :]
    return 0.5 * (gamma + gamma.T)


def project(gamma: np.ndarray, population: np.ndarray) -> np.ndarray:
    return gamma * population[None, :] / population.sum()


def source_connectivity(source_country: str, release: int) -> dict[str, np.ndarray]:
    raw = load_prem(source_country, release)
    pop16 = reference_population(source_country)
    pop15 = merge_population(pop16)
    return {s: intrinsic_connectivity(merge_open_group(raw[s], pop16), pop15) for s in SETTINGS}


def contact_tables(
    country: str,
    demo: Demography,
    calendar: Calendar,
    variant: str = "primary",
) -> ContactTables:
    source_country, release = CONTACT_SOURCES[variant][country]
    gammas = source_connectivity(source_country, release)
    by_setting = {s: np.zeros((calendar.n_years, N_CONTACT_GROUPS, N_CONTACT_GROUPS)) for s in SETTINGS}
    for i in range(calendar.n_years):
        pop = contact_group_population(demo, calendar.first_year + i, N_CONTACT_GROUPS)
        for s in SETTINGS:
            by_setting[s][i] = project(gammas[s], pop)
    return ContactTables(country, variant, source_country, release, calendar.first_year, by_setting)
