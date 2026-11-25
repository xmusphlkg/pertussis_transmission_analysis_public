"""Assemble a country's ``ModelInputs`` from demography, contacts, programme
and NPI inputs.

Calendar: ``spinup_years`` of constant early conditions (seasonality off)
before ``history_start``, then the dated history and projection to the end of
``last_year``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import CONFIG_DIR, REPO_ROOT
from ..grid import AgeGrid
from ..model.tables import Calendar, ModelInputs
from .contacts import contact_tables
from .demography import COUNTRY_ISO3, births_per_day, load_demography, nudge_targets
from .programme import Programme, encode, load_programme

NPI_TIMELINE = REPO_ROOT / "data" / "raw" / "covid_npi_contact_reduction_timeline.csv"
PROGRAMME_DIR = CONFIG_DIR / "programmes"
BORDER_IMPORT_MULTIPLIER = 0.1


@dataclass(frozen=True)
class BundleSettings:
    history_start_year: int = 1970
    spinup_years: int = 70
    last_year: int = 2050
    reference_year: int = 2019
    contact_variant: str = "primary"
    npi_variant: str = "mean"          # mean | lower | upper


def npi_reduction(country: str, calendar: Calendar, variant: str = "mean") -> np.ndarray:
    """Daily non-home contact reduction from the country NPI timeline.

    Within a period the reduction is the period value; after a period's end it
    returns linearly to zero over ``ramp_days`` unless the next period starts.
    """

    column = {"mean": "contact_reduction_mean", "lower": "contact_reduction_lower",
              "upper": "contact_reduction_upper"}[variant]
    table = pd.read_csv(NPI_TIMELINE)
    rows = table[table["country"] == country].sort_values("period_start")
    out = np.zeros(calendar.n_days)
    if rows.empty:
        return out
    days = calendar.dates
    for _, row in rows.iterrows():
        start = np.datetime64(row["period_start"], "D")
        end = np.datetime64(row["period_end"], "D") + np.timedelta64(1, "D")
        value = float(row[column])
        inside = (days >= start) & (days < end)
        out[inside] = np.maximum(out[inside], value)
        ramp = int(row.get("ramp_days", 0) or 0)
        if ramp > 0:
            after = (days >= end) & (days < end + np.timedelta64(ramp, "D"))
            elapsed = (days[after] - end).astype(int)
            out[after] = np.maximum(out[after], value * (1.0 - elapsed / ramp))
    return np.clip(out, 0.0, 1.0)


def importation_multiplier(npi: np.ndarray) -> np.ndarray:
    """Importation falls to ``BORDER_IMPORT_MULTIPLIER`` while NPIs are active."""

    return np.where(npi > 0.0, BORDER_IMPORT_MULTIPLIER, 1.0)


def programme_path(country: str) -> Path:
    return PROGRAMME_DIR / f"{country}.yaml"


def build_inputs(
    country: str,
    settings: BundleSettings = BundleSettings(),
    programme: Programme | None = None,
    delay_pmf: np.ndarray | None = None,
) -> tuple[ModelInputs, pd.DataFrame]:
    """Return the model inputs and the programme audit table for a profile."""

    if country not in COUNTRY_ISO3 and country != "England":
        raise KeyError(f"unknown profile {country}")
    first_year = settings.history_start_year - settings.spinup_years
    calendar = Calendar(first_year, settings.last_year - first_year + 1)
    grid = AgeGrid()
    regional = country == "England"
    if regional:
        from .england import load_england_demography, load_england_programme
        demo = load_england_demography()
        programme = programme or load_england_programme()
    else:
        demo = load_demography(COUNTRY_ISO3[country])
        programme = programme or load_programme(programme_path(country))
    proxy_country = "United_Kingdom" if regional else country
    contacts = contact_tables(proxy_country, demo, calendar, variant=settings.contact_variant)
    if programme.iso3 != COUNTRY_ISO3[proxy_country]:
        raise ValueError(f"programme iso3 {programme.iso3} does not match {country}")
    freeze = date(settings.history_start_year, 1, 1)
    tables = encode(programme, grid, calendar, freeze_before=freeze, delay_pmf=delay_pmf)
    npi = npi_reduction(proxy_country, calendar, settings.npi_variant)
    season_on = (calendar.year_idx + first_year >= settings.history_start_year).astype(float)
    inputs = ModelInputs(
        grid=grid,
        calendar=calendar,
        births_per_day=births_per_day(demo, calendar),
        maternal_effective=tables.maternal_effective,
        npi_reduction=npi,
        import_multiplier=importation_multiplier(npi),
        season_on=season_on,
        nudge_target=nudge_targets(demo, grid, calendar),
        primary_product=tables.primary_product,
        infant_dose=tables.infant_dose,
        contacts_home=contacts.home,
        contacts_nonhome=contacts.nonhome,
        contact_multiplier=np.ones(grid.n_bins),
        events=tables.events,
        reference_year=settings.reference_year,
        metadata={
            "country": country,
            "iso3": COUNTRY_ISO3[proxy_country],
            "programme_status": programme.status,
            "programme_path": str(programme.path) if programme.path else None,
            "contact_source": f"{contacts.source_country} Prem {contacts.release}",
            "contact_variant": settings.contact_variant,
            "history_start_year": settings.history_start_year,
            "spinup_years": settings.spinup_years,
        },
    )
    if regional:
        from .england import apply_observed_maternal
        apply_observed_maternal(inputs, programme)
        inputs.metadata.update({
            "geographic_scope": "England", "geography_code": "GB-ENG",
            "demography_source": "ONS England estimates1971–2025, 2024-based projection2026–2050",
            "births_pre1979": "development proxy: age-zero population",
            "contact_proxy": "United_Kingdom projected to England population",
            "npi_proxy": "United_Kingdom",
            "childhood_coverage_proxy": "UK programme and GBR WUENIC",
            "maternal_coverage": "England observed birth-month April2016–March2026; earlier proxy; later73% scenario",
        })
    return inputs, tables.audit
