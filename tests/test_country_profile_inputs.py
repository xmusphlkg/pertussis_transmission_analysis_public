from __future__ import annotations

import numpy as np
import pandas as pd

from src_python.data import build_who_inputs
from src_python.data.xlsx import read_xlsx_sheets
from src_python.data.build_country_inputs import load_incidence
from src_python.data.build_country_profile_inputs import build_country_profile_inputs
from src_python.data.build_who_inputs import build_ap_introduction, build_reported_cases
from src_python.utils import io as io_utils
from src_python.utils.io import project_path


def test_country_profile_inputs_extracts_measured_schedule_and_maternal_coverage() -> None:
    df = build_country_profile_inputs().set_index("config_key")

    japan = df.loc["Japan"]
    sweden = df.loc["Sweden"]
    united_states = df.loc["United_States"]
    brazil = df.loc["Brazil"]
    thailand = df.loc["Thailand"]
    south_africa = df.loc["South_Africa"]

    expected_dtp = {
        "Australia": (0.9300, 0.9266, "wuenic", "official"),
        "China": (0.9959, 0.9880, "official", "official"),
        "United_Kingdom": (0.9200, 0.9170, "wuenic", "official"),
        "Japan": (0.9782, 0.9862, "official", "official"),
        "New_Zealand": (0.9201, 0.8791, "official", "official"),
        "Sweden": (0.9700, 0.9500, "official", "official"),
        "United_States": (0.9800, 0.9400, "wuenic", "wuenic"),
        "Brazil": (0.8891, 0.8891, "official", "official"),
        "Thailand": (0.9203, 0.8922, "administrative", "administrative"),
        "South_Africa": (0.7590, 0.7390, "official", "official"),
    }
    for country, (dtp1, dtp3, dtp1_source_type, dtp3_source_type) in expected_dtp.items():
        row = df.loc[country]
        assert np.isclose(row["dtp1_coverage"], dtp1)
        assert np.isclose(row["dtp3_coverage"], dtp3)
        assert row["dtp1_coverage_source_type"] == dtp1_source_type
        assert row["dtp3_coverage_source_type"] == dtp3_source_type

    assert not bool(japan["maternal_program"])
    assert japan["maternal_coverage"] == 0.0
    assert japan["routine_age_pattern"] == "M2;M3;M4;Y1"
    assert japan["routine_dose_count"] == 4

    assert bool(sweden["maternal_program"])
    assert np.isclose(sweden["maternal_coverage"], 0.60)
    assert bool(sweden["adolescent_booster"])
    assert "Y14-Y16" in sweden["routine_age_pattern"]

    assert np.isclose(united_states["dtp1_coverage"], 0.98)
    assert np.isclose(united_states["dtp3_coverage"], 0.94)
    assert np.isclose(united_states["maternal_coverage"], 0.57)
    assert united_states["routine_dose_count"] == 6

    assert bool(brazil["maternal_program"])
    assert bool(brazil["adult_program"])
    assert bool(brazil["risk_program"])
    assert np.isclose(brazil["maternal_coverage"], 0.7412)
    assert brazil["routine_age_pattern"] == "M2;M4;M6;M15;Y4"

    assert bool(thailand["maternal_program"])
    assert not bool(thailand["adult_program"])
    assert not bool(thailand["risk_program"])
    assert np.isclose(thailand["maternal_coverage"], 0.0)
    assert thailand["maternal_program_note"] == "1st contact"
    assert thailand["routine_age_pattern"] == "M2;M4;M6;Y1.5;Y4"

    assert bool(south_africa["maternal_program"])
    assert np.isclose(south_africa["maternal_coverage"], 0.45)
    assert south_africa["maternal_coverage_source_type"] == "configuration_assumption"
    assert south_africa["routine_age_pattern"] == "W6;W10;W14;M18;Y6;Y12"
    assert np.isclose(south_africa["routine_first_shot_months"], 6 / 4.345238095238095)
    assert np.isclose(south_africa["routine_last_shot_months"], 144.0)


def test_incidence_loading_from_xlsx_normalizes_reporting_intervals() -> None:
    sheets = read_xlsx_sheets(project_path("data/raw/external/Pertussis Incidence Report.xlsx"))
    assert "BR " in sheets
    assert "TH" in sheets

    source = project_path("data/raw/external/Pertussis Incidence Report.xlsx")
    au = load_incidence(source, "AU")
    gb = load_incidence(source, "GB")
    br = load_incidence(source, "BR")
    th = load_incidence(source, "TH")

    assert au["reporting_frequency"].eq("monthly").all()
    assert gb["reporting_frequency"].eq("weekly").all()
    assert br["reporting_frequency"].eq("monthly").all()
    assert th["reporting_frequency"].eq("monthly").all()
    assert br["Year"].max() == 2026
    assert br["Country"].eq("BR").all()
    assert {"period_start", "period_end", "period_midpoint", "interval_days", "source_sheet"}.issubset(br.columns)
    assert br["interval_days"].between(28, 31).all()
    assert str(au["Date"].iloc[0].date()) == "2015-01-01"


def test_incidence_loading_can_still_cap_at_surveillance_year() -> None:
    df = load_incidence(
        project_path("data/raw/external/Pertussis Incidence Report.xlsx"),
        "AU",
        surveillance_year=2023,
    )

    assert df["Year"].max() == 2023
    assert not df["Year"].gt(2023).any()


def test_new_country_raw_dependencies_are_present() -> None:
    wpp = pd.read_csv(project_path("data/raw/external/wpp2024_population_by_age_sex_1990_2050.csv"))
    wpp_extended = pd.read_csv(project_path("data/raw/external/WPP data.csv"))
    contacts = pd.read_csv(project_path("data/raw/external/contactdata_prem_contact_matrices_16.csv"))
    resistance = pd.read_csv(project_path("data/raw/country_resistance_timeline.csv"))

    wpp_both = wpp.loc[wpp["Sex"].eq("Both sexes")]
    assert set(wpp_both.loc[wpp_both["Iso3"].isin(["BRA", "THA"]), "Iso3"]) == {"BRA", "THA"}
    assert wpp_both.loc[wpp_both["Iso3"].isin(["BRA", "THA"])].groupby("Iso3")["AgeStart"].nunique().eq(101).all()
    # Extended WPP file should cover 1950-2050 with age-0 population (indicator 47)
    wpp_ext_pop = wpp_extended.loc[wpp_extended["IndicatorId"].eq(47)]
    assert set(wpp_ext_pop.loc[wpp_ext_pop["Iso3"].isin(["BRA", "THA"]), "Iso3"]) == {"BRA", "THA"}
    assert wpp_ext_pop["Time"].min() <= 1950
    assert wpp_ext_pop["Time"].max() >= 2050
    assert set(contacts.loc[contacts["country"].isin(["Brazil", "Thailand"]), "country"]) == {"Brazil", "Thailand"}
    assert contacts.loc[contacts["country"].isin(["Brazil", "Thailand"])].groupby("country").size().eq(256).all()
    assert set(resistance.loc[resistance["country"].isin(["Brazil", "Thailand"]), "country"]) == {"Brazil", "Thailand"}
    assert not resistance["country"].eq("Singapore").any()
    us_resistance = resistance.loc[(resistance["country"].eq("United_States")) & (resistance["year"].eq(2024))]
    assert len(us_resistance) == 1
    assert np.isclose(us_resistance["resistant_fraction"].iloc[0], 0.01)
    assert us_resistance["evidence_type"].iloc[0] == "low_detected_model_anchor"
    assert "No population-based national resistant fraction" in us_resistance["notes"].iloc[0]


def test_who_country_level_outputs_are_limited_to_configured_countries() -> None:
    reported = build_reported_cases()
    introduced = build_ap_introduction()

    assert {"Brazil", "Thailand"}.issubset(set(reported["config_key"]))
    assert {"BRA", "THA"}.issubset(set(introduced["iso3"]))
    assert not reported["country_or_region"].eq("Singapore").any()
    assert not introduced["iso3"].eq("SGP").any()


def test_who_country_reference_uses_model_settings_without_generated_profiles(monkeypatch, tmp_path) -> None:
    settings_path = tmp_path / "config/model_settings.yaml"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        """
runtime:
  data_sources:
    who_ap_introduction_xlsx: data/raw/ap.xlsx
    countries:
      AU:
        config_key: Australia
        iso3: AUS
        contactdata_country: Australia
""".lstrip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(io_utils, "PROJECT_ROOT", tmp_path)

    def fake_read_xlsx(path, *, sheet_name=None):
        assert path == tmp_path / "data/raw/ap.xlsx"
        assert sheet_name is None
        return pd.DataFrame({"ISO_3_CODE": ["AUS"], "COUNTRYNAME": ["Australia"]})

    monkeypatch.setattr(build_who_inputs, "read_xlsx", fake_read_xlsx)

    reference = build_who_inputs._country_reference()

    assert not (tmp_path / "config/country_profiles.yaml").exists()
    assert reference.to_dict("records") == [
        {
            "config_key": "Australia",
            "iso3": "AUS",
            "config_country_name": "Australia",
            "who_country_name": "Australia",
        }
    ]
