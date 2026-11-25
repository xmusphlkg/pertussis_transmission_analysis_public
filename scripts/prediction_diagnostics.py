"""Recreate supplementary tables S74–S75 from interval and age-group counts."""
from pathlib import Path

import pandas as pd

DATA = Path("outputs/core2026")
COUNTRIES = ("Australia", "England", "Sweden", "Japan", "New_Zealand", "China", "Brazil", "Thailand")
FREQUENCY_LABELS = {
    "monthly": "Monthly",
    "weekly": "Weekly",
    "mixed_annual_quarterly_or_other": "Annual/quarterly",
}


def wape(observed, expected):
    denominator = observed.sum()
    if denominator == 0:
        return "NA"
    return f"{(expected - observed).abs().sum() / denominator:.1%}"


def seasonal_wape(training, later, cutoff, years, statistic):
    # The last complete reporting years before the forecast origin.
    year_range = list(range(cutoff - years + 1, cutoff + 1))
    history = training[training.reporting_year.isin(year_range)]
    target_seasons = set(later.season_key)
    if later.native_frequency.iloc[0] == "monthly":
        target_seasons = set(range(1, 13))
    for year in year_range:
        seasons = history.loc[history.reporting_year == year, "season_key"]
        if seasons.duplicated().any() or not target_seasons.issubset(seasons):
            raise ValueError(f"Missing or repeated reporting seasons for {year}")
    if history.period_end.max() > pd.Timestamp(cutoff + 1, 1, 1):
        raise ValueError("Benchmark training periods cross the forecast origin")
    seasonal = history.groupby("season_key").observed_cases.agg(statistic)
    expected = later.season_key.map(seasonal)
    if expected.isna().any():
        raise ValueError("No seasonal estimate for a later reporting interval")
    return wape(later.observed_cases, expected)


def main():
    period = pd.read_csv(DATA / "current_model_accuracy_20261103/period_predictions.csv.gz",
                         parse_dates=["period_end"])
    age = pd.read_csv(DATA / "current_preserving_submission_tables_20261103/S1_fit_age_composition_predictions.csv")
    accuracy, benchmarks = [], []
    for country in COUNTRIES:
        setting = country.replace("_", " ")
        values = period[period.country == country]
        training = values[values.role == "training"]
        later = values[values.role == "posttraining_holdout"]
        start, cutoff = int(training.reporting_year.min()), int(training.reporting_year.max())
        row = {
            "Setting": setting,
            "Fitted years": f"{start}–{cutoff}",
            "Observation interval": FREQUENCY_LABELS[training.native_frequency.iloc[0]],
            "Training WAPE": wape(training.observed_cases, training.expected_cases),
            "Later year": "NA",
            "Later WAPE": "NA",
            "Later predicted/observed total": "NA",
            "Later age TV": "NA",
        }
        accuracy.append(row)
        if later.empty:
            continue
        year, = later.reporting_year.unique()
        row["Later year"] = str(year)
        row["Later WAPE"] = wape(later.observed_cases, later.expected_cases)
        observed_total = later.observed_cases.sum()
        if observed_total > 0:
            row["Later predicted/observed total"] = f"{later.expected_cases.sum() / observed_total:.3f}"
        ages = age[(age.country == country) & (age.role == "posttraining_holdout") & (age.year == year)]
        if not ages.empty:
            totals = ages[["observed_cases", "expected_cases"]].sum()
            if (totals <= 0).any():
                raise ValueError(f"Undefined within-year age shares in {setting}")
            shares = ages[totals.index] / totals
            tv = (shares.expected_cases - shares.observed_cases).abs().sum() / 2
            row["Later age TV"] = f"{tv:.3f}"
        benchmarks.append({
            "Setting": setting,
            "Transmission model": row["Later WAPE"],
            "Previous year, same season": seasonal_wape(training, later, cutoff, 1, "mean"),
            "Three-year seasonal mean": seasonal_wape(training, later, cutoff, 3, "mean"),
            "Three-year seasonal median": seasonal_wape(training, later, cutoff, 3, "median"),
        })
    out = Path("build/tables")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(accuracy).to_csv(out / "tableS74.csv", index=False)
    pd.DataFrame(benchmarks).to_csv(out / "tableS75.csv", index=False)


if __name__ == "__main__":
    main()
