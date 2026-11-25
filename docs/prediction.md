# Retrospective prediction comparisons

Run from the repository root with pandas installed.

```sh
python scripts/prediction_diagnostics.py
```

The script recreates tables S74 and S75 in `build/tables/`. It starts from the 1,186 saved native reporting intervals of the eight adopted fits and the observed and modelled counts in the source age groups. The two [source files for cases](../outputs/core2026/current_model_accuracy_20261103/period_predictions.csv.gz) and [ages](../outputs/core2026/current_preserving_submission_tables_20261103/S1_fit_age_composition_predictions.csv) contain aggregate counts.

Table S74 reports the sum of absolute interval errors divided by observed cases (WAPE), the later predicted-to-observed total ratio and the total variation distance between later age distributions. A missing later comparison is displayed as NA. Table S75 compares the model with three seasonal benchmarks in the six settings with later observations. Each forecast uses the previous year's value or the mean or median across the last three training years for the same calendar month or Japanese reporting week. All four forecasts use identical evaluation intervals. These later data had been seen during model development.

To compare the output with the displayed table cells, run

```sh
diff source_data/tables/tableS74.csv build/tables/tableS74.csv
diff source_data/tables/tableS75.csv build/tables/tableS75.csv
```
