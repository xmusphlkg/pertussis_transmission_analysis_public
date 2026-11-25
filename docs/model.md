# Running the transmission and economic models

Run the commands from the repository root with the Python environment described in the README. The BPZE1 producer uses a POSIX file lock on Linux or macOS. Use a fresh output directory inside the checkout for each design. The numerical inputs bundled here are sufficient for the commands below.

## Inputs and fitted parameters

The eight entries in `outputs/core2026/current_preserving_fit_map_20261103.json` identify the adopted fits. In each fit directory, `run_spec.json` gives the surveillance interval and age-count inputs, vaccination history, training years, model version, and fixed parameters. `complete.json` identifies the selected row in `fit_summary.csv`. The corresponding NumPy file stores the fitted coordinates on the optimisation scale. Thailand uses national diagnostic ascertainment. Paths in these specifications are relative to the checkout.

The surveillance inputs are aggregate counts. The demographic inputs are annual World Population Prospects estimates and England-specific population and birth series. The programme input uses the WHO/UNICEF DTP1 and DTP3 coverage records for these settings, including United Kingdom coverage where used for England. Social contact matrices and reference populations come from Prem et al. through the R package contactdata. Australia uses the New Zealand matrix, Japan uses South Korea and England uses the United Kingdom, with adjustment to the setting's population. Economic inputs include setting-specific costs, age-specific risks, life tables, and draws for infant mortality. Source references are recorded in the configuration files and source-data documentation.

The eight fits can be checked independently against the surveillance observations by reevaluating their negative log-posterior objectives.

```bash
python - <<'PY'
import json
from pathlib import Path

import numpy as np

from scripts.run_covid_refit_20261104 import build_problem
from src_python.pertussis.scenarios.fitted import load_fit

fits = json.loads(Path("outputs/core2026/current_preserving_fit_map_20261103.json").read_text())
for country, directory in fits.items():
    spec, row = load_fit(directory)
    coordinates = Path(directory) / f"start_{int(row.start_index)}_coordinates.npy"
    value = build_problem(spec, "baseline").objective_z(np.load(coordinates))
    np.testing.assert_allclose(value, row.objective, rtol=1e-8, atol=1e-5)
    print(country, value)
PY
```

## Standard vaccination programmes

Reconstruct each fitted history, project the 73 distinct coverage strategies over 2027–2056, and use 1,000 paired clinical and cost draws to value outcomes at 20 and 30 years. The policy decision compares net health benefit at each sourced DALY valuation. For an initial run of the projection alone, add `--countries England --workers 1` in place of `--workers 3`. The full economic and optimisation commands below use all eight settings.

```bash
export OPENBLAS_NUM_THREADS=1
python -m scripts.project_current_preserving_checked_20261105 \
  --workers 3 --output-dir results/standard
python -m scripts.evaluate_current_preserving_strategies_20261103 \
  --projection-root results/standard --n-draws 1000 --output-dir results/standard-economics
python -m scripts.optimize_current_preserving_strategies_20261103 \
  --economic-dir results/standard-economics --output-dir results/standard-decisions
```

The projection returns monthly typical onsets, mild onsets and infections by age, with annual maternal and adolescent dose counts. It checks the baseline programme against the retained 30-year reference series. The saved `projection_arrays.npz` in each setting's `current_preserving_strategies_20261103` directory permits direct revaluation. The economic programme defaults to these arrays when `--projection-root` is omitted.

## Intranasal vaccine

The adopted product design crosses 17 clinical profiles with two routes and feasible coverage levels. The plan reports 1,972 product trajectories and 116 zero-effect comparisons. A complete small run for the Swedish reference profile generates six product trajectories, six controls and paired clinical and cost outcomes. The final command repeats the evaluation for all settings and profiles. Numerical outputs include excess or averted cases, optimal programmes and cost ceilings against the best standard programme.

```bash
python -m scripts.run_bpze1_policy_extension_20261107 --plan
python -m scripts.run_bpze1_policy_extension_20261107 \
  --countries Sweden --profiles b050_a050_d03_point --workers 1 \
  --stage all --output-dir results/sweden-reference
python -m scripts.run_bpze1_policy_extension_20261107 \
  --workers 2 --stage all --output-dir results/bpze1
```

## Pandemic recovery sensitivity

The adopted 16 full-coordinate fits are indexed by `validated_fit_map.json` under `outputs/core2026/04_covid/full_refit_20261105T070745Z/`. In the same directory `warm_starts.json` lists the matched starting fits used for each optimisation. To reproduce the policy comparison from the supplied full refits, run the first command. The second command illustrates re-estimation of England's recovery scenario from its recorded starting fit. A full refit uses up to 150 optimisation iterations.

```bash
batch=outputs/core2026/04_covid/full_refit_20261105T070745Z
python -m scripts.run_covid_policy_20261104 \
  --fit-map "$batch/validated_fit_map.json" --observation-modes reference_1 \
  --n-draws 1000 --output-dir results/recovery-policy
python -m scripts.run_covid_refit_20261104 \
  --country England --scenario recovery_365 --scope full --iterations 150 --workers 2 \
  --parent outputs/core2026/covid_refit_20261104/England/recovery_365 \
  --output-root results/recovery-refits
```
