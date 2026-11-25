# Recalculate the infant and programme outcomes

These commands reconstruct ten analysis tables from the included standard-programme projections, profile-specific age summaries, selected intranasal trajectories and matched-mechanism trajectories. Run them from the root of the checkout after installing the Python dependencies.

```bash
export PYTHONPATH=.
results=outputs/core2026/07_manuscript/model_study_revision/results_revision
python analysis/07_manuscript/model_study_revision/results_revision/programme_context.py
python analysis/07_manuscript/model_study_revision/results_revision/infant_outcomes.py
python analysis/07_manuscript/model_study_revision/results_revision/mechanism_outcomes.py
python scripts/bmjgh_figures/extract_maternal_infant_outcomes.py
python scripts/bmjgh_figures/extract_baseline_infant_deaths.py > "$results/baseline_infant_deaths_by_horizon.csv"
sha256sum -c docs/results.sha256
```

The programme tables give baseline estimates for all eight settings and the ten eligible expansions to 70% coverage. The profile tables cover 17 intranasal profiles, both routes and both time horizons. The mechanism tables retain all 16 setting–horizon combinations, including the four where the prespecified matching target could not be reached. The maternal-coverage and baseline-deaths tables use the same 1,000 paired clinical and economic draws as the main analysis. The checksums identify the files produced in the study software environment.
