# Intranasal vaccination, mechanism matching and economic uncertainty

Run these commands from the repository root in the study Python environment. Relative output directories in the commands are inside the checkout.

## Clinical input

The policy model reads the arithmetic culture-AUC ratio and its two interval endpoints from the adopted configuration. The BPZE1 and placebo arms comprised 24 and 21 challenged adults. Their mean AUCs were 3,971.94 and 69,698.38 CFU/mL × days. The whole-arm ratio was 0.05699 (within-arm bootstrap 95% interval 0.00261–0.27746). The public files contain the arm summaries and bootstrap interval endpoints. The original participant workbook is Supplementary Data 2 to Hill et al. (2026), released under CC BY 4.0. [Article and source workbook](https://doi.org/10.1038/s41541-026-01443-7)

To reproduce the clinical calculation and its 20,000 bootstrap replicates locally, download the workbook and run the extraction. The study copy has SHA-256 `52003adac37faead2ff4de4cfd5a0f66f9b453470e6be23f203c8855de5b1995`. The second command below builds the 17 positive product profiles from the resulting AUC distribution.

```sh
clinical_dir=outputs/core2026/05_evidence/mucosal_vaccines/clinical_sources
mkdir -p "$clinical_dir"
curl -fL 'https://static-content.springer.com/esm/art%3A10.1038%2Fs41541-026-01443-7/MediaObjects/41541_2026_1443_MOESM2_ESM.xlsx' \
    -o "$clinical_dir/41541_2026_1443_MOESM2_ESM.xlsx"
python -m scripts.analyse_bpze1_clinical_evidence_20261107
python -m scripts.build_bpze1_policy_design_20261107 --out reproduction/product_design.json
```

## Full policy grid

Each product profile is crossed with standalone and Tdap-coadministration routes at feasible combinations of maternal and adolescent coverage. The complete run generates 1,972 active trajectories and 116 zero-effect replays. The economic analysis uses 1,000 paired clinical and cost draws for each fixed transmission fit. The subgroup summaries use 20- and 30-year horizons.

```sh
python -m scripts.run_bpze1_policy_extension_20261107 --plan
python -m scripts.run_bpze1_policy_extension_20261107 --stage all --workers 2 --output-dir results/bpze1
python -m scripts.analyse_bpze1_policy_extension_20261107 --input-dir results/bpze1
python -m scripts.analyse_bpze1_policy_completion_20261107 --input-dir results/bpze1
python -m scripts.analyse_discount_sensitivity_20261113 --policy-dir results/bpze1 --output-dir reproduction/discount
```

The grid producer writes all profile-specific simulation and cost files to a fresh directory, `results/bpze1`. The first analysis computes age- and time-specific effects, dose gains and mechanism contrasts. The second compares the additional cost per nasal dose against the best standard programme at each DALY valuation. The discount-rate analysis revalues the reference profile at 0%, 3% and 5%, comparing the reconstructed 3% paired components with the original evaluation.

## Matched effects and alternative fitted histories

The adopted-history comparison matches the two mechanism profiles to half the smaller positive reduction in adolescent reported cases. The 365-day recovery and baseline full refits each supply their own history, comparator and matching target. These commands verify the 32 fitted-history matching conditions and the 64 maternal coverage and protection conditions directly from paired draws. The printed job counts include both horizons.

```sh
python -m scripts.analyse_matched_refit_followup_20261113 --stage check
python -m scripts.analyse_maternal_refit_interaction_20261113 --stage check
```

To calculate the underlying trajectories, use these stages after running the full policy grid. The grid stage compares the independent acquisition and onward effects on a 5 × 5 grid. The refits stage regenerates the trajectories for the history-specific comparisons. The mild-infectiousness and maternal-protection commands write new results to the requested directories.

```sh
python -m scripts.analyse_review_revision_20261110 --stage matched --workers 2
python -m scripts.analyse_review_round4_20261112 --stage grid --workers 2 --policy-dir results/bpze1
python -m scripts.analyse_review_round4_20261112 --stage refits --workers 2
python -m scripts.analyse_matched_refit_followup_20261113 --stage all --output-dir reproduction/matched
python -m scripts.analyse_maternal_refit_interaction_20261113 --stage all --output-dir reproduction/maternal_refits
python -m scripts.analyse_mild_transmission_sensitivity_20261117 --output-dir reproduction/mild
python -m scripts.analyse_maternal_assumption_sensitivity_20261117 --output-dir reproduction/maternal
```

## Transmission uncertainty

The saved inputs include 30 transmission parameter sets each for England and Brazil, the corresponding England mortality calibrations, and the paired simulated clinical and cost components. The summary pools 50 draws per parameter set. Historical sensitivity runs for Brazil and Japan bring the total to 66 blocks and 198 active intranasal trajectories.

```sh
python -m scripts.analyse_bpze1_transmission_followup_20261107 --stage summarize
python -m scripts.analyse_bpze1_transmission_followup_20261107 --stage check
```

For a complete recomputation, give the producer a fresh directory. For an individual England particle, append `--only local_England_000`. It rebuilds the standard coverage grid from the selected fitted parameters and compares the new economic components with the saved draws.

```sh
python -m scripts.analyse_bpze1_transmission_followup_20261107 \
    --output-dir reproduction/transmission --stage run --workers 2
```
