# Pertussis Transmission Analysis

This is the public reproducibility snapshot for the manuscript “Programme
strategies for post-pandemic pertussis control in children and adolescents: an
age-structured transmission modelling study”. It contains the current
deterministic scenario model, leakage-safe predictive validation, paired
Latin-hypercube sensitivity workflow, retained source-data tables, figures,
run metadata, and companion Shiny application source.

The release deliberately separates three quantities:

1. next-reporting-interval notification predictions from a stacked ensemble
   containing a semi-mechanistic discrepancy POMP;
2. deterministic conditional policy-scenario contrasts; and
3. paired input-sensitivity diagnostics that are not posterior probabilities.

The repository does not claim absolute national burden, calibrated prediction
in every country, validated annual forecasts to 2050, or posterior policy
intervals. The versioned estimands, gates, and deviations from the superseded
MCMC workflow are recorded in `ANALYSIS_PROTOCOL.md`.

## Contents

- `ANALYSIS_PROTOCOL.md`: versioned questions, estimands, validation rules,
  exclusions, claim boundaries, and deviations.
- `src_python/`: transmission model, calibration, simulation, and predictive
  validation code.
- `scripts_R/`: modular main and extended-figure rendering code.
- `config/`: model, profile, and sensitivity settings.
- `data/raw/` and `data/processed/`: redistributable source extracts and
  processed inputs.
- `manuscript_notes/`: a strict allowlist of scenario, parameter, and
  table-generation inputs required by the computational pipeline; manuscript
  drafts and internal review records are excluded.
- `outputs/tables/`, `outputs/summaries/`, and `outputs/metadata/`: retained
  source data, diagnostics, and run metadata.
- `outputs/figures/` and `outputs/appendix/`: rendered main and extended
  figures.
- `ShinyApp/`: source and compact data bundle for the companion exploratory
  simulator.
- `SOURCE_DATA_MANIFEST.md`: figure-to-script-to-source-data mapping.

Bulk regenerated time series and parquet mirrors are omitted to keep the public
snapshot tractable. They can be recreated by the pipeline.

## Environment

Python 3.12 is recommended. Create an isolated environment and install locked
project requirements:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
Rscript install_R_packages.R
```

## Verify the retained release

The fastest integrity check uses the retained outputs:

```bash
make validate-public
make test
```

Validation checks retained POMP/ensemble artifacts, audited source data,
explicit metadata fields, and Figure 2 inputs for internal consistency.

## Rebuild

Independent simulation and validation tasks use bounded process workers. Set
`N_JOBS` to suit the machine; the audited server used 96 workers and supports
up to 100.

```bash
make full N_JOBS=96
```

For a staged rebuild:

```bash
make data
make calibrate N_JOBS=96
make simulate N_JOBS=96
make calibration-hindcast N_JOBS=96
make panel-pomp-block-stress N_JOBS=96
make publication-data N_JOBS=96
make figures
make validate-public
```

The retired high-dimensional MCMC route remains available only for historical
method research and is not required by `make full`.

## Companion simulator

The hosted simulator is available at
https://lkg1116.shinyapps.io/pertussis_simulation/. To run it locally:

```r
setwd("ShinyApp")
source("install_packages.R")
shiny::runApp(".")
```

The simulator is an independent R implementation with simplified exploratory
defaults. The submitted analyses were not generated with and cannot be
reproduced from it; it is not a stand-alone policy recommendation tool.

## Licence and data access

Reuse is governed by `LICENSE`. Public raw sources are identified in the source
tables. Where third-party redistribution is restricted, this snapshot provides
the source location, access route, and processing code instead of redistributing
the restricted file.
