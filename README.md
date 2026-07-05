# Pertussis Transmission Analysis

This repository contains the public code, input data, source-data tables,
figures, and run metadata for an age-structured pertussis transmission analysis
of vaccination strategies and macrolide resistance.

The repository is intended for reproducibility and data availability. It does
not include submitted manuscript text, submission packages, internal working
records, reference PDFs, or bulk simulation intermediates.

## Contents

- `src_python/`: deterministic transmission model, calibration helpers, and
  scenario runners.
- `scripts_R/`: figure rendering scripts.
- `config/`: model settings and country profiles.
- `data/raw/` and `data/processed/`: public-source extracts and processed model
  inputs used by the analysis.
- `publication_inputs/*.csv`: generated scenario, parameter, prior, and country
  profile tables used by analysis and figure scripts.
- `outputs/tables/` and `outputs/summaries/`: retained source-data tables.
- `outputs/figures/` and `outputs/appendix/`: rendered main and extended figures.
- `outputs/metadata/`: run metadata and Bayesian beta-grid diagnostics.
- `outputs/calibrations/`: retained country calibration configurations.

Large time-series and grid intermediates under `outputs/simulations/` are not
tracked. They can be regenerated locally by the pipeline.

## Quick Start

Create the recommended environment:

```bash
conda create -y -n pertussis_model -c conda-forge python=3.12 numpy pandas scipy pyyaml tqdm joblib numba pyarrow pytest
conda activate pertussis_model
```

Install R packages:

```bash
Rscript install_R_packages.R
```

Run tests:

```bash
pytest
```

Run a public validation check against the retained source-data layer:

```bash
make validate-public
```

## Reproducing Outputs

The full analysis is compute-intensive. On a high-core workstation, increase
`N_JOBS`.

```bash
make data
make calibrate N_JOBS=40
make simulate N_JOBS=40
make bayesian N_JOBS=40
make hindcast N_JOBS=40
make publication-data N_JOBS=40
make figures
```

The retained CSV source-data tables are enough to inspect the figure inputs
without regenerating the large simulation parquet files. See
`SOURCE_DATA_MANIFEST.md` for the main mapping between rendered figures,
scripts, and source-data tables.
