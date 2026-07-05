# Pertussis Immunization Strategy Simulator

This folder contains an R-only Shiny implementation of the interactive parts of
the pertussis transmission project. It does not use Python or reticulate.

## Run

```r
setwd("ShinyApp")
source("install_packages.R")
shiny::runApp(".", host = "0.0.0.0", port = 3838)
```

The app ships with the model settings, country profiles, calibration overlays,
and compact example tables needed for local use.

## Deploy To shinyapps.io

Deploy the whole `ShinyApp/` directory, not just `app.R`:

```r
rsconnect::deployApp("ShinyApp")
```

Runtime data are read from `data/` inside this directory. The app does not call
Python, `reticulate`, or files in the parent repository. On hosted Shiny
platforms the default strategy worker count is 1 to avoid exhausting the
container; set the environment variable `SHINYAPP_N_JOBS` if the deployment plan
has enough CPU for parallel strategy runs.

## Custom Country Uploads

Age profile CSV columns:

```text
age_group,population,vaccine_coverage,reporting_rate,symptom_probability
```

Contact matrix CSV columns:

```text
source_age_group,target_age_group,contacts_per_day
```

Rows must cover the eight model age groups exactly.

## Scope

The app is intended for rapid country adaptation and strategy comparison. It
implements the age-structured SIRWS model, sensitive/resistant strains, vaccine
origin states, intervention scenarios, summary outputs, and strategy ranking.
It intentionally omits the manuscript-only Bayesian grids, long PSA runs, and
publication figure pipeline.
