# Pertussis Immunization Strategy Simulator

This folder contains an independent R-only Shiny implementation for rapid
exploratory scenario analysis. It does not use Python or reticulate, and it is
not a source of or a reproduction route for the analyses submitted with the
manuscript.

## Run

```r
setwd("ShinyApp")
source("install_packages.R")
shiny::runApp(".", host = "0.0.0.0", port = 3838)
```

The app ships with exploratory model settings, country profiles, legacy
country-specific starting overlays, and compact example tables needed for local
use.

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

The app is intended for rapid country adaptation and strategy comparison. Its
independent R implementation includes the age-structured SIRWS structure,
sensitive and resistant strains, vaccine-origin states, intervention scenarios,
summary outputs, and strategy ranking.

Core baseline values consumed with the same meaning as the submitted model are
aligned with `../config/model_settings.yaml`: latent, symptomatic-infectious,
and asymptomatic-infectious durations are 9, 24, and 17.3 days; recent and waned
vaccine-origin stages are 798 days each; both natural-immunity SIRWS stages are
6113.75 days; and sensitive-strain PEP effectiveness is 0.823.

The calibration and runtime paths remain deliberately different. The submitted
analysis used state-space MAP calibration with an annual latent AR(1)
transmission path, defined acceptance gates, and separate parametric-bootstrap
uncertainty analyses. The Shiny app instead uses legacy staged-fit values as
approximate country-specific starting overlays and offers an optional
one-dimensional beta grid adjustment. It also defaults to a shorter burn-in and
coarser interactive solver settings. These overlays and controls have not been
re-estimated under the aligned baseline values and must not be described as
submitted calibration results.

The app omits the manuscript uncertainty workflows, long sensitivity runs,
formal predictive-validation pipeline, and publication figure pipeline. Use the
archived repository configuration, scripts, and source data, rather than Shiny
outputs or downloaded Shiny configurations, to reproduce the submitted
analyses. The simulator is not a stand-alone policy recommendation tool.
