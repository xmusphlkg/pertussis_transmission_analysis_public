# Pertussis vaccination in eight settings

Reproduction materials for *Potential infant health benefits and economic value of adolescent intranasal pertussis vaccination in eight settings: a modelling study*.

The study compares maternal and adolescent vaccination programmes in Australia, England, Sweden, Japan, New Zealand, China, Brazil and Thailand. The model evaluates infant outcomes and the delivered cost per intranasal dose at which a programme reaches the value of the best standard programme.

## Start here

Use Python 3.13.5 for the transmission and economic analyses. Run the commands from the repository root. The first example reproduces supplementary tables S74 and S75 from the observed and fitted surveillance counts.

```sh
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/prediction_diagnostics.py
diff source_data/tables/tableS74.csv build/tables/tableS74.csv
diff source_data/tables/tableS75.csv build/tables/tableS75.csv
```

The figures use R 4.5.0. The figure guide gives package versions, installation commands and links for each figure and table. [Figure and table guide](docs/displays.md)

| Reproduction task | Starting point |
| --- | --- |
| Inspect model assumptions and equations S1–S43 | [Equations](docs/equations.md) · [Configuration](config/) |
| Check the eight fitted models and run the standard and intranasal programme simulations | [Model instructions](docs/model.md) |
| Recalculate the clinical input, mechanism comparisons, price ceilings and sensitivity analyses | [Intervention and economics instructions](docs/analyses.md) |
| Rebuild ten infant and programme results tables from saved model outputs | [Result extraction and checksums](docs/results.md) |
| Recalculate retrospective predictions and seasonal benchmarks | [Prediction comparisons](docs/prediction.md) |
| Render the four main figures and 25 supplementary figures, or find the data for Tables 1–2 and S1–S84 | [Figure and table guide](docs/displays.md) · [Table index](source_data/tables/index.csv) |
| Trace demographic, surveillance, clinical and costing inputs to their sources | [Data sources and reuse](docs/data-sources.md) |

To reproduce the monthly age-accounting check (Table S6) and the infant mortality sensitivity analysis (Table S57), run:

```sh
python -m scripts.check_appendix_age_and_mortality age
python -m scripts.check_appendix_age_and_mortality mortality
```

The release provides the eight adopted fits, 16 alternative recovery fits, standard programme projections, selected intranasal trajectories and the summaries used in the article. A full run generates all 1,972 intranasal product trajectories and 116 zero-effect controls. The clinical input was derived from the public, de-identified participant-level AUC measurements in Hill et al. The repository includes the group estimates and extraction code. Obtain the original observations from the publisher to repeat that analysis. [Trial data and attribution](docs/data-sources.md)
