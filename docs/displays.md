# Figure and table source data

Run the figure scripts from the repository root. They read aggregate results and write to `build/figures/`. Each script uses the numbering of the current article and supplement.

```sh
Rscript -e 'install.packages(c("ggplot2", "patchwork", "scales", "tidyr", "svglite", "ragg", "jsonlite"), repos = "https://cloud.r-project.org")'
Rscript scripts/bmjgh_figures/figure1.R
```

To render every numbered figure on a system with a POSIX shell, run

```sh
for script in scripts/bmjgh_figures/figure*.R; do
    Rscript "$script" || exit 1
done
```

The plots were tested with R 4.5.0, ggplot2 4.0.3, patchwork 1.3.2, scales 1.4.0, tidyr 1.3.2, svglite 2.2.2, ragg 1.5.2 and jsonlite 2.0.0. Cairo graphics and an Arial-compatible font are used for the PDF output.

## Figures

The four main figures have 27 panel files. The combined [main-figure estimates](../source_data/figures/main_figure_estimates.csv) give estimates and interval endpoints in outcome units before axis transformations. The [economic strategy mapping](../source_data/figures/economic_strategy_mapping.csv) identifies the selected route, programme and comparator at each valuation. In S13, `clinical_draw` indexes a simulation draw.

| Figure | Comparison | R script | Source data |
| --- | --- | --- | --- |
| 1 | Projected infant outcomes across settings, vaccine profiles and evaluation horizons | [R](../scripts/bmjgh_figures/figure1.R) | [A](../source_data/figures/figure1_A.csv), [B](../source_data/figures/figure1_B.csv), [C](../source_data/figures/figure1_C.csv), [D](../source_data/figures/figure1_D.csv), [E](../source_data/figures/figure1_E.csv), [F](../source_data/figures/figure1_F.csv), [G](../source_data/figures/figure1_G.csv) |
| 2 | Transmission mechanisms and infant protection | [R](../scripts/bmjgh_figures/figure2.R) | [A](../source_data/figures/figure2_A.csv), [B](../source_data/figures/figure2_B.csv), [C](../source_data/figures/figure2_C.csv), [D](../source_data/figures/figure2_D.csv), [E](../source_data/figures/figure2_E.csv), [F](../source_data/figures/figure2_F.csv) |
| 3 | Additional infant protection across maternal vaccination coverage | [R](../scripts/bmjgh_figures/figure3.R) | [A](../source_data/figures/figure3_A.csv), [B](../source_data/figures/figure3_B.csv), [C](../source_data/figures/figure3_C.csv), [D](../source_data/figures/figure3_D.csv), [E](../source_data/figures/figure3_E.csv), [F](../source_data/figures/figure3_F.csv), [G](../source_data/figures/figure3_G.csv), [H](../source_data/figures/figure3_H.csv) |
| 4 | Delivered-cost ceilings and their determinants | [R](../scripts/bmjgh_figures/figure4.R) | [A](../source_data/figures/figure4_A.csv), [B](../source_data/figures/figure4_B.csv), [C](../source_data/figures/figure4_C.csv), [D](../source_data/figures/figure4_D.csv), [E](../source_data/figures/figure4_E.csv), [F](../source_data/figures/figure4_F.csv) |
| S1 | Evidence and decision framework | [R](../scripts/bmjgh_figures/figureS1.R) | Schematic nodes and connections in the script |
| S2 | Assumed persistence of two-stage nasal protection | [R](../scripts/bmjgh_figures/figureS2.R) | Erlang-2 survival function in the script |
| S3 | Additional infant protection from coadministration across maternal vaccination coverage | [R](../scripts/bmjgh_figures/figureS3.R) | [A](../source_data/figures/figureS3_A.csv), [B](../source_data/figures/figureS3_B.csv), [C](../source_data/figures/figureS3_C.csv), [D](../source_data/figures/figureS3_D.csv), [E](../source_data/figures/figureS3_E.csv), [F](../source_data/figures/figureS3_F.csv), [G](../source_data/figures/figureS3_G.csv), [H](../source_data/figures/figureS3_H.csv) |
| S4 | Changes in reported cases at different ages after expansion of a standard programme | [R](../scripts/bmjgh_figures/figureS4.R) | [A](../source_data/figures/figureS4_A.csv) |
| S5 | Annual and cumulative effects depend on the epidemic trajectory | [R](../scripts/bmjgh_figures/figureS5.R) | [CSV](../source_data/figures/figureS5.csv) |
| S6 | Annual incidence and cumulative effects of nasal vaccination | [R](../scripts/bmjgh_figures/figureS6.R) | [CSV](../source_data/figures/figureS6.csv) |
| S7 | Cumulative case effects across programme start years | [R](../scripts/bmjgh_figures/figureS7.R) | [CSV](../source_data/figures/figureS7.csv) |
| S8 | Additional burden averted in the third decade and annual infant mortality in Brazil and Japan | [R](../scripts/bmjgh_figures/figureS8.R) | [A](../source_data/figures/figureS8_A.csv), [B](../source_data/figures/figureS8_B.csv), [C](../source_data/figures/figureS8_C.csv) |
| S9 | Infant deaths averted by each adolescent intranasal vaccine profile | [R](../scripts/bmjgh_figures/figureS9.R) | [A](../source_data/figures/figureS9_A.csv) |
| S10 | Discounted infant years of life lost and concordance of infant outcomes across vaccine profiles | [R](../scripts/bmjgh_figures/figureS10.R) | [A](../source_data/figures/figureS10_A.csv), [B](../source_data/figures/figureS10_B.csv) |
| S11 | Matching adolescent case reductions under two fitted recovery histories | [R](../scripts/bmjgh_figures/figureS11.R) | [A](../source_data/figures/figureS11_A.csv) |
| S12 | Alternative matches for adolescent cases in Japan at 30 years | [R](../scripts/bmjgh_figures/figureS12.R) | [A](../source_data/figures/figureS12_A.csv), [B](../source_data/figures/figureS12_B.csv) |
| S13 | Local transmission uncertainty at fixed adolescent uptake in England and Brazil | [R](../scripts/bmjgh_figures/figureS13.R) | [draws](../source_data/figures/S13_draws.csv), [B](../source_data/figures/figureS13_B.csv) |
| S14 | Maternal coverage and the components of marginal nasal benefit in China | [R](../scripts/bmjgh_figures/figureS14.R) | [CSV](../source_data/figures/figureS14.csv) |
| S15 | Infant deaths averted by expanding maternal vaccination | [R](../scripts/bmjgh_figures/figureS15.R) | [A](../source_data/figures/figureS15_A.csv), [B](../source_data/figures/figureS15_B.csv) |
| S16 | Delivered-cost ceilings and conditional intervals at every sourced valuation | [R](../scripts/bmjgh_figures/figureS16.R) | [A](../source_data/figures/figureS16_A.csv), [B](../source_data/figures/figureS16_B.csv) |
| S17 | Sensitivity of delivered-cost ceilings to the ages included in mortality estimates | [R](../scripts/bmjgh_figures/figureS17.R) | [A](../source_data/figures/figureS17_A.csv), [B](../source_data/figures/figureS17_B.csv) |
| S18 | Observed and predicted pertussis cases during fitting and available non-training periods | [R](../scripts/bmjgh_figures/figureS18.R) | [CSV](../source_data/figures/figureS18.csv) |
| S19 | Historical monthly cases at two fitting cutoffs | [R](../scripts/bmjgh_figures/figureS19.R) | [CSV](../source_data/figures/figureS19.csv) |
| S20 | Agreement between fitted and observed surveillance counts | [R](../scripts/bmjgh_figures/figureS20.R) | [A](../source_data/figures/figureS20_A.csv) |
| S21 | Case totals and peak timing in later surveillance | [R](../scripts/bmjgh_figures/figureS21.R) | [A](../source_data/figures/figureS21_A.csv), [B](../source_data/figures/figureS21_B.csv) |
| S22 | Protection duration and allocation interact at fixed programme coverage | [R](../scripts/bmjgh_figures/figureS22.R) | [CSV](../source_data/figures/figureS22.csv) |
| S23 | Morbidity and price conditions supporting programme expansion | [R](../scripts/bmjgh_figures/figureS23.R) | [A](../source_data/figures/figureS23_A.csv), [B](../source_data/figures/figureS23_B.csv) |
| S24 | Route-optimal expected NMB in all eight settings over 20 and 30 years | [R](../scripts/bmjgh_figures/figureS24.R) | [CSV](../source_data/figures/figureS24.csv) |
| S25 | Protection duration and nasal price boundaries over two evaluation windows | [R](../scripts/bmjgh_figures/figureS25.R) | [CSV](../source_data/figures/figureS25.csv) |

## Tables

The [table index](../source_data/tables/index.csv) gives titles and row counts for tables 1–2 and S1–S84. The table number identifies the CSV, for example [Table 2](../source_data/tables/table2.csv) and [Table S39](../source_data/tables/tableS39.csv). These files retain the values and precision displayed in the article and supplement. The [analysis file index](../source_data/tables/analysis_sources.csv) maps tables to additional machine-readable estimates, fit summaries or inputs. The [reference list](../source_data/tables/references.csv) resolves citation keys in table cells. The [dose productivity files](../source_data/tables/dose_productivity) correspond to the supplementary workbook sheets.

Tables S14–S16 give aggregate summaries derived from 45 de-identified adult trial participants. The original participant-level AUC measurements are available from the authors of the trial in their published supplementary workbook. [Hill et al.](https://doi.org/10.1038/s41541-026-01443-7)

To draw the model schematic accompanying Table S8, run

```sh
Rscript scripts/bmjgh_figures/tableS8_schematic.R
```
