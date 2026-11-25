# Data sources and reuse

| Input | Source and material used |
| --- | --- |
| Pertussis notifications | The eight fitted specifications identify the processed, setting-specific case series, observation windows and age totals. See the supplementary source tables for definitions and attribution. [Table index](../source_data/tables/index.csv) |
| Immunisation programmes | National schedule and uptake sources are recorded with the programme parameters. Childhood coverage inputs include WHO/UNICEF WUENIC estimates for DTP1 and DTP3. [WHO Immunization Data](https://immunizationdata.who.int/) |
| Demography | Population, births and life expectancy use selected United Nations World Population Prospects 2024 series. England-specific population and birth data are from the Office for National Statistics. [UN WPP](https://population.un.org/wpp/) · [ONS](https://www.ons.gov.uk/) |
| Vaccination in England | The population model uses the UKHSA prenatal vaccination series and COVER childhood coverage statistics. [UKHSA vaccine uptake](https://www.gov.uk/government/collections/vaccine-uptake) |
| Contact patterns | The model uses Prem and colleagues’ projected social contact matrices with their reference populations, distributed in the contactdata R package. The contact-reduction input gives the sources and periods for the non-household adjustments. [Contact matrices](https://doi.org/10.1371/journal.pcbi.1009098) · [contactdata](https://cran.r-project.org/package=contactdata) |
| Clinical AUC | Hill and colleagues’ public supplementary workbook contains the de-identified trial observations. The model uses aggregate estimates derived from 24 BPZE1 and 21 placebo recipients. The study release includes the extraction code and group summaries. [Hill et al.](https://doi.org/10.1038/s41541-026-01443-7) · [Original workbook](https://static-content.springer.com/esm/art%3A10.1038%2Fs41541-026-01443-7/MediaObjects/41541_2026_1443_MOESM2_ESM.xlsx) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Clinical risk, costs and valuations | The economic configuration files and supplementary tables give the published sources, assumptions, currency conversions and setting assignments. The paired draws represent clinical and cost uncertainty. [Table index](../source_data/tables/index.csv) |

The authors generated the projections, paired comparisons, figure data and analysis software. The MIT licence covers the study software. Reused numerical data and the source workbook retain their original terms. [Software licence](../LICENSE)
