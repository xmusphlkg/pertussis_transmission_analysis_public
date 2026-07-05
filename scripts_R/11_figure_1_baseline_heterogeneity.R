#!/usr/bin/env Rscript
## Figure 1: age-structured control problem -----------------------------------
## Layout: (a) Post-pandemic surveillance context
##         (b) Temporal calibration diagnostics
##         (c) Current-practice paediatric age composition
##         (d) Baseline modelled paediatric burden by age stratum

args <- commandArgs(FALSE)
file_arg <- sub("^--file=", "", args[grepl("^--file=", args)])
script_dir <- if (length(file_arg) > 0) dirname(normalizePath(file_arg[[1]])) else file.path(getwd(), "scripts_R")
source(file.path(script_dir, "10_shared.R"))

suppressPackageStartupMessages({
  library(ggrepel)
})

required_columns <- function(df, cols, dataset_name) {
  missing_cols <- setdiff(cols, names(df))
  if (length(missing_cols) > 0) {
    stop(
      dataset_name, " is missing required column(s): ",
      paste(missing_cols, collapse = ", "),
      call. = FALSE
    )
  }
  invisible(df)
}

positive_rate <- function(x) {
  is.finite(x) & !is.na(x) & x > 0
}

read_table <- function(name) {
  readr::read_csv(model_path("outputs", "tables", name), show_col_types = FALSE)
}

## Panel A: WHO regional reported-incidence context ---------------------------

selected_regions <- c(
  "Western Pacific Region", "European Region",
  "Region of the Americas", "African Region", "South-East Asia Region"
)

regional_incidence <- readr::read_csv(
  model_path("data", "processed", "who_pertussis_region_incidence.csv"),
  show_col_types = FALSE
) %>%
  required_columns(
    c("region", "year", "reported_incidence_per_million"),
    "WHO regional incidence input"
  ) %>%
  mutate(reported_incidence_per_100k = reported_incidence_per_million / 10)

regional_focus <- regional_incidence %>%
  filter(region == "Global" | region %in% selected_regions) %>%
  mutate(region = factor(region, levels = c("Global", selected_regions)))

region_short <- c(
  "Global" = "Global",
  "Western Pacific Region" = "WPR",
  "European Region" = "EUR",
  "Region of the Americas" = "AMR",
  "African Region" = "AFR",
  "South-East Asia Region" = "SEAR"
)

regional_focus_latest <- regional_focus %>%
  group_by(region) %>%
  filter(positive_rate(reported_incidence_per_100k)) %>%
  slice_max(year, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  mutate(region_label = region_short[as.character(region)])

p1a <- ggplot(
  regional_focus %>% filter(positive_rate(reported_incidence_per_100k)),
  aes(year, reported_incidence_per_100k, colour = region)
) +
  geom_line(linewidth = 0.58, alpha = 0.92) +
  geom_point(data = regional_focus_latest, size = 1.15) +
  ggrepel::geom_text_repel(
    data = regional_focus_latest,
    aes(label = region_label),
    size = lancet_direct_label_size,
    fontface = "bold",
    nudge_x = 0.55,
    min.segment.length = 0,
    segment.size = 0.15,
    direction = "y",
    hjust = 0,
    show.legend = FALSE
  ) +
  scale_colour_manual(values = region_colours, guide = "none") +
  scale_x_continuous(breaks = seq(2000, 2024, by = 4), expand = expansion(mult = c(0.01, 0.14))) +
  scale_y_log10(breaks = c(0.3, 1, 3, 10, 30), labels = label_lancet_number(accuracy = 0.1)) +
  coord_cartesian(xlim = c(2000, 2025), ylim = c(0.25, 50)) +
  labs(x = "Year", y = "Reported incidence per 100 000/year (log)", tag = "a") +
  theme_lancet(base_size = journal_base_text_size)

## Shared profile order --------------------------------------------------------

pediatric_burden <- read_table("lancet_baseline_pediatric_burden.csv") %>%
  required_columns(
    c(
      "country", "primary_cases_per_100k", "infant_cases_per_100k",
      "child_1_9_cases_per_100k", "adolescent_cases_per_100k",
      "age_case_data_availability", "lancet_endpoint_validation_tier"
    ),
    "Lancet baseline pediatric burden table"
  ) %>%
  mutate(country = stringr::str_replace_all(country, " ", "_"))

country_order <- pediatric_burden %>%
  arrange(desc(primary_cases_per_100k)) %>%
  mutate(country_label_text = format_country(country)) %>%
  pull(country_label_text)

country_who_region <- c(
  Australia = "Western Pacific Region",
  China = "Western Pacific Region",
  Japan = "Western Pacific Region",
  New_Zealand = "Western Pacific Region",
  Sweden = "European Region",
  United_Kingdom = "European Region",
  United_States = "Region of the Americas",
  Brazil = "Region of the Americas",
  Thailand = "South-East Asia Region",
  South_Africa = "African Region"
)

age_stratum_colours <- c(
  "0–11 m" = palette_discrete_primary_9[[7]],
  "1–9 y" = palette_discrete_primary_9[[4]],
  "10–17 y" = palette_discrete_primary_9[[8]]
)

## Panel B: temporal calibration diagnostics ----------------------------------

calibration_fit <- read_table("calibration_all_countries.csv") %>%
  required_columns(
    c(
      "country", "observed_mean_annual_reported_incidence_per_100k",
      "annualized_reported_cases_per_100k", "model_to_observed_reported_incidence_ratio",
      "calibration_interval_smape", "calibration_peak_ratio", "calibration_data_overlap_intervals",
      "calibration_log1p_correlation", "calibration_max_interval_smape",
      "calibration_peak_ratio_min", "calibration_peak_ratio_max", "total_population"
    ),
    "Calibration all-countries table"
  ) %>%
  mutate(
    country = stringr::str_replace_all(country, " ", "_"),
    country_label_text = format_country(country),
    country_code = unname(country_codes[country]),
    who_region = unname(country_who_region[country]),
    who_region = if_else(is.na(who_region), "Other", who_region),
    observed_reported_incidence_per_100k = as.numeric(observed_mean_annual_reported_incidence_per_100k),
    modelled_reported_incidence_per_100k = as.numeric(annualized_reported_cases_per_100k),
    model_observed_ratio = as.numeric(model_to_observed_reported_incidence_ratio),
    calibration_interval_smape = as.numeric(calibration_interval_smape),
    calibration_peak_ratio = as.numeric(calibration_peak_ratio),
    calibration_log1p_correlation = as.numeric(calibration_log1p_correlation),
    calibration_max_interval_smape = as.numeric(calibration_max_interval_smape),
    calibration_peak_ratio_min = as.numeric(calibration_peak_ratio_min),
    calibration_peak_ratio_max = as.numeric(calibration_peak_ratio_max),
    calibration_data_overlap_intervals = as.numeric(calibration_data_overlap_intervals),
    total_population = as.numeric(total_population),
    population_millions = total_population / 1e6
  ) %>%
  filter(
    positive_rate(observed_reported_incidence_per_100k),
    positive_rate(modelled_reported_incidence_per_100k),
    is.finite(calibration_interval_smape),
    positive_rate(calibration_peak_ratio),
    positive_rate(total_population)
  ) %>%
  distinct(country, .keep_all = TRUE)

calibration_thresholds <- calibration_fit %>%
  summarise(
    max_interval_smape = first(na.omit(calibration_max_interval_smape)),
    peak_ratio_min = first(na.omit(calibration_peak_ratio_min)),
    peak_ratio_max = first(na.omit(calibration_peak_ratio_max))
  )

if (nrow(calibration_thresholds) == 0 ||
    !is.finite(calibration_thresholds$max_interval_smape[[1]]) ||
    !is.finite(calibration_thresholds$peak_ratio_min[[1]]) ||
    !is.finite(calibration_thresholds$peak_ratio_max[[1]])) {
  stop("Calibration thresholds could not be recovered for Figure 1B.", call. = FALSE)
}

readr::write_csv(
  calibration_fit %>%
    transmute(
      country,
      country_code,
      who_region,
      population_millions,
      observed_reported_incidence_per_100k,
      modelled_reported_incidence_per_100k,
      model_observed_ratio,
      calibration_interval_smape,
      calibration_peak_ratio,
      calibration_log1p_correlation,
      calibration_max_interval_smape,
      calibration_peak_ratio_min,
      calibration_peak_ratio_max,
      calibration_data_overlap_intervals
    ),
  model_path("outputs", "tables", "figure1b_model_observed_calibration.csv")
)

p1b <- ggplot(
  calibration_fit,
  aes(calibration_interval_smape, calibration_peak_ratio)
) +
  annotate(
    "rect",
    xmin = 0.5,
    xmax = calibration_thresholds$max_interval_smape[[1]],
    ymin = calibration_thresholds$peak_ratio_min[[1]],
    ymax = calibration_thresholds$peak_ratio_max[[1]],
    fill = manuscript_colour("pale_grey"),
    alpha = 0.45
  ) +
  geom_vline(
    xintercept = calibration_thresholds$max_interval_smape[[1]],
    linetype = "dashed",
    linewidth = 0.26,
    colour = manuscript_colour("mid_grey")
  ) +
  geom_hline(
    yintercept = 1,
    linetype = "dashed",
    linewidth = 0.26,
    colour = manuscript_colour("mid_grey")
  ) +
  geom_point(
    aes(fill = who_region, size = total_population),
    shape = 21,
    colour = manuscript_colour("black"),
    alpha = 0.86,
    stroke = 0.22,
    show.legend = TRUE
  ) +
  ggrepel::geom_text_repel(
    aes(label = country_code),
    size = journal_point_label_text_size,
    fontface = "bold",
    min.segment.length = 0,
    segment.size = 0.10,
    max.overlaps = Inf,
    show.legend = FALSE
  ) +
  scale_fill_manual(values = region_colours, guide = "none") +
  scale_size_continuous(
    name = "Population",
    breaks = c(1e7, 1e8, 1e9),
    labels = c("10M", "100M", "1B"),
    range = c(3, 10),
    guide = guide_legend(
      nrow = 1,
      byrow = TRUE,
      title.position = "left",
      override.aes = list(fill = manuscript_colour("light_grey"), alpha = 0.86)
    )
  ) +
  scale_x_continuous(
    breaks = c(0.5, 0.75, 1.0, 1.25, 1.5),
    labels = label_lancet_number(accuracy = 0.01),
    expand = expansion(mult = c(0.03, 0.08))
  ) +
  scale_y_log10(
    breaks = c(0.25, 0.5, 1, 2, 4),
    labels = label_lancet_number(accuracy = 0.01),
    expand = expansion(mult = c(0.06, 0.06))
  ) +
  coord_cartesian(xlim = c(0.5, 1.36), ylim = c(0.25, 4), clip = "on") +
  labs(
    x = "Interval SMAPE",
    y = "Model/observed peak reported cases (log)",
    tag = "b"
  ) +
  theme_lancet(base_size = journal_compact_text_size) +
  theme(
    legend.position = "inside",
    legend.position.inside = c(0.52, 0.98),
    legend.justification = c(0.5, 1),
    legend.title.position = 'left',
    legend.background = element_rect(fill = "#FFFFFFE8", colour = NA),
    legend.key.width = unit(0.28, "cm"),
    legend.margin = margin(0, 0, 0, 0),
    plot.margin = margin(5, 8, 4, 4),
    panel.grid.major.x = element_line(linewidth = 0.16, colour = lancet_grid_light_colour)
  )

## Panel C: baseline paediatric burden composition -----------------------------

baseline_composition <- readr::read_csv(
  model_path("outputs", "summaries", "intervention_scenarios_summary.csv"),
  show_col_types = FALSE
) %>%
  required_columns(
    c(
      "country", "scenario", "total_infant_cases", "total_child_1_9_cases",
      "total_adolescent_cases", "total_child_adolescent_cases"
    ),
    "Intervention scenario summary"
  ) %>%
  filter(scenario == "current") %>%
  mutate(
    country = stringr::str_replace_all(country, " ", "_"),
    country_label_text = format_country(country),
    country_label = factor(country_label_text, levels = rev(country_order)),
    country_code = unname(country_codes[country]),
    total_infant_cases = as.numeric(total_infant_cases),
    total_child_1_9_cases = as.numeric(total_child_1_9_cases),
    total_adolescent_cases = as.numeric(total_adolescent_cases),
    total_child_adolescent_cases = as.numeric(total_child_adolescent_cases)
  ) %>%
  filter(!is.na(country_label), positive_rate(total_child_adolescent_cases)) %>%
  distinct(country, .keep_all = TRUE) %>%
  select(
    country, country_code, country_label,
    total_infant_cases, total_child_1_9_cases, total_adolescent_cases,
    total_child_adolescent_cases
  ) %>%
  pivot_longer(
    c(total_infant_cases, total_child_1_9_cases, total_adolescent_cases),
    names_to = "age_stratum",
    values_to = "symptomatic_cases"
  ) %>%
  mutate(
    age_stratum = factor(
      age_stratum,
      levels = c("total_infant_cases", "total_child_1_9_cases", "total_adolescent_cases"),
      labels = c("0–11 m", "1–9 y", "10–17 y")
    ),
    burden_share = symptomatic_cases / total_child_adolescent_cases
  )

composition_wide <- baseline_composition %>%
  select(country, country_label, age_stratum, burden_share) %>%
  pivot_wider(names_from = age_stratum, values_from = burden_share) %>%
  mutate(
    infant_child_boundary = `0–11 m`,
    child_adolescent_boundary = `0–11 m` + `1–9 y`
  )

composition_boundaries <- composition_wide %>%
  select(country, country_label, infant_child_boundary, child_adolescent_boundary) %>%
  pivot_longer(
    c(infant_child_boundary, child_adolescent_boundary),
    names_to = "boundary_type",
    values_to = "boundary_share"
  ) %>%
  arrange(boundary_type, country_label)

readr::write_csv(
  baseline_composition %>%
    transmute(
      country,
      country_code,
      age_stratum = as.character(age_stratum),
      symptomatic_cases,
      total_child_adolescent_cases,
      burden_share
    ),
  model_path("outputs", "tables", "figure1c_baseline_age_composition.csv")
)

p1c <- ggplot(baseline_composition, aes(burden_share, country_label, fill = age_stratum)) +
  geom_col(width = 0.64, colour = "white", linewidth = 0.18, position = position_stack(reverse = TRUE)) +
  scale_x_continuous(
    labels = label_lancet_percent(accuracy = 1),
    breaks = seq(0, 1, by = 0.25),
    expand = expansion(mult = c(0, 0.01))
  ) +
  scale_fill_manual(values = age_stratum_colours) +
  coord_cartesian(xlim = c(0, 1), clip = "off") +
  labs(
    x = "Share of modelled symptomatic cases\namong ages <18 years",
    y = NULL,
    fill = "Age group",
    tag = "c"
  ) +
  theme_lancet(base_size = journal_compact_text_size) +
     theme(
          legend.position = "top",
          legend.direction = "horizontal",
          legend.background = element_rect(fill = "#FFFFFFE8", colour = NA),
          legend.key.width = unit(0.38, "cm"),
          axis.ticks.y = element_blank()
     )

## Panel D: baseline modelled burden by paediatric age stratum -----------------

burden_long <- pediatric_burden %>%
  mutate(
    country_label_text = format_country(country),
    country_label = factor(country_label_text, levels = rev(country_order))
  ) %>%
  select(country, country_label, infant_cases_per_100k, child_1_9_cases_per_100k, adolescent_cases_per_100k) %>%
  pivot_longer(-c(country, country_label), names_to = "age_stratum", values_to = "cases_per_100k") %>%
  mutate(
    age_stratum = factor(
      age_stratum,
      levels = c("infant_cases_per_100k", "child_1_9_cases_per_100k", "adolescent_cases_per_100k"),
      labels = c("0–11 m", "1–9 y", "10–17 y")
    )
  ) %>%
  filter(positive_rate(cases_per_100k))

readr::write_csv(
  burden_long %>%
    transmute(
      country,
      age_stratum = as.character(age_stratum),
      cases_per_100k
    ),
  model_path("outputs", "tables", "figure1d_baseline_age_burden.csv")
)

p1d <- ggplot(burden_long, aes(cases_per_100k, country_label)) +
  geom_line(aes(group = country_label), colour = lancet_grid_colour, linewidth = 0.34) +
  geom_point(aes(colour = age_stratum, shape = age_stratum), size = 1.85, alpha = 0.90) +
  scale_x_log10(
    breaks = c(50, 100, 300, 1000, 3000),
    labels = label_lancet_comma(accuracy = 1),
    expand = expansion(mult = c(0.03, 0.08))
  ) +
  scale_colour_manual(
    values = age_stratum_colours,
    name = "Age group",
    guide = guide_legend(nrow = 1, byrow = TRUE, title.position = "left")
  ) +
  scale_shape_manual(
    values = c(16, 17, 15),
    name = "Age group",
    guide = guide_legend(nrow = 1, byrow = TRUE, title.position = "left")
  ) +
  labs(
    x = "Annualised symptomatic cases\nper 100 000/year, 2025–50 (log)",
    y = NULL,
    tag = "d"
  ) +
  theme_lancet(base_size = journal_compact_text_size) +
  theme(
    legend.position = "top",
    legend.direction = "horizontal",
    legend.background = element_rect(fill = "#FFFFFFE8", colour = NA),
    legend.key.width = unit(0.38, "cm"),
    axis.ticks.y = element_blank()
  )

## Compose Figure 1 ------------------------------------------------------------

figure1 <- free(p1a) + free(p1b) + p1c + p1d +
  plot_layout(widths = c(0.98, 1.02)) &
  theme_lancet_tags()

save_main_figure(figure1, "figure_1_baseline_heterogeneity", height = 6.2)
cat("Figure 1 saved.\n")
