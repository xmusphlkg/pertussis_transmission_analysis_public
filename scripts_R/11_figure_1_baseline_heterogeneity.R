#!/usr/bin/env Rscript
## Figure 1: age-structured control problem -----------------------------------
## Layout: (a) Post-pandemic surveillance context
##         (b) Current-practice paediatric burden
##         (c) Current-practice paediatric age composition
##         (d) Baseline estimated reported cases and infections by age group

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
      "infant_hospitalizations_per_100k", "infant_deaths_per_100k",
      "child_1_9_cases_per_100k", "adolescent_cases_per_100k",
      "child_adolescent_reported_cases_per_100k", "child_adolescent_infections_per_100k",
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

age_group_colours <- c(
  "Infant" = palette_discrete_primary_9[[7]],
  "Children" = palette_discrete_primary_9[[4]],
  "Adolescent" = palette_discrete_primary_9[[8]]
)

age_group_display_labels <- c(
  "Infant" = "Infants <1 y",
  "Children" = "Children 1-9 y",
  "Adolescent" = "Adolescents 10-17 y"
)

## Panel B: calibration fingerprint -------------------------------------------

calibration_fit <- read_table("calibration_all_countries.csv") %>%
  required_columns(
    c(
      "country", "observed_mean_annual_reported_incidence_per_100k",
      "annualized_reported_cases_per_100k", "model_to_observed_reported_incidence_ratio",
      "calibration_interval_smape", "calibration_peak_ratio", "calibration_data_overlap_intervals",
      "calibration_log1p_correlation", "calibration_max_interval_smape",
      "calibration_peak_ratio_min", "calibration_peak_ratio_max", "total_population",
      "absolute_fit_relative_tolerance"
    ),
    "Calibration all-countries table"
  ) %>%
  mutate(
    country = stringr::str_replace_all(country, " ", "_"),
    country_label_text = format_country(country),
    country_label_factor = factor(country_label_text, levels = rev(country_order)),
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
    absolute_fit_relative_tolerance = as.numeric(absolute_fit_relative_tolerance),
    total_population = as.numeric(total_population),
    population_millions = total_population / 1e6,
    mean_ratio_min = 1 - absolute_fit_relative_tolerance,
    mean_ratio_max = 1 + absolute_fit_relative_tolerance
  ) %>%
  filter(
    !is.na(country_label_factor),
    positive_rate(observed_reported_incidence_per_100k),
    positive_rate(modelled_reported_incidence_per_100k),
    positive_rate(model_observed_ratio),
    is.finite(calibration_interval_smape),
    positive_rate(calibration_peak_ratio),
    positive_rate(total_population)
  ) %>%
  distinct(country, .keep_all = TRUE)

calibration_thresholds <- calibration_fit %>%
  summarise(
    mean_ratio_min = first(na.omit(mean_ratio_min)),
    mean_ratio_max = first(na.omit(mean_ratio_max)),
    max_interval_smape = first(na.omit(calibration_max_interval_smape)),
    peak_ratio_min = first(na.omit(calibration_peak_ratio_min)),
    peak_ratio_max = first(na.omit(calibration_peak_ratio_max))
  )

if (nrow(calibration_thresholds) == 0 ||
    !is.finite(calibration_thresholds$mean_ratio_min[[1]]) ||
    !is.finite(calibration_thresholds$mean_ratio_max[[1]]) ||
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
      estimated_reported_incidence_per_100k = modelled_reported_incidence_per_100k,
      estimated_observed_ratio = model_observed_ratio,
      calibration_interval_smape,
      calibration_peak_ratio,
      calibration_log1p_correlation,
      absolute_fit_relative_tolerance,
      calibration_max_interval_smape,
      calibration_peak_ratio_min,
      calibration_peak_ratio_max,
      calibration_data_overlap_intervals
    ),
  model_path("outputs", "tables", "figure1b_model_observed_calibration.csv")
)

theme_calibration_fingerprint <- theme_lancet_panel(
  base_size = journal_compact_text_size,
  plot_margin = margin(5, 4, 4, 4),
  show_x_grid = TRUE,
  show_y_grid = TRUE
) +
  theme(
    axis.ticks.y = element_blank(),
    legend.position = "none",
    plot.title = element_text(
      family = lancet_font_family,
      face = "bold",
      size = journal_compact_text_size - 0.5,
      hjust = 0.5,
      margin = margin(b = 3)
    ),
    panel.grid.major.x = element_line(linewidth = 0.16, colour = lancet_grid_light_colour)
  )

p1b_mean <- ggplot(calibration_fit, aes(model_observed_ratio, country_label_factor)) +
  annotate(
    "rect",
    xmin = calibration_thresholds$mean_ratio_min[[1]],
    xmax = calibration_thresholds$mean_ratio_max[[1]],
    ymin = -Inf,
    ymax = Inf,
    fill = manuscript_colour("pale_grey"),
    alpha = 0.38
  ) +
  geom_vline(xintercept = 1, linewidth = 0.26, linetype = "dashed", colour = manuscript_colour("black")) +
  geom_point(aes(fill = who_region), shape = 21, size = 2.10, colour = manuscript_colour("black"), stroke = 0.22, alpha = 0.88) +
  scale_fill_manual(values = c(region_colours, "Other" = manuscript_colour("mid_grey")), guide = "none") +
  scale_x_continuous(
    breaks = c(0.75, 1.0, 1.25),
    labels = label_lancet_number(accuracy = 0.01),
    expand = expansion(mult = c(0.03, 0.03))
  ) +
  coord_cartesian(xlim = c(0.75, 1.25)) +
  labs(x = "Mean ratio", y = NULL, tag = "b") +
  theme_calibration_fingerprint

p1b_smape <- ggplot(calibration_fit, aes(calibration_interval_smape, country_label_factor)) +
  geom_vline(
    xintercept = calibration_thresholds$max_interval_smape[[1]],
    linewidth = 0.26,
    linetype = "dashed",
    colour = manuscript_colour("mid_grey")
  ) +
  geom_point(aes(fill = who_region), shape = 21, size = 2.10, colour = manuscript_colour("black"), stroke = 0.22, alpha = 0.88) +
  scale_fill_manual(values = c(region_colours, "Other" = manuscript_colour("mid_grey")), guide = "none") +
  scale_x_continuous(
    breaks = c(0.5, 1.0, 1.3),
    labels = label_lancet_number(accuracy = 0.01),
    expand = expansion(mult = c(0.05, 0.07))
  ) +
  coord_cartesian(xlim = c(0.5, 1.35)) +
  labs(x = "Interval sMAPE", y = NULL) +
  theme_calibration_fingerprint +
  theme(axis.text.y = element_blank())

p1b_peak <- ggplot(calibration_fit, aes(calibration_peak_ratio, country_label_factor)) +
  annotate(
    "rect",
    xmin = calibration_thresholds$peak_ratio_min[[1]],
    xmax = calibration_thresholds$peak_ratio_max[[1]],
    ymin = -Inf,
    ymax = Inf,
    fill = manuscript_colour("pale_grey"),
    alpha = 0.38
  ) +
  geom_vline(xintercept = 1, linewidth = 0.26, linetype = "dashed", colour = manuscript_colour("black")) +
  geom_point(aes(fill = who_region), shape = 21, size = 2.10, colour = manuscript_colour("black"), stroke = 0.22, alpha = 0.88) +
  scale_fill_manual(values = c(region_colours, "Other" = manuscript_colour("mid_grey")), guide = "none") +
  scale_x_log10(
    breaks = c(0.25, 0.5, 1, 2, 4),
    labels = c("0·25", "0·5", "1", "2", "4"),
    expand = expansion(mult = c(0.04, 0.05))
  ) +
  coord_cartesian(xlim = c(0.25, 4)) +
  labs(x = "Peak ratio (log)", y = NULL) +
  theme_calibration_fingerprint +
  theme(axis.text.y = element_blank())

p1b <- p1b_mean + p1b_smape + p1b_peak +
  plot_layout(widths = c(1.05, 0.95, 1.00))

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
    names_to = "age_group",
    values_to = "symptomatic_cases"
  ) %>%
  mutate(
    age_group = factor(
      age_group,
      levels = c("total_infant_cases", "total_child_1_9_cases", "total_adolescent_cases"),
      labels = c("Infant", "Children", "Adolescent")
    ),
    burden_share = symptomatic_cases / total_child_adolescent_cases
  )

composition_wide <- baseline_composition %>%
  select(country, country_label, age_group, burden_share) %>%
  pivot_wider(names_from = age_group, values_from = burden_share) %>%
  mutate(
    infant_child_boundary = Infant,
    child_adolescent_boundary = Infant + Children
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
      age_group = as.character(age_group),
      symptomatic_cases,
      total_child_adolescent_cases,
      burden_share
    ),
  model_path("outputs", "tables", "figure1c_baseline_age_composition.csv")
)

p1c <- ggplot(baseline_composition, aes(burden_share, country_label, fill = age_group)) +
  geom_col(width = 0.64, colour = "white", linewidth = 0.18, position = position_stack(reverse = TRUE)) +
  scale_x_continuous(
    labels = label_lancet_percent(accuracy = 1),
    breaks = seq(0, 1, by = 0.25),
    expand = expansion(mult = c(0, 0.01))
  ) +
  scale_fill_manual(values = age_group_colours, labels = age_group_display_labels) +
  coord_cartesian(xlim = c(0, 1), clip = "off") +
  labs(
    x = "Share of symptomatic cases\namong people aged <18 years",
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

## Panel D: baseline estimated reported cases and infections ------------------

simulation_path <- model_path("outputs", "simulations", "intervention_scenarios.parquet")
if (!file.exists(simulation_path)) {
  stop("Missing simulation time series for Figure 1D: ", simulation_path, call. = FALSE)
}
if (!requireNamespace("arrow", quietly = TRUE)) {
  stop("The arrow package is required to read simulation time series for Figure 1D.", call. = FALSE)
}

current_timeseries <- arrow::read_parquet(
  simulation_path,
  col_select = c("country", "scenario", "age_group", "time", "population", "reported_cases", "total_infections")
) %>%
  as_tibble() %>%
  required_columns(
    c("country", "scenario", "age_group", "time", "population", "reported_cases", "total_infections"),
    "Intervention scenario time series"
  ) %>%
  filter(scenario == "current") %>%
  mutate(country = stringr::str_replace_all(country, " ", "_"))

burden_age_map <- tibble::tribble(
  ~burden_group, ~age_group,
  "Infant", "infant_0_2m",
  "Infant", "infant_3_11m",
  "Children", "child_1_4y",
  "Children", "child_5_9y",
  "Adolescent", "adolescent_10_17y"
) %>%
  mutate(burden_group = factor(burden_group, levels = c("Infant", "Children", "Adolescent")))

analysis_years <- current_timeseries %>%
  group_by(country) %>%
  summarise(analysis_years = pmax((max(time, na.rm = TRUE) - min(time, na.rm = TRUE)) / 365, 1 / 365), .groups = "drop")

burden_population <- current_timeseries %>%
  inner_join(burden_age_map, by = "age_group", relationship = "many-to-many") %>%
  group_by(country, burden_group, age_group) %>%
  summarise(age_population = mean(as.numeric(population), na.rm = TRUE), .groups = "drop") %>%
  group_by(country, burden_group) %>%
  summarise(population = sum(age_population, na.rm = TRUE), .groups = "drop")

burden_events <- current_timeseries %>%
  inner_join(burden_age_map, by = "age_group", relationship = "many-to-many") %>%
  group_by(country, burden_group) %>%
  summarise(
    reported_cases = sum(as.numeric(reported_cases), na.rm = TRUE),
    estimated_infections = sum(as.numeric(total_infections), na.rm = TRUE),
    .groups = "drop"
  )

burden_long <- burden_events %>%
  left_join(burden_population, by = c("country", "burden_group")) %>%
  left_join(analysis_years, by = "country") %>%
  mutate(
    reported_cases_per_100k = reported_cases / pmax(population * analysis_years, 1e-9) * 100000,
    estimated_infections_per_100k = estimated_infections / pmax(population * analysis_years, 1e-9) * 100000
  ) %>%
  select(
    country, burden_group, population, analysis_years,
    reported_cases, estimated_infections, reported_cases_per_100k, estimated_infections_per_100k
  ) %>%
  pivot_longer(
    cols = c(reported_cases_per_100k, estimated_infections_per_100k),
    names_to = "outcome",
    values_to = "rate_per_100k"
  ) %>%
  mutate(
    outcome = factor(
      outcome,
      levels = c("reported_cases_per_100k", "estimated_infections_per_100k"),
      labels = c("Estimated reports", "Estimated infections")
    ),
    burden_group = factor(as.character(burden_group), levels = c("Infant", "Children", "Adolescent")),
    country_label_text = format_country(country),
    country_y = as.numeric(factor(country_label_text, levels = rev(country_order))) +
      if_else(outcome == "Estimated reports", -0.12, 0.12)
  ) %>%
  filter(positive_rate(rate_per_100k))

readr::write_csv(
  burden_long %>%
    transmute(
      country,
      burden_group = as.character(burden_group),
      outcome = as.character(outcome),
      population,
      analysis_years,
      reported_cases,
      estimated_infections,
      rate_per_100k
    ),
  model_path("outputs", "tables", "figure1d_baseline_age_burden.csv")
)

baseline_burden_shapes <- c(
  "Estimated reports" = 16,
  "Estimated infections" = 17
)

p1d_x_limits <- c(
  4,
  max(burden_long$rate_per_100k, na.rm = TRUE) * 1.24
)

p1d <- ggplot() +
  geom_point(
    data = burden_long,
    aes(rate_per_100k, country_y, colour = burden_group, shape = outcome),
    size = 1.75,
    alpha = 0.55,
    stroke = 0.48
  ) +
  scale_x_log10(
    breaks = c(5, 10, 30, 100, 300, 1000, 3000),
    labels = label_lancet_comma(accuracy = 1),
    expand = expansion(mult = c(0.04, 0.12))
  ) +
  scale_y_continuous(
    breaks = seq_along(rev(country_order)),
    labels = rev(country_order),
    expand = expansion(add = c(0.35, 0.35))
  ) +
  scale_colour_manual(
    values = age_group_colours,
    labels = age_group_display_labels,
    guide = "none"
  ) +
  scale_shape_manual(
    values = baseline_burden_shapes,
    name = "Outcome",
    guide = guide_legend(
      nrow = 1,
      byrow = TRUE,
      title.position = "left",
      override.aes = list(alpha = 1)
    )
  ) +
  coord_cartesian(xlim = p1d_x_limits, clip = "off") +
  labs(
    x = "Annualised rate per 100 000/year,\n2025–50 (log)",
    y = NULL,
    tag = "d"
  ) +
  theme_lancet(base_size = journal_compact_text_size) +
  theme(
    legend.position = "top",
    legend.direction = "horizontal",
    legend.box = "vertical",
    legend.background = element_rect(fill = "#FFFFFFE8", colour = NA),
    legend.key.width = unit(0.34, "cm"),
    axis.ticks.y = element_blank()
  )

## Compose Figure 1 ------------------------------------------------------------

figure1 <- free(p1a) + free(p1b) + p1c + p1d +
  plot_layout(widths = c(0.98, 1.02)) &
  theme_lancet_tags()

save_main_figure(figure1, "figure_1_baseline_heterogeneity", height = 6.2)
cat("Figure 1 saved.\n")
