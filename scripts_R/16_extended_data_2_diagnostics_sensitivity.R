args <- commandArgs(FALSE)
file_arg <- sub("^--file=", "", args[grepl("^--file=", args)])
script_dir <- if (length(file_arg) > 0) dirname(normalizePath(file_arg[[1]])) else file.path(getwd(), "scripts_R")
source(file.path(script_dir, "10_shared.R"))

## Extended Data Figure 2: surveillance, calibration, and reporting diagnostics. ----------------------
## Section: extended-data figure setup.
## Purpose: surveillance, calibration, and reporting diagnostics.

observed <- readr::read_csv(model_path("data", "processed", "pertussis_incidence_timeseries.csv"), show_col_types = FALSE)
observed_annual <- observed %>%
  mutate(country = config_key) %>%
  group_by(country, Year) %>%
  summarise(observed_cases = sum(Cases, na.rm = TRUE), .groups = "drop") %>%
  left_join(baseline %>% select(country, total_population), by = "country") %>%
  add_country_label() %>%
  mutate(
    observed_reported_incidence = observed_cases / total_population * 1e5,
    country_code = factor(country_codes[country], levels = country_codes[country_levels])
  )

require_columns <- function(df, cols, dataset_name) {
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

calibration_path <- model_path("outputs", "tables", "calibration_all_countries.csv")
if (!file.exists(calibration_path)) {
  stop(
    "eFigure 2 publication rendering requires outputs/tables/calibration_all_countries.csv from `make calibrate`.",
    call. = FALSE
  )
}
calibration <- readr::read_csv(calibration_path, show_col_types = FALSE)

p_ed2a <- observed_annual %>%
  ggplot(aes(Year, observed_reported_incidence)) +
  geom_line(linewidth = 0.28, colour = manuscript_colour("grey")) +
  facet_wrap(~country_code, scales = "free_y", nrow = 2) +
  scale_x_continuous(breaks = pretty_breaks(n = 2)) +
  scale_y_continuous(labels = label_number(accuracy = 0.1)) +
  labs(x = NULL, y = "Observed reported\nincidence per\n100,000/year") +
  theme_lancet() +
  theme(axis.text.x = element_text(size = 5.2))

if ("calibration_accepted" %in% names(calibration) && !"calibration_success" %in% names(calibration)) {
  calibration$calibration_success <- tolower(as.character(calibration$calibration_accepted)) %in% c("true", "1", "yes", "accepted")
}
require_columns(
  calibration,
  c(
    "country",
    "posterior_interval_low",
    "posterior_interval_high",
    "total_reported_cases",
    "analysis_years",
    "calibration_success"
  ),
  "calibration_all_countries.csv"
)
calibration_diagnostic <- calibration %>%
  filter(.data$calibration_success == TRUE) %>%
  select(country, posterior_interval_low, posterior_interval_high, total_reported_cases, analysis_years, calibration_success) %>%
  mutate(analysis_years = as.numeric(analysis_years)) %>%
  filter(is.finite(.data$analysis_years), .data$analysis_years > 0) %>%
  mutate(model_annual_reported_cases = total_reported_cases / analysis_years)
if (nrow(calibration_diagnostic) == 0) {
  stop(
    "eFigure 2 panel B requires accepted calibration diagnostics with posterior intervals.",
    call. = FALSE
  )
}

p_ed2b <- observed %>%
  group_by(config_key, Year) %>%
  summarise(observed_cases = sum(Cases, na.rm = TRUE), .groups = "drop") %>%
  filter(config_key %in% calibration_diagnostic$country) %>%
  mutate(country_code = factor(country_codes[config_key], levels = country_codes[country_levels])) %>%
  ggplot(aes(Year, observed_cases)) +
  geom_line(linewidth = 0.35, colour = manuscript_colour("grey")) +
  geom_rect(
    data = calibration_diagnostic,
    aes(xmin = -Inf, xmax = Inf, ymin = posterior_interval_low, ymax = posterior_interval_high),
    fill = manuscript_colour("vermillion"),
    alpha = 0.12,
    inherit.aes = FALSE
  ) +
  geom_hline(
    data = calibration_diagnostic,
    aes(yintercept = model_annual_reported_cases),
    linewidth = 0.4,
    colour = manuscript_colour("vermillion"),
    inherit.aes = FALSE
  ) +
  facet_wrap(~country_code, scales = "free_y", nrow = 2) +
  scale_x_continuous(breaks = pretty_breaks(n = 2)) +
  labs(x = NULL, y = "Annual reported cases") +
  theme_lancet() +
  theme(axis.text.x = element_text(size = 5.2))

ed2c_dodge <- position_dodge(width = 0.24)

p_ed2c <- reporting_summary %>%
  select(scenario_label, annualized_reported_cases_per_100k,
         annualized_infections_per_100k, annualized_infant_cases_per_100k) %>%
  pivot_longer(-scenario_label, names_to = "metric", values_to = "value") %>%
  mutate(metric = factor(metric_labels[metric], levels = c("All infections", "Reported cases", "Infant cases"))) %>%
  group_by(scenario_label, metric) %>%
  summarise(
    median = median(value, na.rm = TRUE),
    q025 = interval_quantile(value, 0.025),
    q975 = interval_quantile(value, 0.975),
    q25 = interval_quantile(value, 0.25),
    q75 = interval_quantile(value, 0.75),
    .groups = "drop"
  ) %>%
  ggplot(aes(scenario_label, median, colour = metric, group = metric)) +
  geom_errorbar(aes(ymin = q025, ymax = q975), width = 0.12, linewidth = 0.22, alpha = 0.55,
                position = ed2c_dodge) +
  geom_errorbar(aes(ymin = q25, ymax = q75), width = 0, linewidth = 0.55,
                position = ed2c_dodge) +
  geom_line(linewidth = 0.3, position = ed2c_dodge) +
  geom_point(size = 1.8, position = ed2c_dodge) +
  scale_y_log10(labels = label_number(accuracy = 1)) +
  scale_colour_manual(values = manuscript_metric_colours[c("All infections", "Reported cases", "Infant cases")]) +
  labs(x = NULL, y = "Median incidence\nper 100,000/year\n(log; country-profile ranges)", colour = NULL) +
  theme_lancet() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))

require_columns(calibration, c("country", "reporting_multiplier_by_age"), "calibration_all_countries.csv")
reporting_long <- calibration %>%
  mutate(
    country = str_replace_all(country, " ", "_"),
    country_label = factor(format_country(country), levels = country_label_levels)
  ) %>%
  select(country_label, reporting_multiplier_by_age) %>%
  separate_rows(reporting_multiplier_by_age, sep = ";") %>%
  separate(reporting_multiplier_by_age, into = c("age_group", "reporting_rate"), sep = "=", convert = TRUE) %>%
  mutate(
    age_group = factor(age_group, levels = names(age_labels), labels = age_labels),
    reporting_rate = as.numeric(reporting_rate)
  )

expected_reporting_cells <- expand_grid(
  country_label = country_label_levels,
  age_group = unname(age_labels)
)
observed_reporting_cells <- reporting_long %>%
  transmute(country_label = as.character(country_label), age_group = as.character(age_group)) %>%
  distinct()
missing_reporting_cells <- expected_reporting_cells %>%
  anti_join(observed_reporting_cells, by = c("country_label", "age_group"))
if (nrow(missing_reporting_cells) > 0 || any(!is.finite(reporting_long$reporting_rate))) {
  stop(
    "eFigure 2 panel D requires complete fitted age-specific reporting probabilities. Missing cells: ",
    paste(
      paste0(missing_reporting_cells$country_label, "/", missing_reporting_cells$age_group),
      collapse = ", "
    ),
    call. = FALSE
  )
}

p_ed2d <- reporting_long %>%
  ggplot(aes(age_group, country_label, fill = reporting_rate)) +
  geom_tile(colour = "white", linewidth = lancet_heatmap_tile_linewidth) +
  scale_fill_fraction(
    limits = c(0, 0.7),
    breaks = seq(0, 0.7, by = 0.1),
    labels = label_lancet_percent(accuracy = 1),
    oob = scales::squish,
    guide = guide_lancet_colourbar(
      barwidth = grid::unit(5.6, "cm"),
      barheight = grid::unit(0.24, "cm"),
      title.position = "top"
    )
  ) +
  labs(x = "Age group", y = NULL, fill = "Fitted reporting probability") +
  theme_lancet_heatmap(x_angle = 35, x_hjust = 1)

extended2 <- free(p_ed2a) + free(p_ed2b) + p_ed2c + free(p_ed2d) +
  plot_layout(design = "AA\nBB\nCD", guides = "keep", heights = c(1.05, 1.05, 1.0)) +
  plot_annotation(tag_levels = "A") &
  (theme(plot.margin = margin(5, 3, 3, 5)) + theme_lancet_tags())

save_appendix_figure(extended2, "extended_data_figure_2_diagnostics_sensitivity", width = lancet_double_width * 1.05, height = 7.5)
