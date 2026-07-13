# Retired figure script: not part of the current publication pipeline.
# Set PERTUSSIS_ALLOW_RETIRED_R_FIGURES=1 only to regenerate this provenance-only figure.
if (Sys.getenv("PERTUSSIS_ALLOW_RETIRED_R_FIGURES") != "1") {
  stop(
    "Retired figure script: not part of the current publication pipeline. ",
    "Run scripts_R/10_figure_set.R for current figures, or set ",
    "PERTUSSIS_ALLOW_RETIRED_R_FIGURES=1 to reproduce this provenance-only figure."
  )
}

args <- commandArgs(FALSE)
file_arg <- sub("^--file=", "", args[grepl("^--file=", args)])
script_dir <- if (length(file_arg) > 0) dirname(normalizePath(file_arg[[1]])) else file.path(getwd(), "scripts_R")
source(file.path(script_dir, "10_shared.R"))

## Extended Data Figure 4: calibration acceptance and fit diagnostics. ----------------------
## Section: extended-data figure setup.
## Purpose: calibration acceptance and fit diagnostics.

normalise_bool <- function(x) {
  tolower(as.character(x)) %in% c("true", "1", "yes", "accepted")
}

calibration_path <- model_path("outputs", "tables", "calibration_all_countries.csv")
calibration <- if (file.exists(calibration_path)) {
  readr::read_csv(calibration_path, show_col_types = FALSE)
} else {
  list.files(model_path("outputs", "tables"), pattern = "^calibration_.*\\.csv$", full.names = TRUE) %>%
    discard(~str_detect(basename(.x), "all_countries")) %>%
    map_dfr(readr::read_csv, show_col_types = FALSE)
}

fitted_temporal_columns <- c(
  "fitted_temporal_mean_reported_cases",
  "fitted_temporal_range_low",
  "fitted_temporal_range_high"
)

if (!all(fitted_temporal_columns %in% names(calibration))) {
  # Minimal calibration table: generate a simplified diagnostic figure
  calibration <- calibration %>%
    mutate(
      country = str_replace_all(country, " ", "_"),
      country_label = factor(format_country(country), levels = country_label_levels),
      country_code = country_codes[country],
      calibration_accepted_bool = if ("calibration_accepted" %in% names(.)) {
        normalise_bool(calibration_accepted)
      } else {
        normalise_bool(calibration_success)
      },
      fit_score = as.numeric(ifelse("fit_score" %in% names(.), fit_score, 0))
    )
} else {
  calibration <- calibration %>%
    mutate(
      country = str_replace_all(country, " ", "_"),
      country_label = factor(format_country(country), levels = country_label_levels),
      country_code = country_codes[country],
      calibration_accepted_bool = if ("calibration_accepted" %in% names(.)) {
        normalise_bool(calibration_accepted)
      } else {
        normalise_bool(calibration_success)
      },
      fitted_temporal_mean_reported_cases = as.numeric(fitted_temporal_mean_reported_cases),
      fitted_temporal_range_width = fitted_temporal_range_high - fitted_temporal_range_low,
      relative_fitted_temporal_range_width = fitted_temporal_range_width /
        pmax(fitted_temporal_mean_reported_cases, 1e-9)
    )
}

p_ed4a <- calibration %>%
  mutate(country_burden_order = factor(as.character(country_label), levels = rev(baseline_order))) %>%
  ggplot(aes(fit_score, country_burden_order, fill = calibration_accepted_bool)) +
  geom_col(width = 0.62) +
  geom_text(aes(label = if_else(calibration_accepted_bool, "accepted", "not accepted")), hjust = -0.08, size = 2) +
  scale_x_continuous(labels = label_number(accuracy = 1), expand = expansion(mult = c(0, 0.18))) +
  scale_fill_manual(values = c("TRUE" = manuscript_colour("green"), "FALSE" = manuscript_colour("pale_grey")), guide = "none") +
  labs(x = "Calibration fit score", y = NULL) +
  theme_lancet()

observed_annual <- readr::read_csv(model_path("data", "processed", "pertussis_incidence_timeseries.csv"), show_col_types = FALSE) %>%
  mutate(country = str_replace_all(config_key, " ", "_")) %>%
  group_by(country, Year) %>%
  summarise(observed_cases = sum(Cases, na.rm = TRUE), .groups = "drop") %>%
  mutate(
    country_label = factor(format_country(country), levels = country_label_levels),
    country_code = factor(country_codes[country], levels = country_codes[country_levels])
  )

p_ed4b <- tryCatch({
  has_fitted_temporal_range <- all(c(
    "fitted_temporal_range_low",
    "fitted_temporal_range_high",
    "fitted_temporal_mean_reported_cases"
  ) %in% names(calibration))
  if (has_fitted_temporal_range) {
    observed_annual %>%
      filter(country %in% calibration$country) %>%
      ggplot(aes(Year, observed_cases)) +
      geom_line(linewidth = 0.28, colour = manuscript_colour("grey")) +
      geom_rect(
        data = calibration,
        aes(
          xmin = -Inf,
          xmax = Inf,
          ymin = fitted_temporal_range_low,
          ymax = fitted_temporal_range_high
        ),
        fill = manuscript_colour("vermillion"),
        alpha = 0.12,
        inherit.aes = FALSE
      ) +
      geom_hline(
        data = calibration,
        aes(yintercept = fitted_temporal_mean_reported_cases),
        colour = manuscript_colour("vermillion"),
        linewidth = 0.35,
        inherit.aes = FALSE
      ) +
      facet_wrap(~country_code, scales = "free_y", ncol = 4) +
      scale_x_continuous(breaks = pretty_breaks(n = 3)) +
      scale_y_continuous(labels = label_number(accuracy = 1)) +
      labs(x = NULL, y = "Annual reported cases\n(fitted range shaded)") +
      theme_lancet()
  } else {
    observed_annual %>%
      filter(country %in% calibration$country) %>%
      ggplot(aes(Year, observed_cases)) +
      geom_line(linewidth = 0.28, colour = manuscript_colour("grey")) +
      facet_wrap(~country_code, scales = "free_y", ncol = 4) +
      scale_x_continuous(breaks = pretty_breaks(n = 3)) +
      scale_y_continuous(labels = label_number(accuracy = 1)) +
      labs(x = NULL, y = "Annual reported cases") +
      theme_lancet()
  }
}, error = function(e) {
  ggplot() + annotate("text", x = 0.5, y = 0.5, label = "Observed data\nnot available", size = 3) + theme_void()
})

reporting_long <- tryCatch({
  calibration %>%
    select(country_label, reporting_multiplier_by_age) %>%
    separate_rows(reporting_multiplier_by_age, sep = ";") %>%
    separate(reporting_multiplier_by_age, into = c("age_group", "reporting_rate"), sep = "=", convert = TRUE) %>%
    mutate(
      age_group = factor(age_group, levels = names(age_labels), labels = age_labels),
      reporting_rate = as.numeric(reporting_rate)
    )
}, error = function(e) NULL)

if (!is.null(reporting_long)) {
  p_ed4c <- reporting_long %>%
    ggplot(aes(age_group, country_label, fill = reporting_rate)) +
    geom_tile(colour = "white", linewidth = lancet_heatmap_tile_linewidth) +
    scale_fill_fraction(labels = label_lancet_percent(accuracy = 1)) +
    labs(x = "Age group", y = NULL, fill = "Fitted\nreporting\nprobability") +
    theme_lancet_heatmap(x_angle = 35, x_hjust = 1)
} else {
  p_ed4c <- ggplot() + annotate("text", x = 0.5, y = 0.5, label = "Reporting data\nnot available", size = 3) + theme_void()
}

p_ed4d <- tryCatch({
  req_cols <- c("calibrated_beta", "relative_fitted_temporal_range_width", "data_fit_score")
  if (!all(req_cols %in% names(calibration))) stop("Required columns not available")
  calibration %>%
    ggplot(aes(calibrated_beta, relative_fitted_temporal_range_width)) +
    geom_point(aes(fill = data_fit_score), shape = 21, size = 2.4, stroke = 0.25, colour = "black") +
    geom_text(aes(label = country_code), nudge_y = 0.15, size = 2, check_overlap = TRUE) +
    scale_x_continuous(labels = label_number(accuracy = 0.001)) +
    scale_y_continuous(labels = label_number(accuracy = 0.1)) +
    scale_fill_infant_burden(labels = label_number(accuracy = 1)) +
    labs(x = expression("Calibrated " * beta), y = "Fitted temporal range / fitted mean", fill = "Data fit\nscore") +
    theme_lancet()
}, error = function(e) {
  # Fallback: show calibrated beta vs fit score
  tryCatch({
    calibration %>%
      ggplot(aes(calibrated_beta, data_fit_score)) +
      geom_point(shape = 21, size = 2.4, stroke = 0.25, colour = "black", fill = manuscript_colour("blue")) +
      geom_text(aes(label = country_code), nudge_y = 0.5, size = 2, check_overlap = TRUE) +
      scale_x_continuous(labels = label_number(accuracy = 0.001)) +
      labs(x = expression("Calibrated " * beta), y = "Data fit score") +
      theme_lancet()
  }, error = function(e2) {
    ggplot() + annotate("text", x = 0.5, y = 0.5, label = "Fitted temporal range\nnot available", size = 3) + theme_void()
  })
})

extended4 <- ((p_ed4a | p_ed4b) / (p_ed4c | p_ed4d)) +
  plot_layout(guides = "keep") +
  plot_annotation(tag_levels = "A") &
  (theme(plot.margin = margin(4, 4, 4, 4)) + theme_lancet_tags())

save_appendix_figure(extended4, "extended_data_figure_4_calibration_diagnostics", height = 8.3)
