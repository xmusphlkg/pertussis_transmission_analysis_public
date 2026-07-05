## eFigure 11: implementation and structural robustness diagnostics. ----------------------
## Section: extended figure setup.
## Purpose: implementation and structural robustness diagnostics.

args <- commandArgs(FALSE)
file_arg <- sub("^--file=", "", args[grepl("^--file=", args)])
script_dir <- if (length(file_arg) > 0) dirname(normalizePath(file_arg[[1]])) else file.path(getwd(), "scripts_R")
source(file.path(script_dir, "10_shared.R"))

suppressPackageStartupMessages({
  library(dplyr)
  library(forcats)
  library(ggplot2)
  library(patchwork)
  library(readr)
  library(scales)
  library(stringr)
  library(tidyr)
})

read_csv_local <- function(...) {
  readr::read_csv(model_path(...), show_col_types = FALSE)
}

panel_theme <- theme_lancet_panel(base_size = journal_compact_text_size, plot_margin = margin(3, 3, 3, 3))

pct_label <- function(x) scales::percent(x, accuracy = 1)

parse_range_bounds <- function(x) {
  match <- stringr::str_match(
    x,
    "^\\s*([+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+))\\s*to\\s*([+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+))\\s*$"
  )
  tibble::tibble(
    lower = as.numeric(match[, 2]),
    upper = as.numeric(match[, 3])
  )
}

strategy_labels <- c(
  current = "Current",
  maternal_immunization = "Infant-exposure strategy",
  maternal_direct_antibody_only = "Direct antibody only",
  pregnancy_tdap_scaleup = "Pregnancy Tdap",
  cocooning_adjunct = "Close-contact adjunct",
  adolescent_booster = "Adolescent booster",
  targeted_pep_high_risk = "Targeted PEP",
  resistance_guided_treatment = "Resistance-guided care",
  next_generation_vaccine = "High transmission blocking",
  combined_strategy = "Combined stress test",
  higher_child_coverage = "Nominal coverage floor"
)

portfolio_labels <- c(
  current = "Current",
  routine_timeliness = "Timeliness",
  timeliness_pregnancy_tdap = "Timeliness +\npregnancy Tdap",
  timeliness_targeted_pep = "Timeliness +\ntargeted PEP",
  infant_exposure = "Infant exposure",
  timeliness_infant_exposure = "Timeliness +\ninfant exposure",
  timeliness_infant_exposure_targeted_pep = "Timeliness + exposure +\ntargeted PEP",
  routine_timeliness_resistance_guided = "Timeliness +\nresistance mgmt",
  infant_exposure_resistance_guided = "Exposure +\nresistance mgmt",
  timeliness_infant_exposure_resistance_guided = "Timeliness + exposure +\nresistance mgmt",
  timeliness_infant_exposure_targeted_pep_resistance_guided = "Timeliness + exposure +\nPEP + resistance"
)

portfolio_bounds <- read_csv_local("outputs", "tables", "program_portfolio_factorial_summary.csv") %>%
  pull(iqr_relative_reduction_infant_cases) %>%
  parse_range_bounds()

portfolio <- read_csv_local("outputs", "tables", "program_portfolio_factorial_summary.csv") %>%
  bind_cols(portfolio_bounds) %>%
  mutate(
    portfolio_label = recode(portfolio, !!!portfolio_labels),
    portfolio_label = fct_reorder(portfolio_label, median_relative_reduction_infant_cases),
    resistance_layer = if_else(resistance_guided, "Includes resistance-guided care", "No resistance-guided care"),
    first_label = if_else(countries_ranked_first_infant_cases > 0, paste0(countries_ranked_first_infant_cases, "/10 first"), "")
  )

p_a <- ggplot(portfolio, aes(median_relative_reduction_infant_cases, portfolio_label, colour = resistance_layer)) +
  geom_vline(xintercept = 0, linewidth = 0.25, colour = manuscript_colour("mid_grey")) +
  geom_errorbar(aes(xmin = lower, xmax = upper), width = 0.18, linewidth = 0.34, orientation = "y") +
  geom_point(aes(size = implementation_intensity), fill = "white") +
  geom_text(aes(label = first_label), nudge_x = 0.035, hjust = 0, size = 1.8, show.legend = FALSE) +
  scale_x_continuous(labels = pct_label, limits = c(0, 1.18), expand = expansion(mult = c(0.02, 0.08))) +
  scale_colour_manual(values = manuscript_resistance_layer_colours, name = NULL) +
  scale_size_continuous(range = c(1.2, 2.4), breaks = c(0, 3, 6), name = "Implementation\nintensity") +
  labs(x = "Median infant-case reduction vs current", y = NULL) +
  panel_theme +
  theme(legend.position = "none")

contact <- read_csv_local("outputs", "tables", "infant_contact_sensitivity.csv") %>%
  filter(strategy %in% c("current", "maternal_immunization")) %>%
  mutate(strategy_label = recode(strategy, !!!strategy_labels))

p_b <- ggplot(contact, aes(infant_contact_multiplier, median_infant_cases_per_100k, colour = strategy_label)) +
  geom_line(linewidth = 0.35) +
  geom_point(size = 1.6) +
  scale_x_continuous(breaks = c(0.75, 1, 1.25, 1.5)) +
  scale_y_continuous(labels = label_number(accuracy = 1), expand = expansion(mult = c(0.02, 0.08))) +
  scale_colour_manual(values = manuscript_programme_colours[c("Current", "Infant-exposure strategy")], name = NULL) +
  labs(x = "Infant-contact multiplier", y = "Median infant cases per 100k/y") +
  panel_theme

maternal <- read_csv_local("outputs", "tables", "maternal_duration_sensitivity.csv") %>%
  filter(strategy != "current") %>%
  mutate(strategy_label = recode(strategy, !!!strategy_labels))

p_c <- ggplot(maternal, aes(maternal_protection_duration_days, median_infant_case_reduction_vs_current_5y, colour = strategy_label)) +
  geom_line(linewidth = 0.35) +
  geom_point(size = 1.6) +
  scale_x_continuous(breaks = c(90, 180, 270)) +
  scale_y_continuous(labels = pct_label, limits = c(0, 0.70), expand = expansion(mult = c(0.02, 0.08))) +
  scale_colour_manual(
    values = manuscript_programme_colours[c("Direct antibody only", "Infant-exposure strategy")],
    name = NULL,
    guide = guide_legend(nrow = 2)
  ) +
  labs(x = "Maternal-protection duration, days", y = "Median infant-case reduction, 5 y") +
  panel_theme

temporal <- read_csv_local("outputs", "tables", "temporal_assumption_sensitivity.csv") %>%
  mutate(
    scenario_label = recode(
      scenario,
      burnin_10y = "Burn-in 10 y",
      burnin_15y = "Burn-in 15 y",
      burnin_30y = "Burn-in 30 y",
      npi_country_profile = "Country NPI",
      npi_reduction_half = "Half NPI",
      npi_none = "No NPI"
    ),
    temporal_dimension = recode(
      temporal_dimension,
      burn_in = "Burn-in",
      npi_contact_shock = "NPI contact shock"
    ),
    scenario_label = fct_reorder(scenario_label, median_infant_cases_per_100k_5y)
  )

p_d <- ggplot(temporal, aes(median_infant_cases_per_100k_5y, scenario_label, fill = temporal_dimension)) +
  geom_col(width = 0.65, colour = "black", linewidth = 0.15) +
  scale_x_continuous(labels = label_number(accuracy = 1), expand = expansion(mult = c(0.02, 0.08))) +
  scale_fill_manual(values = manuscript_temporal_colours, name = NULL) +
  labs(x = "Median infant cases per 100k/y, 2025-2029", y = NULL) +
  panel_theme

event_scale <- read_csv_local("outputs", "tables", "deterministic_event_scale_diagnostics.csv") %>%
  mutate(
    scenario_label = recode(scenario, !!!strategy_labels),
    low_event = str_detect(event_scale_flag, "Low infant-event count"),
    scenario_label = factor(
      scenario_label,
      levels = rev(c(
        "Combined stress test", "High transmission blocking", "Infant-exposure strategy",
        "Resistance-guided care", "Close-contact adjunct", "Pregnancy Tdap",
        "Targeted PEP", "Adolescent booster", "Current", "Nominal coverage floor"
      ))
    )
  )

p_e <- ggplot(event_scale, aes(annual_infant_cases_count + 0.5, scenario_label, colour = low_event)) +
  geom_point(position = position_jitter(height = 0.12, width = 0), size = 1.35, alpha = 0.85) +
  scale_x_log10(labels = label_number(accuracy = 1), expand = expansion(mult = c(0.03, 0.08))) +
  scale_colour_manual(
    values = c(`FALSE` = manuscript_colour("mid_grey"), `TRUE` = manuscript_colour("vermillion")),
    labels = c("Other cells", "Low infant-event cells"),
    name = NULL
  ) +
  labs(x = "Annual infant cases, count + 0.5", y = NULL) +
  panel_theme

stochastic <- read_csv_local("outputs", "tables", "individual_stochastic_toy_summary.csv") %>%
  mutate(
    country_label = str_replace_all(country, "_", " "),
    scenario_label = recode(
      scenario,
      homogeneous_all_contacts = "Homogeneous",
      setting_clustered = "Setting clustered",
      setting_clustered_high_household = "High household"
    )
  )

p_f <- ggplot(stochastic, aes(outbreak_probability_20plus, extinction_probability_3_or_fewer, colour = scenario_label)) +
  geom_point(aes(size = mean_household_clusters_touched), alpha = 0.86) +
  geom_text(aes(label = iso3), nudge_y = 0.018, size = 1.7, show.legend = FALSE) +
  scale_x_continuous(labels = pct_label, limits = c(0, 0.32), expand = expansion(mult = c(0.03, 0.08))) +
  scale_y_continuous(labels = pct_label, limits = c(0.52, 0.86), expand = expansion(mult = c(0.03, 0.08))) +
  scale_colour_manual(values = manuscript_stochastic_colours, name = NULL, guide = guide_legend(nrow = 2)) +
  scale_size_continuous(range = c(1.0, 2.5), name = "Mean household\nclusters", guide = guide_legend(nrow = 1)) +
  labs(x = "Pr(outbreak >=20 infections)", y = "Pr(extinction <=3 infections)") +
  panel_theme +
  theme(legend.text = element_text(size = 5.4))

extended11 <- wrap_plots(
  p_a, p_b, p_c,
  p_d, p_e, free(p_f),
  ncol = 3,
  guides = "keep"
) +
  plot_layout(widths = c(1.15, 1.08, 1), heights = c(1.03, 1)) +
  plot_annotation(tag_levels = "A") &
  (theme(plot.margin = margin(4, 4, 4, 4)) + theme_lancet_tags())

save_appendix_figure(
  extended11,
  "extended_data_figure_11_implementation_structural_robustness",
  width = lancet_double_width * 1.16,
  height = 7.9
)

cat("eFigure 11 (implementation and structural robustness diagnostics) saved.\n")
