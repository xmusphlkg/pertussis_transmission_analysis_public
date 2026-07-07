## eFigure 9: scenario-ordering and endpoint-robustness diagnostics. ----------------------
## Section: extended figure setup.
## Purpose: scenario-ordering and endpoint-robustness diagnostics.

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
  library(tidyr)
})

read_csv_local <- function(...) {
  readr::read_csv(model_path(...), show_col_types = FALSE)
}

strategy_labels <- c(
  current = "Current",
  higher_child_coverage = "Nominal coverage floor",
  adolescent_booster = "Adolescent booster",
  pregnancy_tdap_scaleup = "Pregnancy Tdap",
  cocooning_adjunct = "Close-contact adjunct",
  maternal_immunization = "Infant-exposure composite",
  targeted_pep_high_risk = "Targeted PEP",
  resistance_guided_treatment = "Resistance-guided care",
  next_generation_vaccine = "High transmission blocking",
  combined_strategy = "Combined stress test"
)

class_labels <- c(
  combined_stress_test_package = "Combined stress test",
  high_transmission_blocking_vaccine_target = "High transmission blocking",
  infant_protection_and_exposure_reduction = "Infant protection/exposure",
  management_modifiers = "Management modifiers",
  current_practice = "Current practice",
  routine_program_marginal_levers = "Routine marginal strategies"
)

strategy_order <- c(
  "combined_strategy",
  "next_generation_vaccine",
  "maternal_immunization",
  "resistance_guided_treatment",
  "cocooning_adjunct",
  "pregnancy_tdap_scaleup",
  "targeted_pep_high_risk",
  "adolescent_booster",
  "current",
  "higher_child_coverage"
)

programme_strategy_order <- c(
  "timeliness_only",
  "maternal_immunization",
  "cocooning_adjunct",
  "targeted_pep_high_risk",
  "pregnancy_tdap_scaleup",
  "adolescent_booster"
)

programme_strategy_labels <- c(
  timeliness_only = "Routine schedule\ntimeliness",
  maternal_immunization = "Infant-exposure\ncomposite",
  cocooning_adjunct = "Close-contact\nadult adjuncts",
  targeted_pep_high_risk = "Targeted high-risk\nPEP",
  pregnancy_tdap_scaleup = "Pregnancy Tdap\nscale-up",
  adolescent_booster = "Adolescent booster\nscale-up"
)

panel_theme <- theme_lancet_panel(base_size = journal_compact_text_size, plot_margin = margin(3, 3, 3, 3))

pct_label <- function(x) lancet_percent(x, accuracy = 1)

routine <- read_csv_local("outputs", "tables", "routine_timeliness_sensitivity.csv") %>%
  mutate(
    strategy_label = recode(
      strategy,
      current = "Current",
      coverage_floor_only = "Nominal coverage floor",
      timeliness_only = "Timeliness only",
      coverage_floor_plus_timeliness = "Coverage + timeliness"
    ),
    strategy_label = factor(
      strategy_label,
      levels = c("Current", "Nominal coverage floor", "Timeliness only", "Coverage + timeliness")
    )
  )

p_a <- ggplot(routine, aes(strategy_label, median_relative_reduction_infant_cases, fill = timeliness_applied)) +
  geom_hline(yintercept = 0, linewidth = 0.25, colour = "grey45") +
  geom_col(width = 0.68, colour = "black", linewidth = 0.18) +
  geom_text(aes(label = countries_with_positive_reduction), vjust = -0.35, size = journal_heatmap_cell_text_size) +
  scale_y_continuous(labels = pct_label, limits = c(-0.08, 0.42), expand = expansion(mult = c(0.02, 0.10))) +
  scale_fill_manual(values = manuscript_binary_colours, guide = "none") +
  labs(x = NULL, y = "Median infant-case reduction", caption = "Text: countries with positive reduction") +
  panel_theme +
  theme(axis.text.x = element_text(angle = 35, hjust = 1))

horizon <- read_csv_local("outputs", "tables", "intervention_horizon_rank_summary.csv") %>%
  mutate(
    scenario_label = factor(recode(scenario, !!!strategy_labels), levels = rev(strategy_labels[strategy_order])),
    analysis_window = recode(
      analysis_window,
      `2025_2029` = "2025–2029",
      `2025_2034` = "2025–2034",
      `2025_2039` = "2025–2039",
      `2025_2050_full_horizon` = "2025–2050",
      `2030_2050_excluding_initial_transient` = "2030–2050"
    )
  )

p_b <- ggplot(horizon, aes(analysis_window, scenario_label, fill = median_rank)) +
  geom_tile(colour = "white", linewidth = lancet_heatmap_tile_linewidth) +
  geom_text(aes(label = sprintf("%.0f", median_rank)), size = journal_heatmap_cell_text_size, colour = "black") +
  scale_fill_rank(
    limits = c(1, 10),
    name = "Median rank",
    guide = guide_lancet_colourbar(barwidth = unit(4.2, "cm"))
  ) +
  labs(x = "Analysis window", y = NULL) +
  theme_lancet_heatmap(
    base_size = journal_compact_text_size,
    plot_margin = margin(3, 3, 3, 3),
    x_angle = 35,
    x_hjust = 1
  )

age_summary <- read_csv_local("outputs", "tables", "infant_age_split_horizon_sensitivity.csv") %>%
  filter(analysis_window == "2025_2050_full_horizon", scenario %in% strategy_order) %>%
  group_by(scenario, age_group) %>%
  summarise(median_reduction = median(relative_reduction_infant_cases, na.rm = TRUE), .groups = "drop") %>%
  mutate(
    scenario_label = factor(recode(scenario, !!!strategy_labels), levels = rev(strategy_labels[strategy_order])),
    age_group = recode(age_group, infant_0_2m = "0–2 mo", infant_3_11m = "3–11 mo")
  )

p_c <- ggplot(age_summary, aes(age_group, scenario_label, fill = median_reduction)) +
  geom_tile(colour = "white", linewidth = lancet_heatmap_tile_linewidth) +
  geom_text(aes(label = pct_label(median_reduction)), size = journal_heatmap_cell_text_size) +
  scale_fill_reduction(
    midpoint = 0,
    breaks = c(0, 0.25, 0.50, 0.75),
    labels = pct_label,
    name = "Reduction",
    guide = guide_lancet_colourbar(barwidth = unit(2.6, "cm"))
  ) +
  labs(x = "Infant age group", y = NULL) +
  theme_lancet_heatmap(
    base_size = journal_compact_text_size,
    plot_margin = margin(3, 3, 3, 3)
  )

stability <- read_csv_local("outputs", "tables", "intervention_rank_stability_diagnostics.csv") %>%
  filter(scenario %in% strategy_order) %>%
  mutate(
    scenario_label = factor(recode(scenario, !!!strategy_labels), levels = rev(strategy_labels[strategy_order])),
    window_top2 = analysis_window_cells_ranked_top_two / analysis_window_cells,
    infant_top2 = infant_age_window_cells_ranked_top_two / infant_age_window_cells
  ) %>%
  select(scenario_label, window_top2, infant_top2, median_infant_age_window_reduction) %>%
  pivot_longer(c(window_top2, infant_top2), names_to = "diagnostic", values_to = "share") %>%
  mutate(diagnostic = recode(diagnostic, window_top2 = "Window cells top 2", infant_top2 = "Infant-age cells top 2"))

p_d <- ggplot(stability, aes(share, scenario_label, colour = diagnostic)) +
  geom_point(size = 1.7) +
  scale_x_continuous(labels = pct_label, limits = c(0, 1), expand = expansion(mult = c(0.02, 0.05))) +
  scale_colour_manual(values = manuscript_diagnostic_colours[c("Window cells top 2", "Infant-age cells top 2")]) +
  labs(x = "Share of diagnostic cells", y = NULL, colour = NULL) +
  panel_theme

psa <- read_csv_local("outputs", "tables", "joint_psa_rank_acceptability.csv") %>%
  filter(country == "All_countries_pooled", rank == 1, strategy %in% strategy_order) %>%
  distinct(strategy, .keep_all = TRUE) %>%
  mutate(
    strategy_label = factor(recode(strategy, !!!strategy_labels), levels = rev(strategy_labels[strategy_order]))
  )

p_e <- ggplot(psa, aes(probability_top_2, strategy_label)) +
  geom_col(width = 0.65, fill = manuscript_diagnostic_colours["Top 2"], colour = "black", linewidth = 0.15) +
  geom_point(aes(x = probability_within_10_percent_of_best), size = 1.5, colour = manuscript_diagnostic_colours["Within 10% of best"]) +
  scale_x_continuous(labels = pct_label, limits = c(0, 1), expand = expansion(mult = c(0.01, 0.04))) +
  labs(x = "Probability", y = NULL, caption = "Bars: top 2; dots: within 10% of best") +
  panel_theme

age_pattern <- read_csv_local("outputs", "tables", "age_pattern_scenario_ordering_sensitivity.csv") %>%
  filter(ordering_basis %in% c("all_profiles_unweighted", "external_age_pattern_weighted", "external_age_pattern_pass_filter")) %>%
  mutate(
    ordering_basis = recode(
      ordering_basis,
      all_profiles_unweighted = "All profiles",
      external_age_pattern_weighted = "Age-pattern weighted",
      external_age_pattern_pass_filter = "Pass filter"
    ),
    scenario_class_label = factor(recode(scenario_class, !!!class_labels), levels = rev(class_labels))
  )

p_f <- ggplot(age_pattern, aes(ordering_basis, scenario_class_label, fill = class_rank)) +
  geom_tile(colour = "white", linewidth = lancet_heatmap_tile_linewidth) +
  geom_text(aes(label = sprintf("%.0f", class_rank)), size = journal_heatmap_cell_text_size) +
  scale_fill_rank(
    limits = c(1, 6),
    name = "Class rank",
    guide = guide_lancet_colourbar(barwidth = unit(2.6, "cm"))
  ) +
  labs(x = "Ordering basis", y = NULL) +
  theme_lancet_heatmap(
    base_size = journal_compact_text_size,
    plot_margin = margin(3, 3, 3, 3),
    x_angle = 25,
    x_hjust = 1
  )

age_pattern_programme <- read_csv_local("outputs", "tables", "lancet_age_pattern_weighted_strategy_summary.csv") %>%
  filter(
    strategy %in% programme_strategy_order,
    ordering_basis %in% c(
      "all_profiles_unweighted",
      "age_data_profiles_unweighted",
      "age_pattern_weighted"
    )
  ) %>%
  mutate(
    strategy_label = stringr::str_replace_all(as.character(programme_strategy_labels[strategy]), "\n", " "),
    strategy_label_plot = factor(
      programme_strategy_labels[strategy],
      levels = rev(programme_strategy_labels[programme_strategy_order])
    ),
    weighted_median_primary_case_reduction = as.numeric(weighted_median_primary_case_reduction),
    weighted_median_primary_cases_per_100k = as.numeric(weighted_median_primary_cases_per_100k),
    weighted_iqr_primary_case_reduction = as.character(weighted_iqr_primary_case_reduction),
    weighted_iqr_low_primary_case_reduction = as.numeric(stringr::str_match(
      weighted_iqr_primary_case_reduction,
      "^\\s*([-0-9.eE]+)\\s+to\\s+([-0-9.eE]+)\\s*$"
    )[, 2]),
    weighted_iqr_high_primary_case_reduction = as.numeric(stringr::str_match(
      weighted_iqr_primary_case_reduction,
      "^\\s*([-0-9.eE]+)\\s+to\\s+([-0-9.eE]+)\\s*$"
    )[, 3]),
    ordering_basis_label = recode(
      ordering_basis,
      all_profiles_unweighted = "All profiles",
      age_data_profiles_unweighted = "Profiles with age data",
      age_pattern_weighted = "Age-pattern weighted"
    ),
    ordering_basis_label = factor(
      ordering_basis_label,
      levels = c("All profiles", "Profiles with age data", "Age-pattern weighted")
    )
  )

readr::write_csv(
  age_pattern_programme %>%
    transmute(
      ordering_basis,
      ordering_basis_label = as.character(ordering_basis_label),
      strategy,
      strategy_label,
      country_count,
      effective_country_weight_sum,
      weighted_median_primary_case_reduction,
      weighted_iqr_low_primary_case_reduction,
      weighted_iqr_high_primary_case_reduction,
      weighted_iqr_primary_case_reduction,
      weighted_median_primary_cases_per_100k,
      strategy_rank_within_basis
    ),
  model_path("outputs", "tables", "extended_data_figure_9g_age_pattern_programme_robustness.csv")
)

age_pattern_programme_colours <- c(
  "All profiles" = manuscript_colour("mid_grey"),
  "Profiles with age data" = manuscript_colour("sky"),
  "Age-pattern weighted" = manuscript_colour("blue")
)

age_pattern_programme_x_range <- range(
  c(
    0,
    age_pattern_programme$weighted_median_primary_case_reduction,
    age_pattern_programme$weighted_iqr_low_primary_case_reduction,
    age_pattern_programme$weighted_iqr_high_primary_case_reduction
  ),
  na.rm = TRUE
)

age_pattern_programme_x_limits <- c(
  floor((age_pattern_programme_x_range[[1]] - 0.01) / 0.05) * 0.05,
  ceiling((age_pattern_programme_x_range[[2]] + 0.01) / 0.05) * 0.05
)

p_g <- ggplot(
  age_pattern_programme,
  aes(weighted_median_primary_case_reduction, strategy_label_plot, fill = ordering_basis_label)
) +
  geom_vline(xintercept = 0, linewidth = 0.24, colour = manuscript_colour("pale_grey")) +
  geom_col(
    position = position_dodge(width = 0.74),
    width = 0.62,
    colour = "white",
    linewidth = 0.12,
    alpha = 0.92
  ) +
  geom_errorbar(
    aes(
      xmin = weighted_iqr_low_primary_case_reduction,
      xmax = weighted_iqr_high_primary_case_reduction
    ),
    orientation = "y",
    position = position_dodge(width = 0.74),
    width = 0.18,
    linewidth = 0.28,
    colour = lancet_text_colour
  ) +
  scale_x_continuous(
    labels = pct_label,
    breaks = seq(age_pattern_programme_x_limits[[1]], age_pattern_programme_x_limits[[2]], by = 0.05)
  ) +
  scale_fill_manual(
    values = age_pattern_programme_colours,
    labels = c("All", "Age data", "Weighted"),
    name = NULL,
    guide = guide_legend(nrow = 1, byrow = TRUE)
  ) +
  coord_cartesian(xlim = age_pattern_programme_x_limits, clip = "off") +
  labs(x = "Median reduction in symptomatic cases among people aged <18 years", y = NULL) +
  theme_lancet_panel(base_size = journal_compact_text_size, plot_margin = margin(3, 5, 3, 3), show_y_grid = TRUE) +
  theme(
    panel.grid.major.x = element_line(linewidth = 0.16, colour = lancet_grid_light_colour),
    legend.position = "top",
    axis.ticks.y = element_line(linewidth = lancet_axis_linewidth, colour = lancet_text_colour)
  )

extended9_layout <- "
ABC
DEF
GGG
"

extended9 <- wrap_plots(
  A = free(p_a), B = p_b, C = p_c,
  D = p_d, E = free(p_e), F = p_f,
  G = free(p_g),
  design = extended9_layout,
  guides = "keep"
) +
  plot_layout(heights = c(1, 1, 0.76)) +
  plot_annotation(tag_levels = "A") &
  (theme(plot.margin = margin(4, 4, 4, 4)) + theme_lancet_tags())

save_appendix_figure(
  extended9,
  "extended_data_figure_9_scenario_robustness",
  width = lancet_double_width * 1.18,
  height = 10.0
)

cat("eFigure 9 (scenario-ordering and outcome robustness diagnostics) saved.\n")
