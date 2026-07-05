#!/usr/bin/env Rscript
## Figure 3: quantitative endpoint translation -------------------------------
## Layout: (a) Age-stratum effect matrix
##         (b) Infant-to-<18 endpoint translation gap
##         (c) Age contribution to <18 cases averted
##         (d) Age-pattern weighted robustness

args <- commandArgs(FALSE)
file_arg <- sub("^--file=", "", args[grepl("^--file=", args)])
script_dir <- if (length(file_arg) > 0) dirname(normalizePath(file_arg[[1]])) else file.path(getwd(), "scripts_R")
source(file.path(script_dir, "10_shared.R"))

read_table <- function(name) {
  readr::read_csv(model_path("outputs", "tables", name), show_col_types = FALSE)
}

programme_strategies <- c(
  "timeliness_only",
  "maternal_immunization",
  "cocooning_adjunct",
  "targeted_pep_high_risk",
  "pregnancy_tdap_scaleup",
  "adolescent_booster"
)

strategy_source_labels <- c(
  higher_child_coverage = "Coverage",
  timeliness_only = "Timeliness",
  maternal_immunization = "Infant exposure composite",
  cocooning_adjunct = "Adult/contact",
  targeted_pep_high_risk = "Targeted PEP",
  adolescent_booster = "Adolescent",
  pregnancy_tdap_scaleup = "Pregnancy Tdap"
)

strategy_plot_labels <- c(
  higher_child_coverage = "Coverage",
  timeliness_only = "Timeliness",
  maternal_immunization = "Infant exposure\ncomposite",
  cocooning_adjunct = "Adult/contact",
  targeted_pep_high_risk = "Targeted PEP",
  adolescent_booster = "Adolescent",
  pregnancy_tdap_scaleup = "Pregnancy Tdap"
)

endpoint_levels <- c("Infant", "Child 1–9", "Adolescent 10–17", "<18")
age_stratum_levels <- c("0–11 m", "1–9 y", "10–17 y")
age_stratum_colours <- c(
  "0–11 m" = palette_discrete_primary_9[[7]],
  "1–9 y" = palette_discrete_primary_9[[4]],
  "10–17 y" = palette_discrete_primary_9[[8]]
)

endpoint_axis_labels <- c(
  Infant = "0–11 m",
  `Child 1–9` = "1–9 y",
  `Adolescent 10–17` = "10–17 y",
  `<18` = "<18"
)

age_contribution_levels <- age_stratum_levels
age_contribution_colours <- age_stratum_colours

burden <- read_table("lancet_child_adolescent_strategy_burden.csv") %>%
  mutate(
    country = stringr::str_replace_all(country, " ", "_"),
    country_label_text = format_country(country),
    country_code = unname(country_codes[country]),
    strategy = as.character(strategy),
    strategy_label = unname(strategy_source_labels[strategy]),
    strategy_label_plot = factor(
      unname(strategy_plot_labels[strategy]),
      levels = rev(strategy_plot_labels[programme_strategies])
    ),
    primary_cases_per_100k = as.numeric(primary_cases_per_100k),
    infant_cases_per_100k = as.numeric(infant_cases_per_100k),
    child_1_9_cases_per_100k = as.numeric(child_1_9_cases_per_100k),
    adolescent_cases_per_100k = as.numeric(adolescent_cases_per_100k),
    primary_case_reduction = as.numeric(primary_case_reduction)
  ) %>%
  group_by(country) %>%
  mutate(
    current_primary_cases_per_100k = primary_cases_per_100k[strategy == "current"][[1]],
    current_infant_cases_per_100k = infant_cases_per_100k[strategy == "current"][[1]],
    current_child_1_9_cases_per_100k = child_1_9_cases_per_100k[strategy == "current"][[1]],
    current_adolescent_cases_per_100k = adolescent_cases_per_100k[strategy == "current"][[1]],
    infant_case_reduction = 1 - infant_cases_per_100k / pmax(current_infant_cases_per_100k, 1e-9),
    child_1_9_case_reduction = 1 - child_1_9_cases_per_100k / pmax(current_child_1_9_cases_per_100k, 1e-9),
    adolescent_case_reduction = 1 - adolescent_cases_per_100k / pmax(current_adolescent_cases_per_100k, 1e-9)
  ) %>%
  ungroup()

programme_burden <- burden %>%
  filter(strategy %in% programme_strategies)

country_order <- burden %>%
  filter(strategy == "current") %>%
  arrange(desc(current_primary_cases_per_100k)) %>%
  pull(country_label_text)

programme_burden <- programme_burden %>%
  mutate(country_label = factor(country_label_text, levels = rev(country_order)))

## Panel A: age-stratum effect matrix -----------------------------------------

endpoint_country <- programme_burden %>%
  transmute(
    country,
    country_label_text,
    country_code,
    strategy,
    strategy_label,
    strategy_label_plot,
    Infant = infant_case_reduction,
    `Child 1–9` = child_1_9_case_reduction,
    `Adolescent 10–17` = adolescent_case_reduction,
    `<18` = primary_case_reduction
  ) %>%
  pivot_longer(
    cols = all_of(endpoint_levels),
    names_to = "endpoint",
    values_to = "relative_case_reduction"
  ) %>%
  mutate(endpoint = factor(endpoint, levels = endpoint_levels))

endpoint_effect_matrix <- endpoint_country %>%
  group_by(strategy, strategy_label, strategy_label_plot, endpoint) %>%
  summarise(
    country_count = n(),
    median_relative_case_reduction = median(relative_case_reduction, na.rm = TRUE),
    q25_relative_case_reduction = interval_quantile(relative_case_reduction, 0.25),
    q75_relative_case_reduction = interval_quantile(relative_case_reduction, 0.75),
    min_relative_case_reduction = min(relative_case_reduction, na.rm = TRUE),
    max_relative_case_reduction = max(relative_case_reduction, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    endpoint = factor(endpoint, levels = endpoint_levels),
    effect_label = lancet_percent(median_relative_case_reduction, accuracy = 1),
    effect_text_colour = if_else(median_relative_case_reduction >= 0.30, "white", lancet_text_colour)
  )

readr::write_csv(
  endpoint_country %>%
    left_join(
      endpoint_effect_matrix %>%
        select(
          strategy,
          endpoint,
          median_relative_case_reduction,
          q25_relative_case_reduction,
          q75_relative_case_reduction,
          min_relative_case_reduction,
          max_relative_case_reduction
        ),
      by = c("strategy", "endpoint")
    ) %>%
    transmute(
      country,
      strategy,
      strategy_label,
      endpoint = as.character(endpoint),
      relative_case_reduction,
      median_relative_case_reduction,
      q25_relative_case_reduction,
      q75_relative_case_reduction,
      min_relative_case_reduction,
      max_relative_case_reduction
    ),
  model_path("outputs", "tables", "figure3a_endpoint_effect_matrix.csv")
)

p3a <- ggplot(endpoint_effect_matrix, aes(endpoint, strategy_label_plot, fill = median_relative_case_reduction)) +
  geom_tile(colour = "white", linewidth = lancet_heatmap_tile_linewidth) +
  geom_text(aes(label = effect_label, colour = effect_text_colour), size = journal_heatmap_cell_text_size_small) +
  scale_x_discrete(labels = endpoint_axis_labels) +
  scale_fill_reduction(
    midpoint = 0,
    limits = c(-0.25, 0.5),
    breaks = c(-0.25, 0, 0.25, 0.50),
    labels = label_lancet_percent(accuracy = 1),
    oob = scales::squish,
    name = "Median reduction",
    guide = guide_lancet_colourbar(barwidth = unit(3.5, "cm"), barheight = unit(0.18, "cm"))
  ) +
  scale_colour_identity() +
  labs(x = NULL, y = NULL, tag = "a") +
  theme_lancet_heatmap(
    base_size = journal_dense_text_size,
    plot_margin = margin(4, 5, 4, 4)
  )

## Panel B: infant-to-<18 endpoint translation gap ----------------------------

endpoint_gap <- programme_burden %>%
  mutate(
    infant_minus_child_adolescent_gap_pp = 100 * (infant_case_reduction - primary_case_reduction)
  )

endpoint_gap_summary <- endpoint_gap %>%
  group_by(strategy, strategy_label, strategy_label_plot) %>%
  summarise(
    country_count = n(),
    median_gap_pp = median(infant_minus_child_adolescent_gap_pp, na.rm = TRUE),
    q25_gap_pp = interval_quantile(infant_minus_child_adolescent_gap_pp, 0.25),
    q75_gap_pp = interval_quantile(infant_minus_child_adolescent_gap_pp, 0.75),
    min_gap_pp = min(infant_minus_child_adolescent_gap_pp, na.rm = TRUE),
    max_gap_pp = max(infant_minus_child_adolescent_gap_pp, na.rm = TRUE),
    .groups = "drop"
  )

readr::write_csv(
  endpoint_gap %>%
    left_join(
      endpoint_gap_summary %>%
        select(strategy, median_gap_pp, q25_gap_pp, q75_gap_pp, min_gap_pp, max_gap_pp),
      by = "strategy"
    ) %>%
    transmute(
      country,
      strategy,
      strategy_label,
      infant_case_reduction,
      child_adolescent_case_reduction = primary_case_reduction,
      infant_minus_child_adolescent_gap_pp,
      median_gap_pp,
      q25_gap_pp,
      q75_gap_pp,
      min_gap_pp,
      max_gap_pp
    ),
  model_path("outputs", "tables", "figure3b_infant_to_child_adolescent_gap.csv")
)

p3b <- ggplot() +
  geom_vline(xintercept = 0, linewidth = 0.24, colour = manuscript_colour("pale_grey")) +
  geom_segment(
    data = endpoint_gap_summary,
    aes(x = min_gap_pp, xend = max_gap_pp, y = strategy_label_plot, yend = strategy_label_plot),
    linewidth = 0.26,
    colour = manuscript_colour("pale_grey")
  ) +
  geom_segment(
    data = endpoint_gap_summary,
    aes(x = q25_gap_pp, xend = q75_gap_pp, y = strategy_label_plot, yend = strategy_label_plot),
    linewidth = 1.05,
    colour = manuscript_colour("grey"),
    alpha = 0.74
  ) +
  geom_point(
    data = endpoint_gap,
    aes(infant_minus_child_adolescent_gap_pp, strategy_label_plot, colour = strategy),
    alpha = 0.50,
    size = 1.15,
    position = position_jitter(height = 0.075, width = 0)
  ) +
  geom_point(
    data = endpoint_gap_summary,
    aes(median_gap_pp, strategy_label_plot),
    shape = 23,
    fill = manuscript_colour("black"),
    colour = "white",
    stroke = 0.20,
    size = 2.15
  ) +
  scale_x_continuous(
    breaks = seq(-10, 30, by = 10),
    labels = label_lancet_number(accuracy = 1, suffix = " pp")
  ) +
  scale_colour_manual(values = strategy_colours, guide = "none") +
  coord_cartesian(xlim = c(-12, 30), clip = "off") +
  labs(
    x = "Infant minus <18 reduction",
    y = NULL,
    tag = "b"
  ) +
  theme_lancet_panel(base_size = journal_dense_text_size, plot_margin = margin(4, 4, 4, 5), show_y_grid = TRUE) +
  theme(
    panel.grid.major.x = element_line(linewidth = 0.16, colour = lancet_grid_light_colour)
  )

## Panel C: age contribution to <18 cases averted -----------------------------

population <- read_table("calibration_all_countries.csv") %>%
  mutate(country = stringr::str_replace_all(country, " ", "_")) %>%
  transmute(
    country,
    infant_population = as.numeric(infant_population),
    child_1_9_population = as.numeric(child_1_9_population),
    adolescent_population = as.numeric(adolescent_population),
    child_adolescent_population = as.numeric(child_adolescent_population)
  )

age_contribution_country <- programme_burden %>%
  left_join(population, by = "country") %>%
  mutate(
    infant_averted_per_100k_under18 =
      (current_infant_cases_per_100k - infant_cases_per_100k) *
        infant_population / pmax(child_adolescent_population, 1e-9),
    child_1_9_averted_per_100k_under18 =
      (current_child_1_9_cases_per_100k - child_1_9_cases_per_100k) *
        child_1_9_population / pmax(child_adolescent_population, 1e-9),
    adolescent_averted_per_100k_under18 =
      (current_adolescent_cases_per_100k - adolescent_cases_per_100k) *
        adolescent_population / pmax(child_adolescent_population, 1e-9),
    component_sum_averted_per_100k_under18 =
      infant_averted_per_100k_under18 +
      child_1_9_averted_per_100k_under18 +
      adolescent_averted_per_100k_under18,
    primary_averted_per_100k_under18 = current_primary_cases_per_100k - primary_cases_per_100k,
    component_sum_difference_per_100k =
      component_sum_averted_per_100k_under18 - primary_averted_per_100k_under18
  )

age_contribution_long <- age_contribution_country %>%
  select(
    country,
    strategy,
    strategy_label,
    strategy_label_plot,
    primary_averted_per_100k_under18,
    component_sum_averted_per_100k_under18,
    component_sum_difference_per_100k,
    infant_averted_per_100k_under18,
    child_1_9_averted_per_100k_under18,
    adolescent_averted_per_100k_under18
  ) %>%
  pivot_longer(
    cols = c(
      infant_averted_per_100k_under18,
      child_1_9_averted_per_100k_under18,
      adolescent_averted_per_100k_under18
    ),
    names_to = "age_contribution",
    values_to = "cases_averted_per_100k_under18"
  ) %>%
  mutate(
    age_contribution = recode(
      age_contribution,
      infant_averted_per_100k_under18 = "0–11 m",
      child_1_9_averted_per_100k_under18 = "1–9 y",
      adolescent_averted_per_100k_under18 = "10–17 y"
    ),
    age_contribution = factor(age_contribution, levels = age_contribution_levels)
  )

age_contribution_summary <- age_contribution_long %>%
  group_by(strategy, strategy_label, strategy_label_plot, age_contribution) %>%
  summarise(
    country_count = n(),
    median_cases_averted_per_100k_under18 = median(cases_averted_per_100k_under18, na.rm = TRUE),
    q25_cases_averted_per_100k_under18 = interval_quantile(cases_averted_per_100k_under18, 0.25),
    q75_cases_averted_per_100k_under18 = interval_quantile(cases_averted_per_100k_under18, 0.75),
    .groups = "drop"
  ) %>%
  mutate(age_contribution = factor(age_contribution, levels = age_contribution_levels))

readr::write_csv(
  age_contribution_long %>%
    left_join(
      age_contribution_summary %>%
        select(
          strategy,
          age_contribution,
          median_cases_averted_per_100k_under18,
          q25_cases_averted_per_100k_under18,
          q75_cases_averted_per_100k_under18
        ),
      by = c("strategy", "age_contribution")
    ) %>%
    transmute(
      country,
      strategy,
      strategy_label,
      age_contribution = as.character(age_contribution),
      cases_averted_per_100k_under18,
      primary_averted_per_100k_under18,
      component_sum_averted_per_100k_under18,
      component_sum_difference_per_100k,
      median_cases_averted_per_100k_under18,
      q25_cases_averted_per_100k_under18,
      q75_cases_averted_per_100k_under18
    ),
  model_path("outputs", "tables", "figure3c_age_contribution_averted_cases.csv")
)

p3c <- ggplot(age_contribution_summary, aes(median_cases_averted_per_100k_under18, strategy_label_plot, fill = age_contribution)) +
  geom_vline(xintercept = 0, linewidth = 0.24, colour = manuscript_colour("pale_grey")) +
  geom_col(width = 0.68, colour = "white", linewidth = 0.16, position = position_stack(reverse = TRUE)) +
  scale_x_continuous(
    breaks = c(-10, 0, 25, 50, 75),
    labels = label_lancet_number(accuracy = 1)
  ) +
  scale_fill_manual(
    values = age_contribution_colours,
    name = NULL,
    guide = guide_legend(nrow = 1, byrow = TRUE, override.aes = list(linewidth = 0))
  ) +
  coord_cartesian(xlim = c(-8, 72), clip = "off") +
  labs(
    x = "Cases averted per 100 000 <18",
    y = NULL,
    tag = "c"
  ) +
  theme_lancet_panel(base_size = journal_dense_text_size, plot_margin = margin(4, 5, 4, 4), show_y_grid = TRUE) +
  theme(
    panel.grid.major.x = element_line(linewidth = 0.16, colour = lancet_grid_light_colour),
    legend.position = "top"
  )

## Panel D: age-pattern weighted robustness -----------------------------------

age_pattern <- read_table("lancet_age_pattern_weighted_strategy_summary.csv") %>%
  mutate(
    strategy = as.character(strategy),
    strategy_label = unname(strategy_source_labels[strategy]),
    strategy_label_plot = factor(
      unname(strategy_plot_labels[strategy]),
      levels = rev(strategy_plot_labels[programme_strategies])
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
    country_count = as.numeric(country_count),
    effective_country_weight_sum = as.numeric(effective_country_weight_sum),
    ordering_basis_label = case_when(
      ordering_basis == "all_profiles_unweighted" ~ "All profiles",
      ordering_basis == "age_data_profiles_unweighted" ~ "Age-data profiles",
      ordering_basis == "age_pattern_weighted" ~ "Age-pattern weighted",
      TRUE ~ ordering_basis
    ),
    ordering_basis_label = factor(
      ordering_basis_label,
      levels = c("All profiles", "Age-data profiles", "Age-pattern weighted")
    )
  ) %>%
  filter(strategy %in% programme_strategies, ordering_basis_label %in% levels(ordering_basis_label))

readr::write_csv(
  age_pattern %>%
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
  model_path("outputs", "tables", "figure3d_age_pattern_robustness.csv")
)

age_pattern_basis_colours <- c(
  "All profiles" = manuscript_colour("mid_grey"),
  "Age-data profiles" = manuscript_colour("sky"),
  "Age-pattern weighted" = manuscript_colour("blue")
)

age_pattern_x_range <- range(
  c(
    0,
    age_pattern$weighted_median_primary_case_reduction,
    age_pattern$weighted_iqr_low_primary_case_reduction,
    age_pattern$weighted_iqr_high_primary_case_reduction
  ),
  na.rm = TRUE
)
age_pattern_x_limits <- c(
  floor((age_pattern_x_range[[1]] - 0.01) / 0.05) * 0.05,
  ceiling((age_pattern_x_range[[2]] + 0.01) / 0.05) * 0.05
)
age_pattern_x_breaks <- seq(age_pattern_x_limits[[1]], age_pattern_x_limits[[2]], by = 0.05)

p3d <- ggplot(age_pattern, aes(weighted_median_primary_case_reduction, strategy_label_plot, fill = ordering_basis_label)) +
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
  scale_x_continuous(labels = label_lancet_percent(accuracy = 1), breaks = age_pattern_x_breaks) +
  scale_fill_manual(
    values = age_pattern_basis_colours,
    labels = c("All", "Age-data", "Weighted"),
    name = NULL,
    guide = guide_legend(nrow = 1, byrow = TRUE)
  ) +
  coord_cartesian(xlim = age_pattern_x_limits, clip = "off") +
  labs(
    x = "Median <18 reduction",
    y = NULL,
    tag = "d"
  ) +
  theme_lancet_panel(base_size = journal_dense_text_size, plot_margin = margin(4, 4, 4, 5), show_y_grid = TRUE) +
  theme(
    panel.grid.major.x = element_line(linewidth = 0.16, colour = lancet_grid_light_colour),
    legend.position = "top",
    axis.ticks.y = element_line(linewidth = lancet_axis_linewidth, colour = lancet_text_colour)
  )

## Compose Figure 3 ------------------------------------------------------------

figure3 <- free(p3a) + p3b + p3c + p3d +
  plot_layout(widths = c(1.03, 0.97), heights = c(1.00, 1.00)) &
  theme_lancet_tags()

save_main_figure(figure3, "figure_3_endpoint_dependence", height = 6.65)
cat("Figure 3 saved.\n")
