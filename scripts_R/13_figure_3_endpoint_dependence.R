#!/usr/bin/env Rscript
## Figure 3: age-group outcome translation ------------------------------------
## Layout: (a) Age-group effect matrix
##         (b) Infant-to-pooled outcome gap
##         (c) Age contribution to cases averted among people aged <18 years
##         (d) Profile-specific adolescent-booster effects

args <- commandArgs(FALSE)
file_arg <- sub("^--file=", "", args[grepl("^--file=", args)])
script_dir <- if (length(file_arg) > 0) dirname(normalizePath(file_arg[[1]])) else file.path(getwd(), "scripts_R")
source(file.path(script_dir, "10_shared.R"))

read_table <- function(name) {
  readr::read_csv(model_path("outputs", "tables", name), show_col_types = FALSE)
}

programme_strategies <- c(
  "maternal_immunization",
  "timeliness_only",
  "cocooning_adjunct",
  "targeted_pep_high_risk",
  "pregnancy_tdap_scaleup",
  "adolescent_booster"
)

strategy_source_labels <- c(
  higher_child_coverage = "Coverage",
  timeliness_only = "Routine schedule timeliness",
  maternal_immunization = "Infant-exposure reduction composite",
  cocooning_adjunct = "Close-contact adult adjuncts",
  targeted_pep_high_risk = "Targeted high-risk PEP",
  adolescent_booster = "Adolescent booster scale-up",
  pregnancy_tdap_scaleup = "Pregnancy Tdap scale-up"
)

strategy_plot_labels <- c(
  higher_child_coverage = "Coverage",
  timeliness_only = "Routine schedule\ntimeliness",
  maternal_immunization = "Infant-exposure\nreduction composite",
  cocooning_adjunct = "Close-contact\nadult adjuncts",
  targeted_pep_high_risk = "Targeted high-risk\nPEP",
  adolescent_booster = "Adolescent booster\nscale-up",
  pregnancy_tdap_scaleup = "Pregnancy Tdap\nscale-up"
)

endpoint_levels <- c(
  "Infant cases",
  "Infant hospitalisations",
  "Infant deaths",
  "Children cases",
  "Adolescent cases",
  "All <18 cases"
)
age_stratum_levels <- c("Infant", "Children", "Adolescent")
age_stratum_colours <- c(
  "Infant" = palette_discrete_primary_9[[7]],
  "Children" = palette_discrete_primary_9[[4]],
  "Adolescent" = palette_discrete_primary_9[[8]]
)

age_stratum_display_labels <- c(
  "Infant" = "Infants 0-11 m",
  "Children" = "Children 1-9 y",
  "Adolescent" = "Adolescents 10-17 y"
)

endpoint_axis_labels <- c(
  `Infant cases` = "Infant\ncases",
  `Infant hospitalisations` = "Infant\nhospital-\nisations",
  `Infant deaths` = "Infant\ndeaths",
  `Children cases` = "Children\ncases",
  `Adolescent cases` = "Adolescent\ncases",
  `All <18 cases` = "All <18\ncases"
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
    primary_case_reduction = as.numeric(primary_case_reduction),
    relative_reduction_infant_hospitalizations = as.numeric(relative_reduction_infant_hospitalizations),
    relative_reduction_infant_deaths = as.numeric(relative_reduction_infant_deaths)
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

## Panel A: age-group effect matrix -------------------------------------------

endpoint_country <- programme_burden %>%
  transmute(
    country,
    country_label_text,
    country_code,
    strategy,
    strategy_label,
    strategy_label_plot,
    `Infant cases` = infant_case_reduction,
    `Infant hospitalisations` = relative_reduction_infant_hospitalizations,
    `Infant deaths` = relative_reduction_infant_deaths,
    `Children cases` = child_1_9_case_reduction,
    `Adolescent cases` = adolescent_case_reduction,
    `All <18 cases` = primary_case_reduction
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
      outcome = as.character(endpoint),
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
  geom_vline(xintercept = c(3.5, 5.5), colour = "white", linewidth = 1.05) +
  geom_text(aes(label = effect_label, colour = effect_text_colour), size = journal_heatmap_cell_text_size_small) +
  scale_x_discrete(labels = endpoint_axis_labels) +
  scale_fill_gradientn(
    colours = c(manuscript_colour("light_grey"), manuscript_colour("sky"), manuscript_colour("blue")),
    limits = c(0, 0.50),
    breaks = c(0, 0.25, 0.50),
    labels = label_lancet_percent(accuracy = 1),
    oob = scales::squish,
    name = "Median reduction (%)",
    guide = guide_lancet_colourbar(barwidth = unit(3.5, "cm"), barheight = unit(0.18, "cm"))
  ) +
  scale_colour_identity() +
  labs(x = NULL, y = NULL, tag = "a") +
  theme_lancet_heatmap(
    base_size = journal_dense_text_size,
    plot_margin = margin(4, 5, 4, 4),
    x_size = journal_dense_text_size - 0.7
  )

## Panel B: infant-to-pooled outcome gap --------------------------------------

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
      overall_case_reduction = primary_case_reduction,
      infant_minus_overall_gap_pp = infant_minus_child_adolescent_gap_pp,
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
    aes(
      x = q25_gap_pp,
      xend = q75_gap_pp,
      y = strategy_label_plot,
      yend = strategy_label_plot,
      linetype = "IQR"
    ),
    linewidth = 1.05,
    colour = manuscript_colour("grey"),
    alpha = 0.74
  ) +
  geom_point(
    data = endpoint_gap,
    aes(infant_minus_child_adolescent_gap_pp, strategy_label_plot, colour = strategy, shape = "Profiles"),
    alpha = 0.50,
    size = 1.15,
    position = position_jitter(height = 0.075, width = 0)
  ) +
  geom_point(
    data = endpoint_gap_summary,
    aes(median_gap_pp, strategy_label_plot, shape = "Median"),
    fill = manuscript_colour("black"),
    colour = "white",
    stroke = 0.20,
    size = 2.15
  ) +
  scale_x_continuous(
    breaks = seq(-10, 30, by = 10),
    labels = label_lancet_number(accuracy = 1)
  ) +
  scale_colour_manual(values = strategy_colours, guide = "none") +
  scale_shape_manual(
    values = c("Profiles" = 16, "Median" = 23),
    breaks = c("Profiles", "Median"),
    name = "Estimate",
    guide = guide_legend(
      ncol = 1,
      byrow = TRUE,
      title.position = "top",
      title.hjust = 0,
      keywidth = unit(0.36, "cm"),
      keyheight = unit(0.46, "cm"),
      override.aes = list(
        colour = c(manuscript_colour("mid_grey"), manuscript_colour("black")),
        fill = c(manuscript_colour("mid_grey"), manuscript_colour("black")),
        alpha = c(0.65, 1),
        size = c(1.5, 2.2),
        stroke = c(0, 0.2)
      )
    )
  ) +
  scale_linetype_manual(
    values = c("IQR" = "solid"),
    breaks = "IQR",
    labels = c("IQR" = "Cross-profile IQR"),
    name = "Interval",
    guide = guide_legend(
      order = 2,
      title.position = "top",
      title.hjust = 0,
      keywidth = unit(0.52, "cm"),
      keyheight = unit(0.28, "cm"),
      override.aes = list(
        colour = manuscript_colour("grey"),
        linewidth = 1.05,
        alpha = 0.74
      )
    )
  ) +
  coord_cartesian(xlim = c(-12, 30), clip = "off") +
  labs(
    x = "Infant-case reduction minus\nall <18-case reduction (percentage points)",
    y = NULL,
    tag = "b"
  ) +
  theme_lancet_panel(base_size = journal_dense_text_size, plot_margin = margin(4, 4, 4, 5), show_y_grid = TRUE) +
  theme(
    panel.grid.major.x = element_line(linewidth = 0.16, colour = lancet_grid_light_colour),
    legend.position = "inside",
    legend.position.inside = c(1, 0.01),
    legend.justification = c(1, 0),
    legend.direction = "vertical",
    legend.title = element_text(face = "bold", lineheight = 0.90),
    legend.text = element_text(lineheight = 1.16, margin = margin(t = 3, b = 3)),
    legend.key.height = unit(0.46, "cm"),
    legend.spacing.y = unit(8, "pt"),
    legend.background = element_rect(fill = "#FFFFFFE6", colour = NA),
    legend.box.margin = margin(0, 0, 1, 0),
    legend.margin = margin(3, 0, 3, 0)
  )

## Panel C: age contribution to cases averted among people aged <18 years ------

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
      infant_averted_per_100k_under18 = "Infant",
      child_1_9_averted_per_100k_under18 = "Children",
      adolescent_averted_per_100k_under18 = "Adolescent"
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

p3c_extent <- age_contribution_summary %>%
  group_by(strategy, strategy_label_plot) %>%
  summarise(
    stacked_median = sum(median_cases_averted_per_100k_under18, na.rm = TRUE),
    .groups = "drop"
  )

p3c_x_upper <- ceiling((max(p3c_extent$stacked_median, na.rm = TRUE) + 2) / 5) * 5

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
      age_group = as.character(age_contribution),
      cases_averted_per_100k_aged_under_18 = cases_averted_per_100k_under18,
      overall_cases_averted_per_100k_aged_under_18 = primary_averted_per_100k_under18,
      age_group_sum_averted_per_100k_aged_under_18 = component_sum_averted_per_100k_under18,
      component_sum_difference_per_100k,
      median_cases_averted_per_100k_aged_under_18 = median_cases_averted_per_100k_under18,
      q25_cases_averted_per_100k_aged_under_18 = q25_cases_averted_per_100k_under18,
      q75_cases_averted_per_100k_aged_under_18 = q75_cases_averted_per_100k_under18
    ),
  model_path("outputs", "tables", "figure3c_age_contribution_averted_cases.csv")
)

p3c <- ggplot(age_contribution_summary, aes(median_cases_averted_per_100k_under18, strategy_label_plot, fill = age_contribution)) +
  geom_vline(xintercept = 0, linewidth = 0.24, colour = manuscript_colour("pale_grey")) +
  geom_col(
    width = 0.68,
    colour = "white",
    linewidth = 0.16,
    position = position_stack(reverse = TRUE),
    key_glyph = ggplot2::draw_key_point
  ) +
  scale_x_continuous(
    breaks = seq(0, p3c_x_upper, by = 10),
    labels = label_lancet_number(accuracy = 1)
  ) +
  scale_fill_manual(
    values = age_contribution_colours,
    labels = age_stratum_display_labels,
    name = "Age group",
    guide = guide_legend(
      ncol = 1,
      byrow = TRUE,
      title.position = "top",
      title.hjust = 0,
      keywidth = unit(0.36, "cm"),
      keyheight = unit(0.46, "cm"),
      override.aes = list(shape = 22, size = 3.2, colour = NA, stroke = 0)
    )
  ) +
  coord_cartesian(xlim = c(-1, p3c_x_upper), clip = "off") +
  labs(
    x = "Median annualised symptomatic cases averted\nper 100 000 people aged <18 years",
    y = NULL,
    tag = "c"
  ) +
  theme_lancet_panel(base_size = journal_dense_text_size, plot_margin = margin(4, 5, 4, 4), show_y_grid = TRUE) +
  theme(
    panel.grid.major.x = element_line(linewidth = 0.16, colour = lancet_grid_light_colour),
    legend.position = "inside",
    legend.position.inside = c(1, 0.01),
    legend.justification = c(1, 0),
    legend.direction = "vertical",
    legend.title = element_text(face = "bold", lineheight = 0.90),
    legend.text = element_text(lineheight = 1.16, margin = margin(t = 3, b = 3)),
    legend.key.width = unit(0.36, "cm"),
    legend.key.height = unit(0.46, "cm"),
    legend.spacing.y = unit(8, "pt"),
    legend.background = element_rect(fill = "#FFFFFFE6", colour = NA),
    legend.box.margin = margin(0, 0, 1, 0),
    legend.margin = margin(3, 0, 3, 0)
  )

## Panel D: profile-specific adolescent booster effects ------------------------

booster_effect_profile <- programme_burden %>%
  filter(strategy == "adolescent_booster") %>%
  transmute(
    country,
    country_label_text,
    country_code,
    all_under18_symptomatic_case_reduction = primary_case_reduction,
    adolescent_case_reduction,
    all_under18_cases_per_100k = primary_cases_per_100k,
    adolescent_cases_per_100k,
    current_all_under18_cases_per_100k = current_primary_cases_per_100k,
    current_adolescent_cases_per_100k
  ) %>%
  arrange(desc(adolescent_case_reduction), country_label_text) %>%
  mutate(
    adolescent_effect_rank = row_number(),
    country_label_booster = factor(country_label_text, levels = rev(country_label_text)),
    china_profile = country == "China"
  )

booster_effect_long <- booster_effect_profile %>%
  select(
    country,
    country_label_text,
    country_code,
    country_label_booster,
    china_profile,
    all_under18_symptomatic_case_reduction,
    adolescent_case_reduction
  ) %>%
  pivot_longer(
    cols = c(all_under18_symptomatic_case_reduction, adolescent_case_reduction),
    names_to = "outcome",
    values_to = "case_reduction"
  ) %>%
  mutate(
    outcome_label = recode(
      outcome,
      all_under18_symptomatic_case_reduction = "All <18 symptomatic cases",
      adolescent_case_reduction = "Adolescent cases"
    ),
    outcome_label = factor(outcome_label, levels = c("All <18 symptomatic cases", "Adolescent cases"))
  )

booster_china_labels <- booster_effect_long %>%
  filter(china_profile) %>%
  mutate(
    label = lancet_percent(case_reduction, accuracy = 0.1),
    label_nudge_x = if_else(outcome == "adolescent_case_reduction", 0.018, -0.018),
    label_hjust = if_else(outcome == "adolescent_case_reduction", 0, 1)
  )

readr::write_csv(
  booster_effect_profile %>%
    transmute(
      country,
      country_code,
      adolescent_effect_rank,
      all_under18_symptomatic_case_reduction,
      adolescent_case_reduction,
      all_under18_cases_per_100k,
      adolescent_cases_per_100k,
      current_all_under18_cases_per_100k,
      current_adolescent_cases_per_100k
    ),
  model_path("outputs", "tables", "figure3d_adolescent_booster_profile_effects.csv")
)

booster_x_range <- range(
  c(
    0,
    booster_effect_profile$all_under18_symptomatic_case_reduction,
    booster_effect_profile$adolescent_case_reduction
  ),
  na.rm = TRUE
)
booster_x_limits <- c(
  floor((booster_x_range[[1]] - 0.02) / 0.05) * 0.05,
  ceiling((booster_x_range[[2]] + 0.02) / 0.05) * 0.05
)
booster_x_breaks <- seq(booster_x_limits[[1]], booster_x_limits[[2]], by = 0.10)

p3d <- ggplot() +
  geom_vline(xintercept = 0, linewidth = 0.24, colour = manuscript_colour("pale_grey")) +
  geom_segment(
    data = booster_effect_profile %>% filter(!china_profile),
    aes(
      x = all_under18_symptomatic_case_reduction,
      xend = adolescent_case_reduction,
      y = country_label_booster,
      yend = country_label_booster
    ),
    linewidth = 0.34,
    colour = manuscript_colour("light_grey")
  ) +
  geom_segment(
    data = booster_effect_profile %>% filter(china_profile),
    aes(
      x = all_under18_symptomatic_case_reduction,
      xend = adolescent_case_reduction,
      y = country_label_booster,
      yend = country_label_booster
    ),
    linewidth = 0.55,
    colour = manuscript_colour("black")
  ) +
  geom_point(
    data = booster_effect_long %>% filter(!china_profile),
    aes(case_reduction, country_label_booster, shape = outcome_label),
    size = 1.85,
    stroke = 0.30,
    colour = manuscript_colour("mid_grey"),
    fill = "white",
    alpha = 0.86
  ) +
  geom_point(
    data = booster_effect_long %>% filter(china_profile),
    aes(case_reduction, country_label_booster, shape = outcome_label),
    size = 2.45,
    stroke = 0.35,
    colour = lancet_text_colour,
    fill = lancet_text_colour
  ) +
  geom_text(
    data = booster_china_labels,
    aes(
      x = case_reduction + label_nudge_x,
      y = country_label_booster,
      label = label,
      hjust = label_hjust
    ),
    size = journal_point_label_text_size,
    colour = lancet_text_colour
  ) +
  scale_x_continuous(labels = label_lancet_percent(accuracy = 1), breaks = booster_x_breaks) +
  scale_shape_manual(
    values = c("All <18 symptomatic cases" = 21, "Adolescent cases" = 24),
    labels = c("All <18 cases", "Adolescent cases"),
    name = "Outcome",
    guide = guide_legend(
      ncol = 1,
      byrow = TRUE,
      title.position = "top",
      title.hjust = 0,
      keywidth = unit(0.36, "cm"),
      keyheight = unit(0.46, "cm"),
      override.aes = list(fill = "white", colour = manuscript_colour("grey"), size = 2.2)
    )
  ) +
  coord_cartesian(xlim = booster_x_limits, clip = "off") +
  labs(
    x = "Reduction in symptomatic cases\nunder adolescent booster scale-up (%)",
    y = NULL,
    tag = "d"
  ) +
  theme_lancet_panel(base_size = journal_dense_text_size, plot_margin = margin(4, 8, 4, 5), show_y_grid = TRUE) +
  theme(
    panel.grid.major.x = element_line(linewidth = 0.16, colour = lancet_grid_light_colour),
    legend.position = "inside",
    legend.position.inside = c(1, 0.01),
    legend.justification = c(1, 0),
    legend.direction = "vertical",
    legend.title = element_text(face = "bold", lineheight = 0.90),
    legend.text = element_text(lineheight = 1.16, margin = margin(t = 3, b = 3)),
    legend.key.height = unit(0.46, "cm"),
    legend.spacing.y = unit(8, "pt"),
    legend.background = element_rect(fill = "#FFFFFFE6", colour = NA),
    legend.box.margin = margin(0, 0, 1, 0),
    legend.margin = margin(3, 0, 3, 0),
    axis.ticks.y = element_line(linewidth = lancet_axis_linewidth, colour = lancet_text_colour)
  )

## Compose Figure 3 ------------------------------------------------------------

figure3 <- free(p3a) + p3b + p3c + p3d +
  plot_layout(widths = c(1.03, 0.97), heights = c(1.00, 1.00)) &
  theme_lancet_tags()

save_main_figure(figure3, "figure_3_endpoint_dependence", height = 6.65)
cat("Figure 3 saved.\n")
