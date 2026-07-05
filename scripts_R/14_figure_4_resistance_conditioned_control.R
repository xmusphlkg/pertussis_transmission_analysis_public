#!/usr/bin/env Rscript
## Figure 4: resistance and future vaccine mechanism domains ------------------
## Layout: (a) Mechanism contrasts
##         (b) Resistance-guided management trade-off
##         (c) Residual <18 cases under future vaccine mechanism targets
##         (d) VEinf response curves across resistant-fraction anchors

args <- commandArgs(FALSE)
file_arg <- sub("^--file=", "", args[grepl("^--file=", args)])
script_dir <- if (length(file_arg) > 0) dirname(normalizePath(file_arg[[1]])) else file.path(getwd(), "scripts_R")
source(file.path(script_dir, "10_shared.R"))

suppressPackageStartupMessages({
  library(ggrepel)
})

read_table <- function(name) {
  readr::read_csv(model_path("outputs", "tables", name), show_col_types = FALSE)
}

read_summary <- function(name) {
  readr::read_csv(model_path("outputs", "summaries", name), show_col_types = FALSE)
}

bounded_proportional_response <- function(current, scenario) {
  denominator <- pmax(current, scenario, na.rm = FALSE)
  dplyr::case_when(
    !is.finite(current) | !is.finite(scenario) ~ NA_real_,
    denominator > 0 ~ (current - scenario) / denominator,
    TRUE ~ 0
  )
}

pct <- label_lancet_percent(accuracy = 1)

case_benefit_colour <- manuscript_colour("blue")
resistance_colour <- manuscript_colour("vermillion")
management_colour <- manuscript_colour("purple")
pep_restored_colour <- manuscript_colour("green")
lower_reach_colour <- manuscript_discrete_core[["deep_teal"]]

vaccine_profile_labels <- c(
  no_vaccine = "No vaccine",
  symptom_protective = "Current aP",
  infection_blocking = "Inf.-blocking",
  transmission_blocking = "Trans.-blocking",
  next_generation = "High-blocking"
)

vaccine_profile_axis_labels <- c(
  "No vaccine" = "No vaccine",
  "Current aP" = "Current acellular-\npertussis-like",
  "Inf.-blocking" = "Infection-\nblocking",
  "Trans.-blocking" = "Transmission-\nblocking",
  "High-blocking" = "High-blocking",
  "Combined stress test" = "Combined future\nstress test"
)

vaccine_target_levels <- c("infection_blocking", "transmission_blocking", "next_generation")
vaccine_display_levels <- c("symptom_protective", vaccine_target_levels)
vaccine_display_labels <- unname(vaccine_profile_labels[vaccine_display_levels])
residual_display_levels <- c(vaccine_display_labels, "Combined stress test")
vaccine_legend_levels <- residual_display_levels

vaccine_outcome_colours <- c(
  "Current aP" = manuscript_vaccine_profile_colours[["Current aP"]],
  "Inf.-blocking" = manuscript_vaccine_profile_colours[["Inf.-blocking"]],
  "Trans.-blocking" = manuscript_vaccine_profile_colours[["Trans.-blocking"]],
  "High-blocking" = manuscript_vaccine_profile_colours[["Upper-bound"]],
  "Combined stress test" = manuscript_colour("black")
)

scale_colour_vaccine_profiles <- function(guide = "none") {
  scale_colour_manual(
    values = vaccine_outcome_colours,
    limits = unname(vaccine_legend_levels),
    breaks = unname(vaccine_legend_levels),
    drop = FALSE,
    name = "Vaccine profile",
    guide = guide
  )
}

decision_frontier <- read_table("lancet_child_adolescent_decision_frontier.csv") %>%
  mutate(
    country = stringr::str_replace_all(country, " ", "_"),
    country_label_text = format_country(country),
    country_code = unname(country_codes[country]),
    strategy = as.character(strategy),
    primary_case_reduction = as.numeric(primary_case_reduction),
    primary_cases_per_100k = as.numeric(primary_cases_per_100k),
    current_primary_cases_per_100k = as.numeric(current_primary_cases_per_100k),
    annualized_resistant_infections_per_100k = as.numeric(annualized_resistant_infections_per_100k),
    relative_reduction_resistant_infections = as.numeric(relative_reduction_resistant_infections),
    primary_case_rank_within_constraint = as.numeric(primary_case_rank_within_constraint)
  )

current_resistance <- decision_frontier %>%
  filter(optimization_constraint == "program_plus_resistance", strategy == "current") %>%
  transmute(
    country,
    current_resistant_infections_per_100k = annualized_resistant_infections_per_100k,
    current_child_adolescent_cases_per_100k = current_primary_cases_per_100k
  )

## Panel A: mechanism contrasts -------------------------------------------------

mechanism_order <- c(
  "baseline_full_mechanism",
  "no_resistant_importation",
  "equal_treatment_effect",
  "equal_pep_effect",
  "no_treatment_or_pep_differential",
  "fitness_cost"
)

mechanism_labels <- c(
  baseline_full_mechanism = "Baseline",
  no_resistant_importation = "No import.",
  equal_treatment_effect = "Tx equal",
  equal_pep_effect = "PEP equalized",
  no_treatment_or_pep_differential = "Tx + PEP equal",
  fitness_cost = "Fitness cost"
)

mechanism_data <- read_summary("resistance_mechanism_decomposition_summary.csv") %>%
  mutate(
    resistant_infections_per_100k = resistant_infections / total_population / analysis_years * 1e5
  ) %>%
  group_by(mechanism_scenario) %>%
  summarise(
    median_resistant_infections_per_100k = median(resistant_infections_per_100k, na.rm = TRUE),
    q25_resistant_infections_per_100k = interval_quantile(resistant_infections_per_100k, 0.25),
    q75_resistant_infections_per_100k = interval_quantile(resistant_infections_per_100k, 0.75),
    median_child_adolescent_cases_per_100k = median(annualized_child_adolescent_cases_per_100k, na.rm = TRUE),
    q25_child_adolescent_cases_per_100k = interval_quantile(annualized_child_adolescent_cases_per_100k, 0.25),
    q75_child_adolescent_cases_per_100k = interval_quantile(annualized_child_adolescent_cases_per_100k, 0.75),
    median_end_resistant_fraction = median(resistant_fraction_end, na.rm = TRUE),
    resistant_importation = first(resistant_importation),
    treatment_differential = first(treatment_differential),
    pep_differential = first(pep_differential),
    fitness_R = first(fitness_R),
    interpretation = first(mechanism_interpretation),
    .groups = "drop"
  ) %>%
  mutate(
    scenario = factor(mechanism_scenario, levels = mechanism_order),
    mechanism_label = factor(
      mechanism_labels[as.character(scenario)],
      levels = rev(mechanism_labels[mechanism_order])
    ),
    end_fraction_label = lancet_percent(median_end_resistant_fraction, accuracy = 1)
  ) %>%
  filter(!is.na(scenario), is.finite(median_resistant_infections_per_100k)) %>%
  arrange(scenario)

readr::write_csv(
  mechanism_data %>%
    transmute(
      scenario = as.character(scenario),
      mechanism_label = as.character(mechanism_label),
      median_resistant_infections_per_100k,
      q25_resistant_infections_per_100k,
      q75_resistant_infections_per_100k,
      median_child_adolescent_cases_per_100k,
      q25_child_adolescent_cases_per_100k,
      q75_child_adolescent_cases_per_100k,
      median_end_resistant_fraction,
      resistant_importation,
      treatment_differential,
      pep_differential,
      fitness_R,
      interpretation
    ),
  model_path("outputs", "tables", "figure4a_resistance_mechanism_contrasts.csv")
)

p4a_resistant <- ggplot(mechanism_data, aes(median_resistant_infections_per_100k, mechanism_label)) +
  geom_segment(
    aes(
      x = q25_resistant_infections_per_100k,
      xend = q75_resistant_infections_per_100k,
      yend = mechanism_label
    ),
    linewidth = 0.38,
    colour = lancet_grid_colour,
    lineend = "round"
  ) +
  geom_point(
    aes(fill = median_end_resistant_fraction),
    shape = 21,
    size = 2.6,
    colour = manuscript_colour("black"),
    stroke = 0.25
  ) +
  geom_text(
    aes(label = end_fraction_label),
    nudge_x = 0.10,
    hjust = 0,
    size = journal_point_label_text_size,
    colour = manuscript_colour("black")
  ) +
  scale_x_log10(
    breaks = c(1, 10, 1000),
    labels = label_lancet_comma(accuracy = 1),
    expand = expansion(mult = c(0.04, 0.24))
  ) +
  scale_fill_gradientn(
    colours = c(manuscript_colour("light_grey"), manuscript_colour("pale_orange"), resistance_colour),
    limits = c(0, 1),
    guide = "none"
  ) +
  coord_cartesian(xlim = c(1, 3000), clip = "off") +
  labs(
    x = "Resistant infections\nper 100 000/y (log)",
    y = NULL,
    tag = "a"
  ) +
  theme_lancet_panel(base_size = journal_compact_text_size, plot_margin = margin(4, 5, 4, 4), show_y_grid = TRUE) +
  theme(
    axis.ticks.y = element_blank(),
    panel.grid.major.x = element_line(linewidth = 0.16, colour = lancet_grid_light_colour)
  )

p4a_cases <- ggplot(mechanism_data, aes(median_child_adolescent_cases_per_100k, mechanism_label)) +
  geom_segment(
    aes(
      x = q25_child_adolescent_cases_per_100k,
      xend = q75_child_adolescent_cases_per_100k,
      yend = mechanism_label
    ),
    linewidth = 0.38,
    colour = lancet_grid_colour,
    lineend = "round"
  ) +
  geom_point(
    size = 2.4,
    colour = manuscript_colour("black"),
    fill = case_benefit_colour,
    shape = 21,
    stroke = 0.25
  ) +
  scale_x_continuous(
    breaks = c(100, 500),
    labels = label_lancet_comma(accuracy = 1),
    expand = expansion(mult = c(0.04, 0.08))
  ) +
  coord_cartesian(xlim = c(70, 850), clip = "off") +
  labs(
    x = "<18 cases\nper 100 000/y",
    y = NULL
  ) +
  theme_lancet_panel(base_size = journal_compact_text_size, plot_margin = margin(4, 4, 4, 2), show_y_grid = TRUE) +
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    panel.grid.major.x = element_line(linewidth = 0.16, colour = lancet_grid_light_colour)
  )

p4a <- wrap_elements(
  p4a_resistant + p4a_cases +
    plot_layout(widths = c(1.08, 0.92))
)

## Panel B: resistance-management trade-off -------------------------------------

resistance_management <- decision_frontier %>%
  filter(optimization_constraint == "program_plus_resistance", strategy == "resistance_guided_treatment") %>%
  select(
    country,
    country_label_text,
    country_code,
    primary_case_reduction,
    primary_cases_per_100k,
    relative_reduction_resistant_infections,
    residual_resistant_infections_per_100k = annualized_resistant_infections_per_100k
  ) %>%
  left_join(current_resistance, by = "country") %>%
  mutate(
    resistance_reduction_estimable = is.finite(relative_reduction_resistant_infections),
    resistant_reduction_plot = if_else(resistance_reduction_estimable, relative_reduction_resistant_infections, 0),
    current_resistant_infections_plot = pmax(current_resistant_infections_per_100k, 1)
  )

readr::write_csv(
  resistance_management %>%
    transmute(
      country,
      country_code,
      current_resistant_infections_per_100k,
      residual_resistant_infections_per_100k,
      residual_child_adolescent_cases_per_100k = primary_cases_per_100k,
      primary_case_reduction,
      relative_reduction_resistant_infections,
      resistance_reduction_estimable
    ),
  model_path("outputs", "tables", "figure4b_resistance_management_tradeoff.csv")
)

pal_breaks <- c(1, 10, 100, 1000)
pal_limits <- c(1, 3500)

p4b <- ggplot(resistance_management, aes(primary_case_reduction, resistant_reduction_plot)) +
  geom_hline(yintercept = 0, linewidth = 0.22, colour = "grey72") +
  geom_vline(xintercept = 0, linewidth = 0.22, colour = "grey72") +
  geom_abline(slope = 1, intercept = 0, linewidth = 0.22, linetype = "22", colour = manuscript_colour("mid_grey")) +
  geom_point(
    data = filter(resistance_management, resistance_reduction_estimable),
    aes(fill = current_resistant_infections_plot),
    shape = 21,
    size = 2.8,
    colour = manuscript_colour("black"),
    stroke = 0.24,
    alpha = 0.88
  ) +
  geom_point(
    data = filter(resistance_management, !resistance_reduction_estimable),
    shape = 21,
    size = 2.0,
    fill = "white",
    colour = manuscript_colour("mid_grey"),
    stroke = 0.45
  ) +
  ggrepel::geom_text_repel(
    aes(label = country_code),
    size = journal_direct_label_text_size,
    fontface = "bold",
    seed = 4,
    box.padding = 0.16,
    point.padding = 0.08,
    segment.size = 0.12,
    min.segment.length = 0,
    max.overlaps = Inf,
    show.legend = FALSE
  ) +
  scale_x_continuous(labels = pct, breaks = seq(0, 0.8, by = 0.2)) +
  scale_y_continuous(labels = pct, breaks = seq(0, 0.8, by = 0.2)) +
  scale_fill_gradientn(
    colours = c(manuscript_colour("light_grey"), manuscript_colour("pale_orange"), resistance_colour),
    trans = "log10",
    breaks = pal_breaks,
    limits = pal_limits,
    labels = label_lancet_comma(accuracy = 1),
    name = "Current resistant infections per 100 000/y",
    guide = guide_colourbar(
      title.position = "top",
      barwidth = unit(4.2, "cm"),
      barheight = unit(0.18, "cm"),
      ticks.linewidth = 0.20,
      order = 1
    )
  ) +
  coord_cartesian(xlim = c(-0.02, 0.80), ylim = c(-0.05, 0.82), clip = "off") +
  labs(
    x = "<18 case reduction",
    y = "Resistant-infection\nreduction",
    tag = "b"
  ) +
  theme_lancet_panel(base_size = journal_base_text_size, plot_margin = margin(4, 5, 4, 4), show_y_grid = TRUE) +
  theme(
    panel.grid.major.x = element_line(linewidth = 0.16, colour = lancet_grid_light_colour),
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.title = element_text(size = journal_compact_text_size - 0.2, face = "bold", lineheight = 0.9),
    legend.text = element_text(size = journal_compact_text_size - 0.5),
    legend.key.width = unit(0.34, "cm")
  )

## Near-term implementation sensitivity source-data diagnostic -------------------
## Retained for appendix robustness/source-data use; not included in main Figure 4.

implementation_summary <- read_summary("treatment_implementation_sensitivity_summary.csv")

near_term_current <- implementation_summary %>%
  filter(scenario == "current_near_term") %>%
  transmute(
    country,
    current_child_adolescent_cases = total_child_adolescent_cases,
    current_resistant_infections = resistant_infections
  )

implementation_country <- implementation_summary %>%
  left_join(near_term_current, by = "country") %>%
  mutate(
    raw_child_adolescent_case_reduction = 1 - total_child_adolescent_cases / current_child_adolescent_cases,
    child_adolescent_case_reduction = bounded_proportional_response(
      current_child_adolescent_cases,
      total_child_adolescent_cases
    ),
    raw_resistant_infection_reduction = if_else(
      current_resistant_infections > 0,
      1 - resistant_infections / current_resistant_infections,
      NA_real_
    ),
    resistant_infection_reduction = if_else(
      current_resistant_infections > 0,
      bounded_proportional_response(current_resistant_infections, resistant_infections),
      NA_real_
    )
  )

implementation_sensitivity <- implementation_country %>%
  group_by(scenario, implementation_uptake, pep_restored, pep_coverage_multiplier, implementation_note) %>%
  summarise(
    median_child_adolescent_case_reduction = median(child_adolescent_case_reduction, na.rm = TRUE),
    q25_child_adolescent_case_reduction = interval_quantile(child_adolescent_case_reduction, 0.25),
    q75_child_adolescent_case_reduction = interval_quantile(child_adolescent_case_reduction, 0.75),
    median_resistant_infection_reduction = median(resistant_infection_reduction, na.rm = TRUE),
    q25_resistant_infection_reduction = interval_quantile(resistant_infection_reduction, 0.25),
    q75_resistant_infection_reduction = interval_quantile(resistant_infection_reduction, 0.75),
    .groups = "drop"
  ) %>%
  mutate(
    pep_assumption = case_when(
      pep_restored == "yes" ~ "Treatment + restored PEP effect",
      TRUE ~ "Guided treatment only"
    ),
    reach_status = if_else(pep_coverage_multiplier < 1, "Lower PEP reach", "Baseline PEP reach"),
    implementation_group = case_when(
      reach_status == "Lower PEP reach" ~ "Lower PEP reach",
      pep_assumption == "Treatment + restored PEP effect" ~ "Tx + restored PEP",
      TRUE ~ "Guided tx only"
    ),
    uptake_offset = case_when(
      scenario == "current_near_term" ~ 0,
      pep_assumption == "Guided treatment only" & reach_status == "Baseline PEP reach" ~ -0.018,
      pep_assumption == "Treatment + restored PEP effect" & reach_status == "Baseline PEP reach" ~ 0.018,
      reach_status == "Lower PEP reach" ~ 0.046,
      TRUE ~ 0
    ),
    uptake_plot = implementation_uptake + uptake_offset
  )

reduction_axis_floor <- -1.0
reduction_axis_ceiling <- 0.85

implementation_plot <- bind_rows(
  implementation_sensitivity %>%
    transmute(
      scenario,
      uptake = implementation_uptake,
      uptake_offset,
      uptake_plot,
      pep_assumption,
      reach_status,
      implementation_group,
      metric = "<18 cases",
      median_reduction = median_child_adolescent_case_reduction,
      q25_reduction = q25_child_adolescent_case_reduction,
      q75_reduction = q75_child_adolescent_case_reduction,
      implementation_note
    ),
  implementation_sensitivity %>%
    transmute(
      scenario,
      uptake = implementation_uptake,
      uptake_offset,
      uptake_plot,
      pep_assumption,
      reach_status,
      implementation_group,
      metric = "Resistant infections",
      median_reduction = median_resistant_infection_reduction,
      q25_reduction = q25_resistant_infection_reduction,
      q75_reduction = q75_resistant_infection_reduction,
      implementation_note
    )
) %>%
  mutate(
    metric = factor(metric, levels = c("<18 cases", "Resistant infections")),
    pep_assumption = factor(pep_assumption, levels = c("Guided treatment only", "Treatment + restored PEP effect")),
    reach_status = factor(reach_status, levels = c("Baseline PEP reach", "Lower PEP reach")),
    implementation_group = factor(
      implementation_group,
      levels = c("Guided tx only", "Tx + restored PEP", "Lower PEP reach")
    ),
    q25_reduction_plot = pmax(q25_reduction, reduction_axis_floor),
    q75_reduction_plot = pmin(q75_reduction, reduction_axis_ceiling),
    q25_truncated = q25_reduction < reduction_axis_floor,
    q75_truncated = q75_reduction > reduction_axis_ceiling,
    effect_measure = "bounded_proportional_response"
  )

readr::write_csv(
  implementation_plot,
  model_path("outputs", "tables", "figure4_resistance_implementation_sensitivity_appendix.csv")
)

make_implementation_panel <- function(metric_name, panel_tag, y_label = NULL) {
  panel_data <- filter(implementation_plot, metric == metric_name)

  ggplot(
    panel_data,
    aes(uptake_plot, median_reduction)
  ) +
    geom_hline(yintercept = 0, linewidth = 0.22, colour = "grey72") +
    geom_line(
      aes(colour = implementation_group, group = implementation_group),
      linewidth = 0.38,
      alpha = 0.70,
      show.legend = FALSE
    ) +
    geom_linerange(
      aes(ymin = q25_reduction_plot, ymax = q75_reduction_plot, colour = implementation_group),
      linewidth = 0.34,
      alpha = 0.58,
      show.legend = FALSE
    ) +
    geom_point(
      aes(colour = implementation_group),
      size = 1.9,
      shape = 16,
      alpha = 0.92,
      stroke = 0.52
    ) +
    scale_x_continuous(labels = pct, breaks = c(0, 0.25, 0.5, 0.75, 1)) +
    scale_y_continuous(labels = pct, breaks = c(-1.0, -0.5, 0, 0.5, 0.8)) +
    scale_colour_manual(
      values = c(
        "Guided tx only" = management_colour,
        "Tx + restored PEP" = pep_restored_colour,
        "Lower PEP reach" = lower_reach_colour
      ),
      name = "Implementation group",
      guide = guide_legend(
        title.position = "top",
        nrow = 1,
        override.aes = list(shape = 16, alpha = 1, size = 2.1)
      )
    ) +
    coord_cartesian(xlim = c(-0.07, 1.17), ylim = c(reduction_axis_floor, reduction_axis_ceiling), clip = "on") +
    labs(
      title = metric_name,
      x = "Resistance-guided testing/treatment uptake",
      y = y_label,
      tag = panel_tag
    ) +
    theme_lancet_panel(base_size = journal_compact_text_size, plot_margin = margin(3, 4, 3, 4), show_y_grid = TRUE) +
    theme(
      panel.grid.major.x = element_line(linewidth = 0.16, colour = lancet_grid_light_colour),
      plot.title = element_text(face = "bold", hjust = 0.5, size = journal_base_text_size)
    )
}

implementation_cases_panel <- make_implementation_panel("<18 cases", "c", "Bounded response")
implementation_resistance_panel <- make_implementation_panel("Resistant infections", "d", NULL)

## Panels C-D: future vaccine mechanism targets --------------------------------

current_vaccine_reference <- vaccine_summary %>%
  filter(as.character(scenario) == "symptom_protective") %>%
  transmute(
    country = stringr::str_replace_all(country, " ", "_"),
    current_infections_per_100k = as.numeric(annualized_infections_per_100k),
    current_primary_cases_per_100k = as.numeric(annualized_child_adolescent_cases_per_100k)
  )

vaccine_mechanism_outcomes <- vaccine_summary %>%
  filter(as.character(scenario) %in% vaccine_display_levels) %>%
  mutate(
    country = stringr::str_replace_all(country, " ", "_"),
    scenario = as.character(scenario)
  ) %>%
  left_join(current_vaccine_reference, by = "country") %>%
  mutate(
    country_code = unname(country_codes[country]),
    scenario_label = factor(unname(vaccine_profile_labels[scenario]), levels = vaccine_display_labels),
    primary_cases_per_100k = as.numeric(annualized_child_adolescent_cases_per_100k),
    primary_case_reduction = 1 - primary_cases_per_100k / pmax(current_primary_cases_per_100k, 1e-9),
    relative_reduction_total_infections = 1 - as.numeric(annualized_infections_per_100k) / pmax(current_infections_per_100k, 1e-9)
  ) %>%
  filter(!is.na(primary_case_reduction), !is.na(relative_reduction_total_infections), !is.na(primary_cases_per_100k))

combined_stress_outcomes <- read_table("lancet_child_adolescent_strategy_burden.csv") %>%
  mutate(
    country = stringr::str_replace_all(country, " ", "_"),
    country_code = unname(country_codes[country]),
    strategy = as.character(strategy),
    primary_cases_per_100k = as.numeric(primary_cases_per_100k)
  ) %>%
  filter(strategy == "combined_strategy") %>%
  transmute(
    country,
    country_code,
    scenario = strategy,
    scenario_label = "Combined stress test",
    primary_cases_per_100k,
    outcome_type = "Upper-bound stress test"
  )

vaccine_residual <- vaccine_mechanism_outcomes %>%
  transmute(
    country,
    country_code,
    scenario,
    scenario_label = as.character(scenario_label),
    primary_cases_per_100k,
    outcome_type = "Vaccine mechanism"
  ) %>%
  bind_rows(combined_stress_outcomes) %>%
  mutate(
    scenario_label = factor(scenario_label, levels = rev(residual_display_levels)),
    outcome_type = factor(outcome_type, levels = c("Vaccine mechanism", "Upper-bound stress test"))
  )

vaccine_burden_summary <- vaccine_residual %>%
  group_by(scenario_label) %>%
  summarise(
    median = median(primary_cases_per_100k, na.rm = TRUE),
    q25 = interval_quantile(primary_cases_per_100k, 0.25),
    q75 = interval_quantile(primary_cases_per_100k, 0.75),
    q025 = interval_quantile(primary_cases_per_100k, 0.025),
    q975 = interval_quantile(primary_cases_per_100k, 0.975),
    .groups = "drop"
  )

readr::write_csv(
  vaccine_residual %>%
    mutate(scenario_label = as.character(scenario_label)) %>%
    select(country, country_code, scenario, scenario_label, outcome_type, primary_cases_per_100k) %>%
    left_join(vaccine_burden_summary, by = "scenario_label"),
  model_path("outputs", "tables", "figure4c_future_vaccine_residual_burden.csv")
)

p4c <- ggplot(vaccine_residual, aes(primary_cases_per_100k, scenario_label, colour = scenario_label)) +
  geom_errorbar(
    data = vaccine_burden_summary,
    aes(x = median, xmin = q025, xmax = q975, y = scenario_label),
    inherit.aes = FALSE,
    orientation = "y",
    width = 0.28,
    linewidth = 0.25,
    colour = manuscript_colour("pale_grey")
  ) +
  geom_errorbar(
    data = vaccine_burden_summary,
    aes(x = median, xmin = q25, xmax = q75, y = scenario_label),
    inherit.aes = FALSE,
    orientation = "y",
    width = 0,
    linewidth = 0.72,
    colour = manuscript_colour("grey")
  ) +
  geom_point(
    data = vaccine_residual %>% filter(outcome_type == "Vaccine mechanism"),
    size = 1.50,
    alpha = 0.76,
    position = position_jitter(height = 0.11, width = 0)
  ) +
  geom_point(
    data = vaccine_residual %>% filter(outcome_type == "Upper-bound stress test"),
    inherit.aes = FALSE,
    aes(primary_cases_per_100k, scenario_label),
    shape = 21,
    size = 1.80,
    fill = "white",
    colour = manuscript_colour("black"),
    stroke = 0.35,
    alpha = 0.82,
    position = position_jitter(height = 0.11, width = 0)
  ) +
  geom_point(
    data = vaccine_burden_summary,
    aes(x = median, y = scenario_label),
    inherit.aes = FALSE,
    shape = 23,
    size = 2.7,
    fill = "black",
    colour = "white",
    stroke = 0.20
  ) +
  scale_x_log10(
    breaks = c(0.1, 1, 10, 100, 1000),
    labels = lancet_midline_decimal(c("0.1", "1", "10", "100", "1,000"))
  ) +
  scale_y_discrete(labels = vaccine_profile_axis_labels) +
  scale_colour_vaccine_profiles() +
  labs(
    x = "Residual <18 cases per 100 000/year\n(log scale; lower is better)",
    y = NULL,
    tag = "c"
  ) +
  theme_lancet(base_size = journal_compact_text_size) +
  theme(axis.text.y = element_text(lineheight = 0.95))

veinf_resistance_levels <- c(0, 0.50, 1.00)

veinf_grid <- read_summary("veinf_resistance_grid_summary.csv") %>%
  transmute(
    country = stringr::str_replace_all(country, " ", "_"),
    grid_VE_inf = round(as.numeric(grid_VE_inf), 1),
    grid_resistance_prevalence = round(as.numeric(grid_resistance_prevalence), 2),
    primary_cases_per_100k = as.numeric(annualized_child_adolescent_cases_per_100k),
    all_infections_per_100k = as.numeric(annualized_infections_per_100k)
  ) %>%
  filter(
    grid_resistance_prevalence %in% veinf_resistance_levels,
    grid_VE_inf >= 0,
    grid_VE_inf <= 0.6
  )

veinf_reference <- veinf_grid %>%
  filter(grid_VE_inf == 0) %>%
  transmute(
    country,
    grid_resistance_prevalence,
    reference_primary_cases_per_100k = primary_cases_per_100k,
    reference_all_infections_per_100k = all_infections_per_100k
  )

veinf_response <- veinf_grid %>%
  left_join(veinf_reference, by = c("country", "grid_resistance_prevalence")) %>%
  mutate(
    resistance_label = factor(
      paste0(lancet_percent(grid_resistance_prevalence, accuracy = 1), " resistant"),
      levels = paste0(lancet_percent(veinf_resistance_levels, accuracy = 1), " resistant")
    ),
    child_adolescent_case_reduction_vs_veinf0 =
      1 - primary_cases_per_100k / pmax(reference_primary_cases_per_100k, 1e-9),
    all_infection_reduction_vs_veinf0 =
      1 - all_infections_per_100k / pmax(reference_all_infections_per_100k, 1e-9)
  ) %>%
  filter(
    is.finite(child_adolescent_case_reduction_vs_veinf0),
    is.finite(all_infection_reduction_vs_veinf0)
  )

veinf_response_summary <- veinf_response %>%
  group_by(grid_resistance_prevalence, resistance_label, grid_VE_inf) %>%
  summarise(
    median_case_reduction = median(child_adolescent_case_reduction_vs_veinf0, na.rm = TRUE),
    q25_case_reduction = interval_quantile(child_adolescent_case_reduction_vs_veinf0, 0.25),
    q75_case_reduction = interval_quantile(child_adolescent_case_reduction_vs_veinf0, 0.75),
    median_all_infection_reduction = median(all_infection_reduction_vs_veinf0, na.rm = TRUE),
    q25_all_infection_reduction = interval_quantile(all_infection_reduction_vs_veinf0, 0.25),
    q75_all_infection_reduction = interval_quantile(all_infection_reduction_vs_veinf0, 0.75),
    .groups = "drop"
  )

veinf_response_source <- veinf_response %>%
  left_join(veinf_response_summary, by = c("grid_resistance_prevalence", "resistance_label", "grid_VE_inf")) %>%
  arrange(grid_resistance_prevalence, grid_VE_inf, country) %>%
  transmute(
    country,
    resistance_prevalence = grid_resistance_prevalence,
    resistance_label = as.character(resistance_label),
    VE_inf = grid_VE_inf,
    child_adolescent_cases_per_100k = primary_cases_per_100k,
    reference_child_adolescent_cases_per_100k = reference_primary_cases_per_100k,
    child_adolescent_case_reduction_vs_veinf0,
    all_infections_per_100k,
    reference_all_infections_per_100k,
    all_infection_reduction_vs_veinf0,
    median_child_adolescent_case_reduction = median_case_reduction,
    q25_child_adolescent_case_reduction = q25_case_reduction,
    q75_child_adolescent_case_reduction = q75_case_reduction,
    median_all_infection_reduction,
    q25_all_infection_reduction,
    q75_all_infection_reduction
  )

readr::write_csv(
  veinf_response_source,
  model_path("outputs", "tables", "figure4d_future_veinf_response_curve.csv")
)

veinf_response_colours <- c(
  "0% resistant" = manuscript_colour("mid_grey"),
  "50% resistant" = manuscript_colour("orange"),
  "100% resistant" = manuscript_colour("vermillion")
)

veinf_response_labels <- c(
  "0% resistant" = "0%",
  "50% resistant" = "50%",
  "100% resistant" = "100%"
)

veinf_response_linetypes <- c(
  "0% resistant" = "solid",
  "50% resistant" = "dashed",
  "100% resistant" = "solid"
)

p4d <- ggplot(
  veinf_response_summary,
  aes(grid_VE_inf, median_case_reduction, colour = resistance_label, fill = resistance_label)
) +
  geom_ribbon(
    aes(ymin = q25_case_reduction, ymax = q75_case_reduction),
    alpha = 0.09,
    colour = NA
  ) +
  geom_line(aes(linetype = resistance_label), linewidth = 0.62) +
  geom_point(size = 1.75, stroke = 0) +
  scale_x_continuous(
    breaks = seq(0, 0.6, by = 0.2),
    labels = label_lancet_percent(accuracy = 1),
    limits = c(0, 0.68),
    expand = expansion(mult = c(0, 0))
  ) +
  scale_y_continuous(
    breaks = seq(0, 0.8, by = 0.2),
    labels = label_lancet_percent(accuracy = 1),
    limits = c(0, 0.82),
    expand = expansion(mult = c(0, 0.02))
  ) +
  scale_colour_manual(
    values = veinf_response_colours,
    breaks = names(veinf_response_colours),
    labels = veinf_response_labels,
    name = "Resistant fraction",
    guide = guide_legend(
      nrow = 1,
      byrow = TRUE,
      override.aes = list(alpha = 1, linewidth = 0.7, size = 2)
    )
  ) +
  scale_linetype_manual(
    values = veinf_response_linetypes,
    breaks = names(veinf_response_linetypes),
    labels = veinf_response_labels,
    name = "Resistant fraction",
    guide = guide_legend(nrow = 1, byrow = TRUE)
  ) +
  scale_fill_manual(values = veinf_response_colours, guide = "none") +
  coord_cartesian(clip = "off") +
  labs(
    x = expression(paste("Vaccine infectiousness effect (", VE[inf], ")")),
    y = expression(atop("Relative reduction in <18 cases", paste("vs no ", VE[inf], " effect"))),
    tag = "d"
  ) +
  theme_lancet(base_size = journal_compact_text_size) +
  theme(
    panel.grid.major.x = element_line(linewidth = 0.16, colour = lancet_grid_light_colour),
    plot.margin = margin(2, 6, 4, 4),
    legend.position = "top",
    legend.direction = "horizontal",
    legend.title = element_text(face = "bold", size = journal_compact_text_size, colour = lancet_text_colour),
    legend.text = element_text(size = journal_compact_text_size, colour = lancet_text_colour),
    legend.key.size = unit(0.26, "cm"),
    legend.spacing.x = unit(3, "pt")
  )

## Compose Figure 4 ------------------------------------------------------------

resistance_row <- free(p4a) + p4b +
  plot_layout(widths = c(1.16, 1.00))

future_vaccine_row <- p4c + p4d +
  plot_layout(widths = c(1.04, 0.96))

figure4 <- resistance_row / future_vaccine_row +
  plot_layout(heights = c(1.0, 1.05)) &
  theme_lancet_tags()

save_main_figure(figure4, "figure_4_resistance_conditioned_control", height = 7.0)
cat("Figure 4 saved.\n")
