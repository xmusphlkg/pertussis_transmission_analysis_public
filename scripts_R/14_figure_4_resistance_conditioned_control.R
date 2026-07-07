#!/usr/bin/env Rscript
## Figure 4: resistance and future vaccine mechanism scenarios ----------------
## Layout: (a) Mechanism contrasts
##         (b) Resistance-guided management pathway
##         (c) Remaining cases under future vaccine mechanism targets
##         (d) infectiousness-effect thresholds across resistant-strain seeding levels

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
figure4_country_codes <- country_codes
figure4_country_codes[["United_Kingdom"]] <- "GBR"

vaccine_profile_labels <- c(
  no_vaccine = "No vaccine",
  symptom_protective = "Current aP-like mechanism scenario",
  infection_blocking = "Infection-blocking mechanism",
  transmission_blocking = "Transmission-blocking mechanism",
  next_generation = "High transmission-blocking mechanism"
)

vaccine_profile_axis_labels <- c(
  "No vaccine" = "No vaccine",
  "Current aP-like mechanism scenario" = "Current aP-like\nmechanism scenario",
  "Infection-blocking mechanism" = "Infection-blocking\nmechanism",
  "Transmission-blocking mechanism" = "Transmission-blocking\nmechanism",
  "High transmission-blocking mechanism" = "High transmission-\nblocking mechanism",
  "Combined future upper-bound scenario" = "Combined future\nupper-bound scenario"
)

vaccine_target_levels <- c("infection_blocking", "transmission_blocking", "next_generation")
vaccine_display_levels <- c("symptom_protective", vaccine_target_levels)
vaccine_display_labels <- unname(vaccine_profile_labels[vaccine_display_levels])
residual_display_levels <- c(vaccine_display_labels, "Combined future upper-bound scenario")
vaccine_legend_levels <- residual_display_levels

vaccine_outcome_colours <- c(
  "Current aP-like mechanism scenario" = manuscript_vaccine_profile_colours[["Current aP"]],
  "Infection-blocking mechanism" = manuscript_vaccine_profile_colours[["Inf.-blocking"]],
  "Transmission-blocking mechanism" = manuscript_vaccine_profile_colours[["Trans.-blocking"]],
  "High transmission-blocking mechanism" = manuscript_vaccine_profile_colours[["Upper-bound"]],
  "Combined future upper-bound scenario" = manuscript_colour("black")
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
    country_code = unname(figure4_country_codes[country]),
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
  no_resistant_importation = "Remove further resistant importation",
  equal_treatment_effect = "Equalise treatment benefit",
  equal_pep_effect = "Equalise resistant-strain PEP effectiveness",
  no_treatment_or_pep_differential = "Equalise treatment and PEP effects",
  fitness_cost = "Add resistant-strain fitness cost"
)

mechanism_plot_order <- c(
  "no_resistant_importation",
  "equal_treatment_effect",
  "equal_pep_effect",
  "no_treatment_or_pep_differential",
  "fitness_cost"
)

mechanism_axis_labels <- c(
  "Remove further resistant importation" = "Remove further\nresistant importation",
  "Equalise treatment benefit" = "Equalise treatment\nbenefit",
  "Equalise resistant-strain PEP effectiveness" = "Equalise resistant-strain\nPEP effectiveness",
  "Equalise treatment and PEP effects" = "Equalise treatment\nand PEP effects",
  "Add resistant-strain fitness cost" = "Add resistant-strain\nfitness cost"
)

mechanism_country <- read_summary("resistance_mechanism_decomposition_summary.csv") %>%
  mutate(
    country = stringr::str_replace_all(country, " ", "_"),
    scenario = mechanism_scenario,
    resistant_infections_per_100k = resistant_infections / total_population / analysis_years * 1e5,
    child_adolescent_cases_per_100k = annualized_child_adolescent_cases_per_100k
  )

mechanism_data <- mechanism_country %>%
  group_by(mechanism_scenario) %>%
  summarise(
    median_resistant_infections_per_100k = median(resistant_infections_per_100k, na.rm = TRUE),
    q25_resistant_infections_per_100k = interval_quantile(resistant_infections_per_100k, 0.25),
    q75_resistant_infections_per_100k = interval_quantile(resistant_infections_per_100k, 0.75),
    median_child_adolescent_cases_per_100k = median(child_adolescent_cases_per_100k, na.rm = TRUE),
    q25_child_adolescent_cases_per_100k = interval_quantile(child_adolescent_cases_per_100k, 0.25),
    q75_child_adolescent_cases_per_100k = interval_quantile(child_adolescent_cases_per_100k, 0.75),
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
      levels = c("Baseline", mechanism_labels[mechanism_plot_order])
    ),
    end_fraction_label = lancet_percent(median_end_resistant_fraction, accuracy = 1)
  ) %>%
  filter(!is.na(scenario), is.finite(median_resistant_infections_per_100k)) %>%
  arrange(scenario)

mechanism_baseline <- mechanism_country %>%
  filter(scenario == "baseline_full_mechanism") %>%
  transmute(
    country,
    baseline_resistant_infections_per_100k = resistant_infections_per_100k,
    baseline_child_adolescent_cases_per_100k = child_adolescent_cases_per_100k
  )

mechanism_delta <- mechanism_country %>%
  left_join(mechanism_baseline, by = "country") %>%
  filter(scenario != "baseline_full_mechanism") %>%
  mutate(
    mechanism_label = factor(
      mechanism_labels[scenario],
      levels = mechanism_labels[mechanism_plot_order]
    ),
    resistant_infection_reduction_vs_baseline = if_else(
      baseline_resistant_infections_per_100k > 0,
      1 - resistant_infections_per_100k / baseline_resistant_infections_per_100k,
      NA_real_
    ),
    child_adolescent_case_reduction_vs_baseline =
      1 - child_adolescent_cases_per_100k / pmax(baseline_child_adolescent_cases_per_100k, 1e-9)
  ) %>%
  pivot_longer(
    c(resistant_infection_reduction_vs_baseline, child_adolescent_case_reduction_vs_baseline),
    names_to = "metric",
    values_to = "reduction_vs_baseline"
  ) %>%
  mutate(
    metric = recode(
      metric,
      resistant_infection_reduction_vs_baseline = "Resistant infections",
      child_adolescent_case_reduction_vs_baseline = "Cases aged <18 years"
    ),
    metric = factor(metric, levels = c("Resistant infections", "Cases aged <18 years"))
  )

mechanism_y_lookup <- setNames(
  seq_along(levels(mechanism_delta$mechanism_label)),
  levels(mechanism_delta$mechanism_label)
)

mechanism_metric_offsets <- c(
  "Cases aged <18 years" = -0.18,
  "Resistant infections" = 0.18
)

mechanism_delta <- mechanism_delta %>%
  mutate(
    mechanism_y = unname(mechanism_y_lookup[as.character(mechanism_label)]),
    mechanism_y_metric = mechanism_y + unname(mechanism_metric_offsets[as.character(metric)])
  )

mechanism_delta_summary <- mechanism_delta %>%
  group_by(scenario, mechanism_label, metric) %>%
  summarise(
    mechanism_y = first(mechanism_y),
    mechanism_y_metric = first(mechanism_y_metric),
    median_reduction_vs_baseline = median(reduction_vs_baseline, na.rm = TRUE),
    q25_reduction_vs_baseline = interval_quantile(reduction_vs_baseline, 0.25),
    q75_reduction_vs_baseline = interval_quantile(reduction_vs_baseline, 0.75),
    .groups = "drop"
  )

readr::write_csv(
  mechanism_delta %>%
    left_join(
      mechanism_delta_summary %>%
        select(
          scenario,
          mechanism_label,
          metric,
          median_reduction_vs_baseline,
          q25_reduction_vs_baseline,
          q75_reduction_vs_baseline
        ),
      by = c("scenario", "mechanism_label", "metric")
    ) %>%
    transmute(
      country,
      scenario,
      mechanism_label = as.character(mechanism_label),
      mechanism_axis_label = unname(mechanism_axis_labels[as.character(mechanism_label)]),
      metric = as.character(metric),
      mechanism_y,
      mechanism_y_metric,
      reduction_vs_baseline,
      median_reduction_vs_baseline,
      q25_reduction_vs_baseline,
      q75_reduction_vs_baseline,
      resistant_infections_per_100k,
      baseline_resistant_infections_per_100k,
      child_adolescent_cases_per_100k,
      baseline_child_adolescent_cases_per_100k,
      end_resistant_fraction = resistant_fraction_end,
      resistant_importation,
      treatment_differential,
      pep_differential,
      fitness_R,
      interpretation = mechanism_interpretation
    ),
  model_path("outputs", "tables", "figure4a_resistance_mechanism_contrasts.csv")
)

mechanism_y_breaks <- tibble::tibble(
  mechanism_label = names(mechanism_y_lookup),
  mechanism_y = unname(mechanism_y_lookup),
  mechanism_axis_label = unname(mechanism_axis_labels[mechanism_label])
)

mechanism_outcome_colours <- c(
  "Cases aged <18 years" = case_benefit_colour,
  "Resistant infections" = resistance_colour
)

mechanism_outcome_breaks <- levels(mechanism_delta$metric)
mechanism_outcome_labels <- c(
  "Resistant infections" = "Resistant infections",
  "Cases aged <18 years" = "All <18 cases"
)

mechanism_point_legend <- tibble::tibble(
  reduction_vs_baseline = 0,
  mechanism_y_metric = mechanism_y_breaks$mechanism_y[[1]],
  point_summary = factor(
    c("Profiles", "Cross-profile median"),
    levels = c("Profiles", "Cross-profile median")
  )
)

mechanism_profile_points <- mechanism_delta %>%
  filter(is.finite(reduction_vs_baseline))

p4a <- ggplot() +
  geom_vline(xintercept = 0, linewidth = 0.24, colour = manuscript_colour("pale_grey")) +
  geom_point(
    data = mechanism_profile_points,
    aes(reduction_vs_baseline, mechanism_y_metric, colour = metric),
    shape = 16,
    size = 1.18,
    alpha = 0.52,
    position = position_jitter(width = 0, height = 0.075, seed = 4)
  ) +
  geom_point(
    data = mechanism_delta_summary,
    aes(median_reduction_vs_baseline, mechanism_y_metric, colour = metric, fill = metric),
    shape = 23,
    size = 2.35,
    stroke = 0.38
  ) +
  geom_point(
    data = mechanism_point_legend,
    aes(reduction_vs_baseline, mechanism_y_metric, shape = point_summary),
    colour = manuscript_colour("grey"),
    fill = "white",
    alpha = 0,
    size = 2.0,
    show.legend = TRUE
  ) +
  scale_x_continuous(
    breaks = seq(0, 1, by = 0.25),
    labels = label_lancet_percent(accuracy = 1),
    expand = expansion(mult = c(0.02, 0.03))
  ) +
  scale_y_continuous(
    breaks = mechanism_y_breaks$mechanism_y,
    labels = mechanism_y_breaks$mechanism_axis_label,
    expand = expansion(add = c(0.28, 0.28))
  ) +
  scale_colour_manual(
    values = mechanism_outcome_colours[mechanism_outcome_breaks],
    breaks = mechanism_outcome_breaks,
    labels = unname(mechanism_outcome_labels[mechanism_outcome_breaks]),
    name = "Outcome",
    guide = guide_legend(
      nrow = 1,
      byrow = TRUE,
      title.position = "top",
      title.hjust = 0,
      keywidth = unit(0.34, "cm"),
      keyheight = unit(0.42, "cm"),
      order = 1,
      override.aes = list(
        shape = 16,
        colour = unname(mechanism_outcome_colours[mechanism_outcome_breaks]),
        fill = NA,
        size = 2.1,
        alpha = 1
      )
    )
  ) +
  scale_fill_manual(
    values = mechanism_outcome_colours,
    guide = "none"
  ) +
  scale_shape_manual(
    values = c("Profiles" = 16, "Cross-profile median" = 23),
    name = "Point",
    guide = guide_legend(
      nrow = 1,
      byrow = TRUE,
      title.position = "top",
      title.hjust = 0,
      keywidth = unit(0.34, "cm"),
      keyheight = unit(0.42, "cm"),
      order = 2,
      override.aes = list(
        fill = c(NA, manuscript_colour("grey")),
        colour = manuscript_colour("grey"),
        alpha = c(0.72, 1),
        size = c(1.7, 2.25)
      )
    )
  ) +
  coord_cartesian(xlim = c(-0.08, 1.03), clip = "off") +
  labs(
    x = "Reduction versus full-resistance\nbaseline (%)",
    y = NULL,
    tag = "a"
  ) +
  theme_lancet_panel(base_size = journal_compact_text_size, plot_margin = margin(4, 5, 4, 4), show_x_grid = TRUE, show_y_grid = TRUE) +
  theme(
    axis.ticks.y = element_blank(),
    axis.text.y = element_text(lineheight = 0.94),
    legend.position = "inside",
    legend.position.inside = c(1, 0.01),
    legend.justification = c(1, 0),
    legend.direction = "vertical",
    legend.box = "vertical",
    legend.box.just = "left",
    legend.title = element_text(face = "bold", lineheight = 0.90, hjust = 0),
    legend.text = element_text(lineheight = 1.12, hjust = 0, margin = margin(t = 2, b = 2)),
    legend.key.height = unit(0.42, "cm"),
    legend.spacing.x = unit(4, "pt"),
    legend.spacing.y = unit(1, "pt"),
    legend.background = element_rect(fill = "#FFFFFFE8", colour = NA),
    legend.box.margin = margin(0, 0, 0, 0),
    legend.margin = margin(2, 3, 2, 3),
    panel.grid.major.x = element_line(linewidth = 0.16, colour = lancet_grid_light_colour)
  )

## Panel B: resistance-management pathway --------------------------------------

resistance_management <- decision_frontier %>%
  filter(optimization_constraint == "program_plus_resistance", strategy == "resistance_guided_treatment") %>%
  select(
    country,
    country_label_text,
    country_code,
    primary_case_reduction,
    primary_cases_per_100k,
    relative_reduction_resistant_infections,
    remaining_resistant_infections_per_100k = annualized_resistant_infections_per_100k
  ) %>%
  left_join(current_resistance, by = "country") %>%
  mutate(
    resistance_reduction_estimable = is.finite(relative_reduction_resistant_infections),
    resistant_reduction_plot = if_else(resistance_reduction_estimable, relative_reduction_resistant_infections, 0),
    current_resistant_infections_plot = pmax(current_resistant_infections_per_100k, 1)
  )

p4b_label_data <- resistance_management

readr::write_csv(
  resistance_management %>%
    transmute(
      country,
      country_code,
      current_resistant_infections_per_100k,
      remaining_resistant_infections_per_100k,
      remaining_child_adolescent_cases_per_100k = primary_cases_per_100k,
      primary_case_reduction,
      relative_reduction_resistant_infections,
      resistance_reduction_estimable
    ),
  model_path("outputs", "tables", "figure4b_resistance_management_tradeoff.csv")
)

pal_breaks <- c(1, 100, 1000)
pal_limits <- c(1, 3500)

p4b <- ggplot(resistance_management, aes(primary_case_reduction, resistant_reduction_plot)) +
  geom_hline(yintercept = 0, linewidth = 0.22, colour = "grey72") +
  geom_vline(xintercept = 0, linewidth = 0.22, colour = "grey72") +
  geom_abline(slope = 1, intercept = 0, linewidth = 0.22, linetype = "22", colour = manuscript_colour("mid_grey")) +
  annotate(
    "text",
    x = 0.23,
    y = 0.32,
    label = "equal reduction",
    angle = 38,
    size = lancet_pt_to_geom_size(6.2),
    colour = manuscript_colour("mid_grey"),
    hjust = 0
  ) +
  geom_point(
    data = filter(resistance_management, resistance_reduction_estimable),
    aes(colour = current_resistant_infections_plot),
    shape = 16,
    size = 2.45,
    alpha = 0.78
  ) +
  geom_point(
    data = filter(resistance_management, !resistance_reduction_estimable),
    aes(shape = "No baseline resistant infections"),
    size = 2.35,
    colour = manuscript_colour("mid_grey"),
    fill = "white",
    stroke = 0.35,
    alpha = 0.82
  ) +
  ggrepel::geom_text_repel(
    data = p4b_label_data,
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
  scale_colour_gradientn(
    colours = c(manuscript_colour("light_grey"), manuscript_colour("pale_orange"), resistance_colour),
    trans = "log10",
    breaks = pal_breaks,
    limits = pal_limits,
    labels = label_lancet_comma(accuracy = 1),
    name = "Current resistant infections\nper 100 000/year (log scale)",
    guide = guide_colourbar(
      title.position = "top",
      barwidth = unit(2.6, "cm"),
      barheight = unit(0.18, "cm"),
      ticks.linewidth = 0.20,
      order = 1
    )
  ) +
  scale_shape_manual(
    values = c("No baseline resistant infections" = 21),
    guide = "none"
  ) +
  coord_cartesian(xlim = c(-0.03, 0.85), ylim = c(-0.06, 0.86), clip = "off") +
  labs(
    x = "Reduction in symptomatic cases\namong people aged <18 years",
    y = "Reduction in resistant infections",
    tag = "b"
  ) +
  theme_lancet_panel(base_size = journal_base_text_size, plot_margin = margin(4, 5, 4, 4), show_y_grid = TRUE) +
  theme(
    panel.grid.major.x = element_line(linewidth = 0.16, colour = lancet_grid_light_colour),
    legend.position = "inside",
    legend.position.inside = c(1, 0.01),
    legend.justification = c(1, 0),
    legend.direction = "horizontal",
    legend.title = element_text(size = journal_compact_text_size - 0.2, face = "bold", lineheight = 0.9),
    legend.text = element_text(size = journal_compact_text_size - 0.5),
    legend.key.width = unit(0.34, "cm"),
    legend.background = element_rect(fill = "#FFFFFFE8", colour = NA),
    legend.margin = margin(2, 3, 2, 3)
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
      pep_assumption == "Treatment + restored PEP effect" ~ "Treatment + restored PEP",
      TRUE ~ "Guided treatment only"
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
      metric = "Cases aged <18 years",
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
    metric = factor(metric, levels = c("Cases aged <18 years", "Resistant infections")),
    pep_assumption = factor(pep_assumption, levels = c("Guided treatment only", "Treatment + restored PEP effect")),
    reach_status = factor(reach_status, levels = c("Baseline PEP reach", "Lower PEP reach")),
    implementation_group = factor(
      implementation_group,
      levels = c("Guided treatment only", "Treatment + restored PEP", "Lower PEP reach")
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
        "Guided treatment only" = management_colour,
        "Treatment + restored PEP" = pep_restored_colour,
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

implementation_cases_panel <- make_implementation_panel("Cases aged <18 years", "c", "Bounded response")
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
    country_code = unname(figure4_country_codes[country]),
    scenario_label = factor(unname(vaccine_profile_labels[scenario]), levels = vaccine_display_labels),
    primary_cases_per_100k = as.numeric(annualized_child_adolescent_cases_per_100k),
    primary_case_reduction = 1 - primary_cases_per_100k / pmax(current_primary_cases_per_100k, 1e-9),
    relative_reduction_total_infections = 1 - as.numeric(annualized_infections_per_100k) / pmax(current_infections_per_100k, 1e-9)
  ) %>%
  filter(!is.na(primary_case_reduction), !is.na(relative_reduction_total_infections), !is.na(primary_cases_per_100k))

combined_stress_outcomes <- read_table("lancet_child_adolescent_strategy_burden.csv") %>%
  mutate(
    country = stringr::str_replace_all(country, " ", "_"),
    country_code = unname(figure4_country_codes[country]),
    strategy = as.character(strategy),
    primary_cases_per_100k = as.numeric(primary_cases_per_100k)
  ) %>%
  filter(strategy == "combined_strategy") %>%
  transmute(
    country,
    country_code,
    scenario = strategy,
    scenario_label = "Combined future upper-bound scenario",
    primary_cases_per_100k,
    outcome_type = "Upper-bound scenario"
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
    country_label_text = format_country(country),
    outcome_type = factor(outcome_type, levels = c("Vaccine mechanism", "Upper-bound scenario"))
  )

vaccine_burden_summary <- vaccine_residual %>%
  group_by(scenario_label) %>%
  summarise(
    median = median(primary_cases_per_100k, na.rm = TRUE),
    q25 = interval_quantile(primary_cases_per_100k, 0.25),
    q75 = interval_quantile(primary_cases_per_100k, 0.75),
    q025 = interval_quantile(primary_cases_per_100k, 0.025),
    q975 = interval_quantile(primary_cases_per_100k, 0.975),
    min_primary_cases_per_100k = min(primary_cases_per_100k, na.rm = TRUE),
    max_primary_cases_per_100k = max(primary_cases_per_100k, na.rm = TRUE),
    .groups = "drop"
  )

readr::write_csv(
  vaccine_residual %>%
    mutate(scenario_label = as.character(scenario_label)) %>%
    left_join(vaccine_burden_summary, by = "scenario_label") %>%
    transmute(
      country,
      country_code,
      country_label_text,
      scenario,
      scenario_label,
      outcome_type = as.character(outcome_type),
      primary_cases_per_100k,
      min_primary_cases_per_100k,
      max_primary_cases_per_100k,
      median,
      q25,
      q75,
      q025,
      q975
    ),
  model_path("outputs", "tables", "figure4c_future_vaccine_residual_burden.csv")
)

p4c <- ggplot() +
  geom_point(
    data = vaccine_residual,
    aes(primary_cases_per_100k, scenario_label, colour = scenario_label, shape = "Profiles"),
    size = 1.18,
    alpha = 0.52,
    position = position_jitter(width = 0, height = 0.075, seed = 4)
  ) +
  geom_point(
    data = vaccine_burden_summary,
    aes(median, scenario_label, shape = "Median"),
    size = 2.45,
    fill = lancet_text_colour,
    colour = "white",
    stroke = 0.18
  ) +
  scale_x_log10(
    breaks = c(0.1, 1, 10, 100, 1000),
    labels = lancet_midline_decimal(c("0.1", "1", "10", "100", "1,000"))
  ) +
  scale_y_discrete(labels = vaccine_profile_axis_labels) +
  scale_colour_vaccine_profiles() +
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
  coord_cartesian(xlim = c(0.08, 1400), clip = "off") +
  labs(
    x = "Remaining cases per 100 000/year\namong people aged <18 years (log scale)",
    y = NULL,
    tag = "c"
  ) +
  theme_lancet_panel(base_size = journal_dense_text_size, plot_margin = margin(4, 5, 4, 5), show_y_grid = TRUE) +
  theme(
    axis.ticks.y = element_blank(),
    axis.text.y = element_text(lineheight = 0.95),
    legend.position = "inside",
    legend.position.inside = c(0.01, 1),
    legend.justification = c(0, 1),
    legend.direction = "vertical",
    legend.title = element_text(face = "bold", lineheight = 0.90),
    legend.text = element_text(lineheight = 1.16, margin = margin(t = 3, b = 3)),
    legend.key.height = unit(0.46, "cm"),
    legend.spacing.y = unit(8, "pt"),
    legend.background = element_rect(fill = "#FFFFFFE6", colour = NA),
    legend.box.margin = margin(1, 0, 0, 0),
    legend.margin = margin(3, 0, 3, 0),
    panel.grid.major.x = element_line(linewidth = 0.16, colour = lancet_grid_light_colour)
  )

veinf_summary_names <- c(
  "veinf_resistance_grid_summary.csv",
  "veinf_resistance_grid_low_seed_summary.csv"
)

veinf_grid_raw <- bind_rows(
  lapply(
    veinf_summary_names[file.exists(model_path("outputs", "summaries", veinf_summary_names))],
    read_summary
  )
) %>%
  mutate(
    country = stringr::str_replace_all(country, " ", "_"),
    starting_resistant_fraction = round(as.numeric(grid_resistance_prevalence), 2),
    infectiousness_reduction = round(as.numeric(grid_VE_inf), 2),
    infant_cases_per_100k = as.numeric(annualized_infant_cases_per_100k)
  )

veinf_plot_resistance_levels <- c(0, 0.01, 0.03, 0.05)
veinf_resistance_levels <- veinf_plot_resistance_levels[
  veinf_plot_resistance_levels %in% sort(unique(veinf_grid_raw$starting_resistant_fraction))
]

threshold_comparator_labels <- c(
  "Infant-exposure composite",
  "Resistance-guided management"
)

veinf_threshold_grid <- veinf_grid_raw %>%
  filter(
    starting_resistant_fraction %in% veinf_resistance_levels,
    infectiousness_reduction >= 0,
    infectiousness_reduction <= 0.6
  )

veinf_threshold_comparators <- read_summary("intervention_scenarios_summary.csv") %>%
  filter(as.character(scenario) %in% c("maternal_immunization", "resistance_guided_treatment")) %>%
  transmute(
    country = stringr::str_replace_all(country, " ", "_"),
    comparator = recode(
      as.character(scenario),
      maternal_immunization = "infant_exposure_reduction_strategy",
      resistance_guided_treatment = "resistance_guided_treatment"
    ),
    comparator_label = recode(
      comparator,
      infant_exposure_reduction_strategy = "Infant-exposure composite",
      resistance_guided_treatment = "Resistance-guided management"
    ),
    comparator_infant_cases_per_100k = as.numeric(annualized_infant_cases_per_100k)
  )

veinf_country_thresholds <- veinf_threshold_grid %>%
  inner_join(veinf_threshold_comparators, by = "country", relationship = "many-to-many") %>%
  group_by(
    country,
    starting_resistant_fraction,
    comparator,
    comparator_label,
    comparator_infant_cases_per_100k
  ) %>%
  summarise(
    minimum_infectiousness_reduction = {
      eligible <- infectiousness_reduction[infant_cases_per_100k <= comparator_infant_cases_per_100k]
      if (length(eligible) == 0) NA_real_ else min(eligible, na.rm = TRUE)
    },
    .groups = "drop"
  ) %>%
  mutate(
    resistance_label = factor(
      paste0(lancet_percent(starting_resistant_fraction, accuracy = 1), " resistant-strain seeding"),
      levels = paste0(lancet_percent(veinf_resistance_levels, accuracy = 1), " resistant-strain seeding")
    ),
    comparator_label = factor(comparator_label, levels = threshold_comparator_labels),
    reached_comparator = is.finite(minimum_infectiousness_reduction)
  )

veinf_threshold_summary <- veinf_country_thresholds %>%
  group_by(starting_resistant_fraction, resistance_label, comparator, comparator_label) %>%
  summarise(
    median_minimum_infectiousness_reduction = median(minimum_infectiousness_reduction, na.rm = TRUE),
    q25_minimum_infectiousness_reduction = interval_quantile(minimum_infectiousness_reduction, 0.25),
    q75_minimum_infectiousness_reduction = interval_quantile(minimum_infectiousness_reduction, 0.75),
    min_minimum_infectiousness_reduction = suppressWarnings(min(minimum_infectiousness_reduction, na.rm = TRUE)),
    max_minimum_infectiousness_reduction = suppressWarnings(max(minimum_infectiousness_reduction, na.rm = TRUE)),
    countries_reaching_comparator = sum(reached_comparator, na.rm = TRUE),
    countries_evaluated = n(),
    .groups = "drop"
  ) %>%
  mutate(
    min_minimum_infectiousness_reduction = if_else(
      is.infinite(min_minimum_infectiousness_reduction),
      NA_real_,
      min_minimum_infectiousness_reduction
    ),
    max_minimum_infectiousness_reduction = if_else(
      is.infinite(max_minimum_infectiousness_reduction),
      NA_real_,
      max_minimum_infectiousness_reduction
    ),
    countries_label = paste0(countries_reaching_comparator, "/", countries_evaluated, " profiles"),
    threshold_basis = "VE_inf-only grid; VE_sus and VE_dur held at the grid baseline.",
    label_vjust = case_when(
      comparator == "infant_exposure_reduction_strategy" & starting_resistant_fraction > 0 ~ 1.55,
      TRUE ~ -0.62
    )
  )

veinf_threshold_source <- veinf_country_thresholds %>%
  left_join(
    veinf_threshold_summary %>%
      select(
        starting_resistant_fraction,
        comparator,
        median_minimum_infectiousness_reduction,
        q25_minimum_infectiousness_reduction,
        q75_minimum_infectiousness_reduction,
        min_minimum_infectiousness_reduction,
        max_minimum_infectiousness_reduction,
        countries_reaching_comparator,
        countries_evaluated,
        threshold_basis
      ),
    by = c("starting_resistant_fraction", "comparator")
  ) %>%
  arrange(starting_resistant_fraction, comparator, country) %>%
  transmute(
    country,
    starting_resistant_fraction,
    resistance_label = as.character(resistance_label),
    comparator,
    comparator_label = as.character(comparator_label),
    comparator_infant_cases_per_100k,
    minimum_infectiousness_reduction,
    reached_comparator,
    median_minimum_infectiousness_reduction,
    q25_minimum_infectiousness_reduction,
    q75_minimum_infectiousness_reduction,
    min_minimum_infectiousness_reduction,
    max_minimum_infectiousness_reduction,
    countries_reaching_comparator,
    countries_evaluated,
    threshold_basis
  )

readr::write_csv(
  veinf_threshold_source,
  model_path("outputs", "tables", "figure4d_future_veinf_thresholds.csv")
)

veinf_attainment_country <- veinf_country_thresholds %>%
  select(
    country,
    starting_resistant_fraction,
    resistance_label,
    comparator,
    comparator_label,
    comparator_infant_cases_per_100k,
    minimum_infectiousness_reduction,
    reached_comparator
  ) %>%
  mutate(join_key = 1L) %>%
  inner_join(
    tibble(
      join_key = 1L,
      infectiousness_reduction = sort(unique(veinf_threshold_grid$infectiousness_reduction))
    ),
    by = "join_key",
    relationship = "many-to-many"
  ) %>%
  select(-join_key) %>%
  mutate(
    matched_comparator = is.finite(minimum_infectiousness_reduction) &
      minimum_infectiousness_reduction <= infectiousness_reduction
  )

veinf_attainment_summary <- veinf_attainment_country %>%
  group_by(
    comparator,
    comparator_label,
    starting_resistant_fraction,
    resistance_label,
    infectiousness_reduction
  ) %>%
  summarise(
    profiles_matching_comparator = sum(matched_comparator, na.rm = TRUE),
    countries_evaluated = n(),
    .groups = "drop"
  ) %>%
  mutate(
    profiles_label = paste0(profiles_matching_comparator, "/", countries_evaluated, " reached")
  )

veinf_attainment_plot_summary <- veinf_attainment_summary %>%
  filter(starting_resistant_fraction %in% veinf_resistance_levels) %>%
  mutate(
    seed_label = factor(
      lancet_percent(starting_resistant_fraction, accuracy = 1),
      levels = lancet_percent(veinf_resistance_levels, accuracy = 1)
    )
  )

veinf_attainment_source <- veinf_attainment_country %>%
  left_join(
    veinf_attainment_summary %>%
      select(
        comparator,
        starting_resistant_fraction,
        infectiousness_reduction,
        profiles_matching_comparator,
        countries_evaluated
      ),
    by = c("comparator", "starting_resistant_fraction", "infectiousness_reduction")
  ) %>%
  arrange(comparator, starting_resistant_fraction, infectiousness_reduction, country) %>%
  transmute(
    country,
    starting_resistant_fraction,
    resistance_label = as.character(resistance_label),
    comparator,
    comparator_label = as.character(comparator_label),
    infectiousness_reduction,
    comparator_infant_cases_per_100k,
    minimum_infectiousness_reduction,
    reached_comparator,
    matched_comparator,
    profiles_matching_comparator,
    countries_evaluated
  )

readr::write_csv(
  veinf_attainment_source,
  model_path("outputs", "tables", "figure4d_future_veinf_attainment.csv")
)

readr::write_csv(
  veinf_attainment_plot_summary %>%
    transmute(
      comparator,
      comparator_label = as.character(comparator_label),
      starting_resistant_fraction,
      seed_label = as.character(seed_label),
      infectiousness_reduction,
      profiles_matching_comparator,
      countries_evaluated
    ),
  model_path("outputs", "tables", "figure4d_future_veinf_seeded_attainment_summary.csv")
)

veinf_seed_colours <- c(
  "0%" = manuscript_colour("black"),
  "1%" = okabe_ito[[6]],
  "3%" = okabe_ito[[2]],
  "5%" = okabe_ito[[8]]
)

veinf_seed_linetypes <- c(
  "0%" = "solid",
  "1%" = "longdash",
  "3%" = "dotdash",
  "5%" = "dashed"
)

veinf_seed_shapes <- c(
  "0%" = 16,
  "1%" = 17,
  "3%" = 15,
  "5%" = 18
)

p4d <- ggplot(
  veinf_attainment_plot_summary,
  aes(
    infectiousness_reduction,
    profiles_matching_comparator,
    colour = seed_label,
    linetype = seed_label,
    shape = seed_label,
    group = seed_label
  )
) +
  geom_step(direction = "hv", linewidth = 0.78, alpha = 0.88, lineend = "round") +
  geom_point(size = 1.25, stroke = 0.20, alpha = 0.88) +
  geom_hline(
    yintercept = c(2, 6, 10),
    linewidth = 0.18,
    colour = manuscript_colour("pale_grey")
  ) +
  facet_wrap(
    vars(comparator_label),
    nrow = 1,
    labeller = as_labeller(c(
      "Infant-exposure composite" = "Match infant-exposure\ncomposite",
      "Resistance-guided management" = "Match resistance-guided\nmanagement"
    ))
  ) +
  scale_x_continuous(
    breaks = seq(0, 0.6, by = 0.1),
    labels = label_lancet_percent(accuracy = 1),
    expand = expansion(mult = c(0.01, 0.02))
  ) +
  scale_y_continuous(
    breaks = seq(0, 10, by = 2),
    limits = c(0, 10),
    expand = expansion(mult = c(0.02, 0.06))
  ) +
  scale_colour_manual(
    values = veinf_seed_colours,
    breaks = names(veinf_seed_colours),
    labels = names(veinf_seed_colours),
    name = "Resistant seeding level",
    guide = guide_legend(nrow = 1, byrow = TRUE, override.aes = list(linewidth = 0.8, size = 2.0))
  ) +
  scale_linetype_manual(
    values = veinf_seed_linetypes,
    breaks = names(veinf_seed_linetypes),
    labels = names(veinf_seed_linetypes),
    name = "Resistant seeding level"
  ) +
  scale_shape_manual(
    values = veinf_seed_shapes,
    breaks = names(veinf_seed_shapes),
    labels = names(veinf_seed_shapes),
    name = "Resistant seeding level"
  ) +
  coord_cartesian(xlim = c(0, 0.6), ylim = c(0, 10.2), clip = "off") +
  labs(
    x = "Vaccine reduction in onward infectiousness (%)",
    y = "Profiles reaching comparator infant-case burden\n(out of 10)",
    tag = "d"
  ) +
  theme_lancet_panel(base_size = journal_compact_text_size, plot_margin = margin(2, 6, 4, 4), show_x_grid = TRUE, show_y_grid = TRUE) +
  theme(
    axis.text.x = element_text(size = journal_compact_text_size - 1),
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.title = element_text(face = "bold", size = journal_compact_text_size, colour = lancet_text_colour),
    legend.text = element_text(size = journal_compact_text_size, colour = lancet_text_colour),
    legend.key.size = unit(0.26, "cm"),
    legend.spacing.x = unit(3, "pt"),
    strip.background = element_blank(),
    strip.text = element_text(face = "bold", size = journal_compact_text_size, lineheight = 0.92, colour = lancet_text_colour),
    panel.spacing.x = unit(0.35, "cm")
  )

## Compose Figure 4 ------------------------------------------------------------

module_title_plot <- function(label) {
  wrap_elements(
    full = grid::textGrob(
      label,
      x = unit(0, "npc"),
      y = unit(0.55, "npc"),
      hjust = 0,
      vjust = 0.5,
      gp = grid::gpar(
        fontfamily = lancet_font_family,
        fontface = "bold",
        fontsize = journal_base_text_size + 0.5,
        col = lancet_text_colour
      )
    )
  )
}

separator_line_plot <- function() {
  wrap_elements(
    full = grid::segmentsGrob(
      x0 = unit(0, "npc"),
      x1 = unit(1, "npc"),
      y0 = unit(0.5, "npc"),
      y1 = unit(0.5, "npc"),
      gp = grid::gpar(col = manuscript_colour("pale_grey"), lwd = 0.7)
    )
  )
}

resistance_row <- p4a + p4b +
  plot_layout(widths = c(1.16, 0.94))

future_vaccine_row <- free(p4c) + p4d +
  plot_layout(widths = c(0.82, 1.22))

figure4 <- module_title_plot("Resistance-management pathway") /
  resistance_row /
  separator_line_plot() /
  module_title_plot("Future vaccine-mechanism threshold analyses") /
  future_vaccine_row +
  plot_layout(heights = c(0.055, 1.0, 0.025, 0.055, 1.05)) &
  theme_lancet_tags()

save_main_figure(figure4, "figure_4_resistance_conditioned_control", height = 7.45)
cat("Figure 4 saved.\n")
