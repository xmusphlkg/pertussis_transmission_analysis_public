args <- commandArgs(FALSE)
file_arg <- sub("^--file=", "", args[grepl("^--file=", args)])
script_dir <- if (length(file_arg) > 0) dirname(normalizePath(file_arg[[1]])) else file.path(getwd(), "scripts_R")
source(file.path(script_dir, "10_shared.R"))

## Extended Data Figure 7: intervention strategy extended outcomes. ----------------------
## Section: extended-data figure setup.
## Purpose: intervention strategy extended outcomes.

efig7_intervention_labels <- c(
  higher_child_coverage = "Nominal\ncoverage",
  adolescent_booster = "Adolescent\nbooster",
  pregnancy_tdap_scaleup = "Pregnancy\nTdap",
  cocooning_adjunct = "Close-contact\nadjunct",
  maternal_immunization = "Infant-exposure\nstrategy",
  targeted_pep_high_risk = "Targeted\nPEP",
  resistance_guided_treatment = "Resistance-guided\nmanagement",
  next_generation_vaccine = "High-blocking\nvaccine target",
  combined_strategy = "Combined\nstrategy"
)

metric_short_labels <- c(
  relative_reduction_infant_cases = "Infant",
  relative_reduction_reported_cases = "Reported",
  relative_reduction_total_infections = "All",
  relative_reduction_resistant_infections = "Resistant"
)

intervention_effects <- intervention_summary %>%
  mutate(scenario_key = as.character(scenario)) %>%
  filter(scenario_key %in% intervention_levels) %>%
  mutate(
    scenario = factor(scenario_key, levels = intervention_levels),
    scenario_label = factor(efig7_intervention_labels[scenario_key], levels = efig7_intervention_labels[intervention_levels]),
    scenario_short = factor(efig7_intervention_labels[scenario_key], levels = efig7_intervention_labels[intervention_levels])
  )

intervention_levers <- tribble(
  ~scenario, ~lever,
  "higher_child_coverage", "Nominal\ncoverage",
  "adolescent_booster", "Adolescent\nbooster",
  "pregnancy_tdap_scaleup", "Pregnancy\nTdap",
  "cocooning_adjunct", "Close-contact\nadjunct",
  "maternal_immunization", "Pregnancy\nTdap",
  "maternal_immunization", "Close-contact\nadjunct",
  "targeted_pep_high_risk", "Targeted\nPEP",
  "resistance_guided_treatment", "Resistance-guided\nmanagement",
  "next_generation_vaccine", "High-blocking\nvaccine target",
  "combined_strategy", "Pregnancy\nTdap",
  "combined_strategy", "Close-contact\nadjunct",
  "combined_strategy", "Adolescent\nbooster",
  "combined_strategy", "Targeted\nPEP",
  "combined_strategy", "Resistance-guided\nmanagement",
  "combined_strategy", "Transmission-blocking\nvaccine"
) %>%
  mutate(active = TRUE)

lever_matrix <- expand_grid(
  scenario = intervention_levels,
  lever = c(
    "Nominal\ncoverage", "Adolescent\nbooster", "Pregnancy\nTdap",
    "Close-contact\nadjunct", "Targeted\nPEP", "Resistance-guided\nmanagement",
    "High-blocking\nvaccine target", "Transmission-blocking\nvaccine"
  )
) %>%
  left_join(intervention_levers, by = c("scenario", "lever")) %>%
  mutate(
    active = replace_na(active, FALSE),
    scenario_label = factor(efig7_intervention_labels[scenario], levels = rev(efig7_intervention_labels[intervention_levels])),
    lever = factor(lever, levels = c(
      "Nominal\ncoverage", "Adolescent\nbooster", "Pregnancy\nTdap",
      "Close-contact\nadjunct", "Targeted\nPEP", "Resistance-guided\nmanagement",
      "High-blocking\nvaccine target", "Transmission-blocking\nvaccine"
    ))
  )

p_ed10a <- lever_matrix %>%
  ggplot(aes(lever, scenario_label, fill = active)) +
  geom_tile(colour = "white", linewidth = lancet_heatmap_tile_linewidth) +
  scale_fill_manual(
    values = c("TRUE" = manuscript_colour("blue"), "FALSE" = manuscript_colour("light_grey")),
    guide = "none"
  ) +
  scale_x_discrete(guide = guide_axis(n.dodge = 2)) +
  labs(x = NULL, y = NULL) +
  theme_lancet_heatmap(x_angle = 0, x_hjust = 0.5, x_size = 4.9)

intervention_outcomes <- intervention_effects %>%
  select(country_label, scenario_short, relative_reduction_infant_cases, relative_reduction_reported_cases, relative_reduction_total_infections, relative_reduction_resistant_infections) %>%
  pivot_longer(-c(country_label, scenario_short), names_to = "metric", values_to = "value") %>%
  mutate(metric = factor(metric_short_labels[metric], levels = c("Infant", "Reported", "All", "Resistant")))

p_ed10b <- intervention_outcomes %>%
  ggplot(aes(scenario_short, country_label, fill = value)) +
  geom_tile(colour = "white", linewidth = lancet_heatmap_tile_linewidth) +
  facet_wrap(~metric, nrow = 2) +
  scale_fill_reduction(
    midpoint = 0,
    limits = c(-0.25, 0.90),
    breaks = seq(-0.2, 0.8, by = 0.2),
    labels = label_lancet_percent(accuracy = 1),
    oob = scales::squish,
    na.value = manuscript_colour("light_grey"),
    guide = guide_lancet_colourbar(
      barwidth = grid::unit(6.2, "cm"),
      barheight = grid::unit(0.24, "cm"),
      title.position = "left"
    )
  ) +
  labs(x = "Intervention\nstrategy", y = NULL, fill = "Relative\nreduction") +
  theme_lancet_heatmap(x_angle = 45, x_hjust = 1, x_size = 4.3) +
  theme(
    legend.key.width = unit(6.2, "cm"),
    legend.key.height = unit(0.24, "cm")
  )

## Panel C: Infant-exposure reduction strategy decomposition ----------------------
## Section: Panel C.
## Purpose: Infant-exposure reduction strategy decomposition.

maternal_decomp_levels <- c(
  "pregnancy_tdap_scaleup", "maternal_adult_boosting_only",
  "maternal_cocooning_only", "maternal_immunization"
)
maternal_decomp_labels <- c(
  pregnancy_tdap_scaleup = "Pregnancy Tdap",
  maternal_adult_boosting_only = "Adult boosting",
  maternal_cocooning_only = "Contact reduction",
  maternal_immunization = "Infant-exposure strategy"
)
maternal_decomp_colours <- c(
  "Pregnancy Tdap" = manuscript_colour("sky"),
  "Adult boosting" = manuscript_colour("orange"),
  "Contact reduction" = manuscript_colour("green"),
  "Infant-exposure strategy" = manuscript_colour("magenta")
)

maternal_decomposition_components <- intervention_summary %>%
  mutate(scenario_key = if ("intervention" %in% names(.)) as.character(intervention) else as.character(scenario)) %>%
  filter(scenario_key %in% maternal_decomp_levels) %>%
  transmute(country_label, scenario = scenario_key, relative_reduction_infant_cases)

maternal_decomposition_summary <- read_model_table_optional(
  model_path("outputs", "summaries", "maternal_decomposition_summary")
)
if (nrow(maternal_decomposition_components) == 0 && nrow(maternal_decomposition_summary) > 0) {
  maternal_current <- intervention_summary %>%
    mutate(scenario_key = if ("intervention" %in% names(.)) as.character(intervention) else as.character(scenario)) %>%
    filter(scenario_key == "current") %>%
    select(country, current_infant = annualized_infant_cases_per_100k)

  maternal_decomposition_components <- maternal_decomposition_summary %>%
    add_country_label() %>%
    left_join(maternal_current, by = "country") %>%
    transmute(
      country_label,
      scenario,
      relative_reduction_infant_cases = (current_infant - annualized_infant_cases_per_100k) /
        pmax(current_infant, 1e-9)
    )
}

# Decomposition components are compared with the full infant-exposure reduction strategy.
maternal_decomp <- maternal_decomposition_components %>%
  filter(scenario %in% maternal_decomp_levels) %>%
  mutate(
    component = factor(
      maternal_decomp_labels[as.character(scenario)],
      levels = maternal_decomp_labels[maternal_decomp_levels]
    )
  )

missing_decomp_components <- setdiff(maternal_decomp_levels, unique(as.character(maternal_decomp$scenario)))
expected_decomp_cells <- expand_grid(
  scenario = maternal_decomp_levels,
  country_label = country_label_levels
)
observed_decomp_cells <- maternal_decomp %>%
  transmute(
    scenario = as.character(scenario),
    country_label = as.character(country_label)
  ) %>%
  distinct()
missing_decomp_cells <- expected_decomp_cells %>%
  anti_join(observed_decomp_cells, by = c("scenario", "country_label"))
if (length(missing_decomp_components) > 0 || nrow(missing_decomp_cells) > 0) {
  stop(
    "eFigure 7 panel C requires complete infant-exposure decomposition source data. Missing components: ",
    paste(missing_decomp_components, collapse = ", "),
    "; missing component-country cells: ",
    paste(
      paste0(missing_decomp_cells$scenario, "/", missing_decomp_cells$country_label),
      collapse = ", "
    ),
    call. = FALSE
  )
}

# Compute median and empirical ranges across country profiles.
maternal_decomp_agg <- maternal_decomp %>%
  group_by(component) %>%
  summarise(
    median_reduction = median(relative_reduction_infant_cases, na.rm = TRUE),
    q025 = interval_quantile(relative_reduction_infant_cases, 0.025),
    q975 = interval_quantile(relative_reduction_infant_cases, 0.975),
    q25 = interval_quantile(relative_reduction_infant_cases, 0.25),
    q75 = interval_quantile(relative_reduction_infant_cases, 0.75),
    .groups = "drop"
  )

p_ed10c <- ggplot(maternal_decomp,
                  aes(relative_reduction_infant_cases, component, colour = component)) +
  geom_vline(xintercept = 0, linewidth = 0.25, colour = manuscript_colour("pale_grey")) +
  geom_point(size = 1.3, alpha = 0.6,
             position = position_jitter(height = 0.1, width = 0)) +
  geom_errorbar(
    data = maternal_decomp_agg,
    aes(xmin = q025, xmax = q975, y = component),
    width = 0.22, linewidth = 0.28, colour = manuscript_colour("mid_grey"), alpha = 0.55,
    inherit.aes = FALSE, orientation = "y"
  ) +
  geom_errorbar(
    data = maternal_decomp_agg,
    aes(xmin = q25, xmax = q75, y = component),
    width = 0, linewidth = 0.7, colour = manuscript_colour("mid_grey"),
    inherit.aes = FALSE, orientation = "y"
  ) +
  geom_point(
    data = maternal_decomp_agg,
    aes(x = median_reduction, y = component),
    shape = 18, size = 3.0, colour = "black",
    inherit.aes = FALSE
  ) +
  scale_x_continuous(labels = percent_format(accuracy = 1)) +
  scale_colour_manual(values = maternal_decomp_colours, guide = "none") +
  labs(x = "Relative reduction in infant cases vs current\n(country-profile ranges)", y = NULL) +
  theme_lancet() +
  theme(plot.margin = margin(3, 10, 3, 3))

## Compose eFigure 7 (submitted panels) ----------------------
## Section: figure assembly.
## Purpose: Compose eFigure 7 (submitted panels).

extended10 <- free(p_ed10a) + free(p_ed10b) + free(p_ed10c) +
  plot_layout(design = "AC\nBB\nBB", guides = "keep", widths = c(0.95, 1.05), heights = c(0.95, 1, 1)) +
  plot_annotation(tag_levels = "A") &
  (theme(plot.margin = margin(4, 5, 4, 4)) + theme_lancet_tags())

save_appendix_figure(extended10, "extended_data_figure_7_intervention_extended", width = lancet_double_width * 1.08, height = 8.6)
