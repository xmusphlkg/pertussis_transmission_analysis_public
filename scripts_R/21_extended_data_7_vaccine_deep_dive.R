args <- commandArgs(FALSE)
file_arg <- sub("^--file=", "", args[grepl("^--file=", args)])
script_dir <- if (length(file_arg) > 0) dirname(normalizePath(file_arg[[1]])) else file.path(getwd(), "scripts_R")
source(file.path(script_dir, "10_shared.R"))

## Extended Data Figure 5: vaccine mechanism deep dive. ----------------------
## Section: extended-data figure setup.
## Purpose: vaccine mechanism deep dive.

vaccine_short_labels <- c(
  no_vaccine = "No vaccine",
  symptom_protective = "Current aP",
  infection_blocking = "Inf.-blocking",
  transmission_blocking = "Trans.-blocking",
  next_generation = "Upper-bound"
)

metric_short_labels <- c(
  relative_reduction_infant_cases = "Infant",
  relative_reduction_reported_cases = "Reported",
  relative_reduction_total_infections = "All",
  relative_reduction_resistant_infections = "Resistant"
)

scenario_table <- readr::read_csv(model_path("publication_inputs", "scenario_table.csv"), show_col_types = FALSE) %>%
  mutate(
    scenario = factor(scenario, levels = vaccine_levels),
    scenario_label = factor(vaccine_labels[as.character(scenario)], levels = vaccine_labels[vaccine_levels])
  )

vaccine_parameters <- scenario_table %>%
  select(scenario_label, VE_sus, VE_sym, VE_inf, VE_dur) %>%
  pivot_longer(-scenario_label, names_to = "effect", values_to = "value") %>%
  mutate(effect = factor(effect, levels = c("VE_sus", "VE_sym", "VE_inf", "VE_dur")))

p_ed7a <- vaccine_parameters %>%
  ggplot(aes(effect, scenario_label, fill = value)) +
  geom_tile(colour = "white", linewidth = lancet_heatmap_tile_linewidth) +
  geom_text(aes(label = lancet_percent(value, accuracy = 1)), size = 1.9) +
  scale_fill_fraction(labels = label_lancet_percent(accuracy = 1)) +
  labs(x = "Vaccine effect parameter", y = NULL, fill = "Value") +
  theme_lancet_heatmap()

vaccine_reduction_data <- vaccine_summary %>%
  filter(scenario != "no_vaccine") %>%
  mutate(scenario_short = factor(vaccine_short_labels[as.character(scenario)], levels = vaccine_short_labels[vaccine_levels[-1]])) %>%
  select(country_label, scenario_short, relative_reduction_infant_cases, relative_reduction_total_infections, relative_reduction_reported_cases, relative_reduction_resistant_infections) %>%
  pivot_longer(-c(country_label, scenario_short), names_to = "metric", values_to = "value") %>%
  mutate(metric = factor(metric_short_labels[metric], levels = c("Infant", "Reported", "All", "Resistant")))

p_ed7b <- vaccine_reduction_data %>%
  ggplot(aes(scenario_short, country_label, fill = value)) +
  geom_tile(colour = "white", linewidth = lancet_heatmap_tile_linewidth) +
  facet_wrap(~metric, nrow = 1) +
  scale_fill_reduction(
    midpoint = 0,
    limits = c(0, 1),
    breaks = seq(0, 1, by = 0.2),
    labels = label_lancet_percent(accuracy = 1),
    oob = scales::squish,
    na.value = manuscript_colour("light_grey"),
    guide = guide_lancet_colourbar(
      barwidth = grid::unit(5.8, "cm"),
      barheight = grid::unit(0.24, "cm"),
      title.position = "top"
    )
  ) +
  labs(x = NULL, y = NULL, fill = "Relative reduction") +
  theme_lancet_heatmap(x_angle = 35, x_hjust = 1)

origin_columns <- c(
  maternal_origin_infection_share = "Maternal",
  dose1_origin_infection_share = "Dose 1",
  dose2_origin_infection_share = "Dose 2",
  dose3plus_origin_infection_share = "Dose 3+",
  waned_origin_infection_share = "Waned"
)

origin_share_data <- vaccine_summary %>%
  mutate(scenario_short = factor(vaccine_short_labels[as.character(scenario)], levels = vaccine_short_labels[vaccine_levels])) %>%
  select(scenario_short, all_of(names(origin_columns))) %>%
  pivot_longer(-scenario_short, names_to = "origin", values_to = "share") %>%
  mutate(origin = factor(origin_columns[origin], levels = origin_columns)) %>%
  group_by(scenario_short, origin) %>%
  summarise(
    median_share = median(share, na.rm = TRUE),
    q025 = interval_quantile(share, 0.025),
    q975 = interval_quantile(share, 0.975),
    .groups = "drop"
  ) %>%
  mutate(
    interval_text = interval_label(median_share, q025, q975, formatter = label_lancet_percent(accuracy = 1)),
    text_colour = if_else(median_share < 0.08, "white", "black")
  )

p_ed7c <- origin_share_data %>%
  ggplot(aes(origin, scenario_short, fill = median_share)) +
  geom_tile(colour = "white", linewidth = lancet_heatmap_tile_linewidth) +
  geom_text(aes(label = interval_text, colour = text_colour), size = 1.75, lineheight = 0.82) +
  scale_fill_infant_burden(
    labels = label_lancet_percent(accuracy = 1),
    guide = guide_colourbar(
      barwidth = grid::unit(40, "mm"),
      barheight = grid::unit(3, "mm"),
      title.position = "top"
    )
  ) +
  scale_colour_identity(guide = "none") +
  labs(x = "Infection source history", y = NULL, fill = "Median infection share") +
  theme_lancet_heatmap(x_angle = 35, x_hjust = 1)

vaccine_sim <- read_model_table_optional(model_path("outputs", "simulations", "vaccine_scenarios"))
require_efigure_timeseries(vaccine_sim, "eFigure 5 representative vaccine trajectories", "vaccine_scenarios")

if (nrow(vaccine_sim) > 0) {
  vaccine_ts <- vaccine_sim %>%
    filter(country %in% c("Australia", "China")) %>%
    add_country_label() %>%
    left_join(baseline %>% select(country, infant_population), by = "country") %>%
    mutate(
      scenario = factor(scenario, levels = vaccine_levels),
      scenario_short = factor(vaccine_short_labels[as.character(scenario)], levels = vaccine_short_labels[vaccine_levels])
    ) %>%
    group_by(country_label, scenario_short, time) %>%
    summarise(
      simulation_year = first(time) / 365,
      infant_population = max(infant_population, na.rm = TRUE),
      infant_case_rate = sum(infant_case_rate_per_day, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(infant_case_incidence = infant_case_rate / pmax(infant_population, 1e-9) * 365 * 1e5)

  p_ed7d <- vaccine_ts %>%
    ggplot(aes(simulation_year, infant_case_incidence, colour = scenario_short)) +
    geom_line(linewidth = 0.3) +
    facet_wrap(~country_label, scales = "free_y", nrow = 1) +
    scale_x_continuous(breaks = seq(0, 30, by = 10)) +
    scale_y_continuous(labels = label_number(accuracy = 1)) +
    labs(x = "Simulation year", y = "Infant cases per 100,000 infants/year", colour = NULL)
} else {
  vaccine_ts <- vaccine_summary %>%
    filter(country %in% c("Australia", "China")) %>%
    mutate(scenario_short = factor(vaccine_short_labels[as.character(scenario)], levels = vaccine_short_labels[vaccine_levels]))

  p_ed7d <- vaccine_ts %>%
    ggplot(aes(scenario_short, annualized_infant_cases_per_100k, colour = scenario_short, group = 1)) +
    geom_line(linewidth = 0.3, colour = manuscript_colour("grey")) +
    geom_point(size = 1.8) +
    facet_wrap(~country_label, scales = "free_y", nrow = 1) +
    scale_y_log10(labels = label_number(accuracy = 1)) +
    labs(x = NULL, y = "Annualized infant cases per 100,000 (log)", colour = NULL)
}

p_ed7d <- p_ed7d +
  scale_colour_manual(values = manuscript_vaccine_profile_colours, guide = guide_legend(nrow = 2, byrow = TRUE)) +
  theme_lancet()
if (nrow(vaccine_sim) == 0) {
  p_ed7d <- p_ed7d + theme(axis.text.x = element_text(angle = 35, hjust = 1))
}

extended7 <- free(p_ed7b) + free(p_ed7c) + free(p_ed7d) +
  plot_layout(design = "AA\nBC", guides = "keep", heights = c(1.22, 1), widths = c(1.05, 0.95)) +
  plot_annotation(tag_levels = "A") &
  (theme(plot.margin = margin(3, 3, 3, 3)) + theme_lancet_tags())

save_appendix_figure(extended7, "extended_data_figure_5_vaccine_deep_dive", height = 7.4)
