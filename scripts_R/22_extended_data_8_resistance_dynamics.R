args <- commandArgs(FALSE)
file_arg <- sub("^--file=", "", args[grepl("^--file=", args)])
script_dir <- if (length(file_arg) > 0) dirname(normalizePath(file_arg[[1]])) else file.path(getwd(), "scripts_R")
source(file.path(script_dir, "10_shared.R"))

## Extended Data Figure 6: resistance dynamics. ----------------------
## Section: extended-data figure setup.
## Purpose: resistance dynamics.

resistance_targets <- readr::read_csv(model_path("publication_inputs", "resistance_scenario_table.csv"), show_col_types = FALSE) %>%
  transmute(
    scenario = as.character(scenario),
    scenario_target = as.numeric(target_prevalence_at_analysis_start)
  )

p_ed8a <- resistance_summary %>%
  mutate(scenario_key = as.character(scenario)) %>%
  left_join(resistance_targets, by = c("scenario_key" = "scenario")) %>%
  mutate(target_resistance = if_else(scenario_key == "country_timeline", resistant_fraction_start, scenario_target)) %>%
  ggplot(aes(target_resistance, resistant_fraction_start, fill = scenario_label)) +
  geom_abline(slope = 1, intercept = 0, linewidth = 0.25, linetype = "dashed", colour = manuscript_colour("mid_grey")) +
  geom_point(shape = 21, size = 2.1, stroke = 0.25, colour = "black", alpha = 0.92) +
  scale_x_continuous(labels = percent_format(accuracy = 1)) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  scale_fill_manual(values = manuscript_resistance_scenario_colours) +
  labs(x = "Target resistant fraction at analysis start", y = "Realized starting resistant fraction", fill = NULL) +
  theme_lancet()

p_ed8b <- resistance_summary %>%
  filter(!is.na(scenario_label)) %>%
  mutate(resistant_rate = resistant_infections / pmax(total_population * analysis_years, 1e-9) * 1e5) %>%
  ggplot(aes(scenario_label, country_label, fill = resistant_rate)) +
  geom_tile(colour = "white", linewidth = lancet_heatmap_tile_linewidth) +
  scale_fill_infant_burden(
    labels = label_lancet_number(accuracy = 1),
    guide = guide_colourbar(
      barwidth = grid::unit(42, "mm"),
      barheight = grid::unit(3, "mm"),
      title.position = "top"
    )
  ) +
  labs(x = NULL, y = NULL, fill = "Resistant infections per 100 000/year") +
  theme_lancet_heatmap(x_angle = 35, x_hjust = 1)

treatment_pep_data <- resistance_summary %>%
  mutate(
    `Treated cases` = treated_cases / pmax(total_population * analysis_years, 1e-9) * 1e5,
    `PEP-averted cases` = PEP_averted_cases / pmax(total_population * analysis_years, 1e-9) * 1e5
  ) %>%
  select(scenario_label, `Treated cases`, `PEP-averted cases`) %>%
  pivot_longer(-scenario_label, names_to = "metric", values_to = "rate") %>%
  filter(!is.na(scenario_label)) %>%
  group_by(scenario_label, metric) %>%
  summarise(
    median_rate = median(rate, na.rm = TRUE),
    q025 = interval_quantile(rate, 0.025),
    q975 = interval_quantile(rate, 0.975),
    q25 = interval_quantile(rate, 0.25),
    q75 = interval_quantile(rate, 0.75),
    .groups = "drop"
  )

ed8c_dodge <- position_dodge(width = 0.24)

p_ed8c <- treatment_pep_data %>%
  ggplot(aes(scenario_label, median_rate, colour = metric, group = metric)) +
  geom_errorbar(aes(ymin = q025, ymax = q975), width = 0.12, linewidth = 0.22, alpha = 0.55,
                position = ed8c_dodge) +
  geom_errorbar(aes(ymin = q25, ymax = q75), width = 0, linewidth = 0.55,
                position = ed8c_dodge) +
  geom_line(linewidth = 0.3, position = ed8c_dodge) +
  geom_point(size = 1.8, position = ed8c_dodge) +
  scale_y_continuous(labels = label_number(accuracy = 0.1)) +
  scale_colour_manual(values = manuscript_metric_colours[c("Treated cases", "PEP-averted cases")]) +
  labs(x = NULL, y = "Median events\nper 100,000/year\n(country-profile ranges)", colour = NULL) +
  theme_lancet() +
  theme(axis.text.x = element_text(angle = 35, hjust = 1))

resistance_sim <- read_model_table_optional(model_path("outputs", "simulations", "resistance_scenarios"))
require_efigure_timeseries(resistance_sim, "eFigure 6 sensitive and resistant strain trajectories", "resistance_scenarios")

if (nrow(resistance_sim) > 0) {
  resistance_ts <- resistance_sim %>%
    filter(country %in% c("Australia", "China"), scenario == "country_timeline") %>%
    add_country_label() %>%
    mutate(strain_label = factor(str_to_title(strain), levels = c("Sensitive", "Resistant"))) %>%
    group_by(country_label, strain_label, time) %>%
    summarise(
      simulation_year = first(time) / 365,
      total_population = max(total_population, na.rm = TRUE),
      infection_rate = sum(total_infection_rate_per_day, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(infection_incidence = infection_rate / pmax(total_population, 1e-9) * 365 * 1e5)

  p_ed8d <- resistance_ts %>%
    ggplot(aes(simulation_year, infection_incidence, colour = strain_label)) +
    geom_line(linewidth = 0.3) +
    facet_wrap(~country_label, scales = "free_y", nrow = 1) +
    scale_x_continuous(breaks = seq(0, 30, by = 10)) +
    scale_y_continuous(labels = label_number(accuracy = 1)) +
    labs(x = "Simulation year", y = "Infections per 100,000/year", colour = NULL)
} else {
  resistance_ts <- resistance_summary %>%
    filter(country %in% c("Australia", "China"), as.character(scenario) == "country_timeline") %>%
    mutate(
      `Sensitive` = pmax(total_infections - resistant_infections, 0) /
        pmax(total_population * analysis_years, 1e-9) * 1e5,
      `Resistant` = resistant_infections /
        pmax(total_population * analysis_years, 1e-9) * 1e5
    ) %>%
    select(country_label, Sensitive, Resistant) %>%
    pivot_longer(c(Sensitive, Resistant), names_to = "strain_label", values_to = "infection_incidence") %>%
    mutate(strain_label = factor(strain_label, levels = c("Sensitive", "Resistant")))

  p_ed8d <- resistance_ts %>%
    ggplot(aes(strain_label, infection_incidence, colour = strain_label)) +
    geom_point(size = 2.0) +
    facet_wrap(~country_label, scales = "free_y", nrow = 1) +
    scale_y_log10(labels = label_number(accuracy = 1)) +
    labs(x = NULL, y = "Infections per 100,000/year (log)", colour = NULL)
}

p_ed8d <- p_ed8d +
  scale_colour_manual(values = manuscript_metric_colours[c("Sensitive", "Resistant")]) +
  theme_lancet()

extended8 <- free(p_ed8b) + free(p_ed8c) + free(p_ed8d) +
  plot_layout(design = "AB\nCC", guides = "keep", widths = c(1.12, 0.88), heights = c(0.92, 1.08)) +
  plot_annotation(tag_levels = "A") &
  (theme(plot.margin = margin(5, 3, 3, 5)) + theme_lancet_tags())

save_appendix_figure(extended8, "extended_data_figure_6_resistance_dynamics", height = 7.4)
