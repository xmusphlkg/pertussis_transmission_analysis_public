#!/usr/bin/env Rscript
## Figure 5: future vaccine mechanism targets ---------------------------------
## Layout: (a) Vaccine mechanism matrix
##         (b) Infection reduction vs <18 case reduction for vaccine mechanisms
##         (c) Residual <18 cases under mechanism targets and stress-test upper bound
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

## Panel A: vaccine mechanism matrix ------------------------------------------

scenario_params <- readr::read_csv(
  model_path("publication_inputs", "scenario_table.csv"),
  show_col_types = FALSE
) %>%
  filter(scenario %in% vaccine_levels) %>%
  mutate(scenario_label = factor(vaccine_profile_labels[scenario], levels = vaccine_profile_labels[vaccine_levels])) %>%
  select(scenario_label, VE_sus, VE_sym, VE_inf, VE_dur) %>%
  pivot_longer(-scenario_label, names_to = "parameter", values_to = "value") %>%
  mutate(
    parameter = factor(
      parameter,
      levels = c("VE_sus", "VE_sym", "VE_inf", "VE_dur"),
      labels = c("Susceptibility", "Symptoms", "Infectiousness", "Duration")
    ),
    text_colour = if_else(value <= 0.20 | value >= 0.70, "white", "black")
  )

readr::write_csv(
  scenario_params %>%
    transmute(
      scenario_label = as.character(scenario_label),
      parameter = as.character(parameter),
      value
    ),
  model_path("outputs", "tables", "figure5a_vaccine_mechanism_matrix.csv")
)

p5a <- ggplot(scenario_params, aes(parameter, scenario_label, fill = value)) +
  geom_tile(colour = "white", linewidth = lancet_heatmap_tile_linewidth) +
  geom_text(aes(label = format_lancet_fixed(value, digits = 2), colour = text_colour),
    size = journal_table_cell_text_size,
    fontface = "bold"
  ) +
  scale_fill_fraction(
    limits = c(0, 1),
    breaks = seq(0, 1, by = 0.25),
    labels = label_lancet_percent(accuracy = 1),
    name = "Effect size",
    guide = "none",
    oob = scales::squish
  ) +
  scale_colour_identity(guide = "none") +
  scale_y_discrete(labels = vaccine_profile_axis_labels) +
  labs(x = "Vaccine effect on", y = NULL, tag = "a") +
  theme_lancet_heatmap(base_size = journal_compact_text_size)

## Panels B-C: vaccine mechanism outcomes -------------------------------------

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

vaccine_tradeoff <- vaccine_mechanism_outcomes %>%
  filter(scenario %in% vaccine_target_levels)

readr::write_csv(
  vaccine_tradeoff %>%
    transmute(
      country,
      scenario,
      scenario_label = as.character(scenario_label),
      relative_reduction_total_infections,
      primary_case_reduction
    ),
  model_path("outputs", "tables", "figure5b_child_adolescent_vaccine_tradeoff.csv")
)

p5b <- ggplot(
  vaccine_tradeoff,
  aes(relative_reduction_total_infections, primary_case_reduction, colour = scenario_label)
) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", linewidth = 0.25, colour = manuscript_colour("mid_grey")) +
  geom_point(size = 1.65, alpha = 0.76) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.25), labels = label_lancet_percent(accuracy = 1), expand = expansion(mult = c(0, 0))) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.25), labels = label_lancet_percent(accuracy = 1), expand = expansion(mult = c(0, 0))) +
  scale_colour_vaccine_profiles() +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), clip = "off") +
  labs(
    x = "All-infection reduction\nvs current aP",
    y = "<18 case reduction\nvs current aP",
    tag = "b"
  ) +
  theme_lancet(base_size = journal_compact_text_size) +
  theme(
    plot.margin = margin(5, 8, 4, 4),
    panel.grid.major.x = element_line(linewidth = 0.16, colour = lancet_grid_light_colour),
    legend.position = "none"
  )

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
  model_path("outputs", "tables", "figure5c_residual_child_adolescent_cases.csv")
)

p5c <- ggplot(vaccine_residual, aes(primary_cases_per_100k, scenario_label, colour = scenario_label)) +
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

## Panel D: infectiousness-effect response curves ------------------------------

veinf_grid <- read_summary("veinf_resistance_grid_summary.csv") %>%
  transmute(
    country = stringr::str_replace_all(country, " ", "_"),
    grid_VE_inf = round(as.numeric(grid_VE_inf), 1),
    grid_resistance_prevalence = round(as.numeric(grid_resistance_prevalence), 2),
    primary_cases_per_100k = as.numeric(annualized_child_adolescent_cases_per_100k),
    all_infections_per_100k = as.numeric(annualized_infections_per_100k)
  ) %>%
  filter(
    grid_resistance_prevalence %in% c(0, 0.5, 1),
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
      levels = paste0(lancet_percent(c(0, 0.5, 1), accuracy = 1), " resistant")
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
  model_path("outputs", "tables", "figure5d_veinf_response_curve.csv")
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

p5d <- ggplot(
  veinf_response_summary,
  aes(grid_VE_inf, median_case_reduction, colour = resistance_label, fill = resistance_label)
) +
  geom_ribbon(
    aes(ymin = q25_case_reduction, ymax = q75_case_reduction),
    alpha = 0.14,
    colour = NA
  ) +
  geom_line(linewidth = 0.62) +
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
    labels = veinf_response_labels,
    name = "Resistant fraction",
    guide = guide_legend(nrow = 1, byrow = TRUE, override.aes = list(alpha = 1, linewidth = 0.7, size = 2))
  ) +
  scale_fill_manual(values = veinf_response_colours, guide = "none") +
  coord_cartesian(clip = "off") +
  labs(
    x = expression(bold("Vaccine infectiousness effect (") * bolditalic(VE)[bold(inf)] * bold(")")),
    y = "Relative reduction in <18 cases\nvs no VEinf effect",
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

## Compose Figure 5 ------------------------------------------------------------

figure5 <- p5a + p5b + p5c + free(p5d) +
  plot_layout(
    design = "
AB
CD
",
    widths = c(0.96, 1.04),
    heights = c(0.95, 1.05),
    guides = "keep"
  ) &
  theme_lancet_tags()

save_main_figure(figure5, "figure_5_future_vaccine_mechanisms", height = 7.0)
cat("Figure 5 saved.\n")
