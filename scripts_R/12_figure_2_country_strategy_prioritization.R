#!/usr/bin/env Rscript
## Figure 2: implementable programme strategies --------------------------------
## Layout: (a) Programme-only effect distributions
##         (b) Country-by-strategy quantitative programme effects
##         (c) Absolute burden after the lowest-burden programme
##         (d) Runner-up excess burden

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

strategy_order <- c(
  "timeliness_only",
  "maternal_immunization",
  "cocooning_adjunct",
  "targeted_pep_high_risk",
  "adolescent_booster",
  "pregnancy_tdap_scaleup"
)

strategy_labels <- c(
  higher_child_coverage = "Nominal coverage floor",
  timeliness_only = "Routine timeliness",
  maternal_immunization = "Infant-exposure reduction composite",
  cocooning_adjunct = "Close-contact adult adjunct",
  targeted_pep_high_risk = "Targeted high-risk PEP",
  adolescent_booster = "Adolescent booster",
  pregnancy_tdap_scaleup = "Pregnancy Tdap scale-up"
)

strategy_short <- c(
  higher_child_coverage = "Coverage",
  timeliness_only = "Timeliness",
  maternal_immunization = "Infant\nexposure\ncomposite",
  cocooning_adjunct = "Adult/contact",
  targeted_pep_high_risk = "Targeted\nPEP",
  adolescent_booster = "Adolescent",
  pregnancy_tdap_scaleup = "Pregnancy\nTdap"
)

strategy_heatmap_labels <- c(
  higher_child_coverage = "Coverage",
  timeliness_only = "Timeliness",
  maternal_immunization = "Infant\nexposure\ncomposite",
  cocooning_adjunct = "Adult/\ncontact",
  targeted_pep_high_risk = "Targeted\nPEP",
  adolescent_booster = "Adolescent",
  pregnancy_tdap_scaleup = "Pregnancy\nTdap"
)

selected_strategy_labels <- c(
  higher_child_coverage = "Coverage",
  timeliness_only = "Timeliness",
  maternal_immunization = "Infant exposure\ncomposite",
  cocooning_adjunct = "Adult/contact",
  targeted_pep_high_risk = "Targeted PEP",
  adolescent_booster = "Adolescent",
  pregnancy_tdap_scaleup = "Pregnancy Tdap"
)

frontier_raw <- read_table("lancet_child_adolescent_decision_frontier.csv") %>%
  mutate(
    country = stringr::str_replace_all(country, " ", "_"),
    country_label_text = format_country(country),
    strategy = as.character(strategy),
    strategy_label = coalesce(strategy_labels[strategy], strategy),
    primary_case_reduction = as.numeric(primary_case_reduction),
    primary_cases_per_100k = as.numeric(primary_cases_per_100k),
    current_primary_cases_per_100k = as.numeric(current_primary_cases_per_100k),
    implementation_intensity = as.numeric(implementation_intensity),
    non_dominated_lancet_outcome = as.logical(non_dominated_lancet_outcome),
    primary_case_rank_within_constraint = as.numeric(primary_case_rank_within_constraint)
  )

country_order <- frontier_raw %>%
  filter(strategy == "current", optimization_constraint == "program_only") %>%
  arrange(desc(current_primary_cases_per_100k)) %>%
  pull(country_label_text)

program_frontier <- frontier_raw %>%
  filter(optimization_constraint == "program_only", strategy %in% strategy_order) %>%
  mutate(
    country_label = factor(country_label_text, levels = rev(country_order)),
    strategy_factor = factor(strategy, levels = strategy_order),
    strategy_label_plot = factor(strategy_labels[strategy], levels = rev(strategy_labels[strategy_order])),
    strategy_short_plot = factor(strategy_short[strategy], levels = strategy_short[strategy_order])
  )

## Panel A: programme-only effect distributions --------------------------------

strategy_distribution <- program_frontier %>%
  group_by(strategy, strategy_label_plot, strategy_short_plot) %>%
  summarise(
    median_reduction = median(primary_case_reduction, na.rm = TRUE),
    q25_reduction = interval_quantile(primary_case_reduction, 0.25),
    q75_reduction = interval_quantile(primary_case_reduction, 0.75),
    min_reduction = min(primary_case_reduction, na.rm = TRUE),
    max_reduction = max(primary_case_reduction, na.rm = TRUE),
    countries_ranked_first = sum(primary_case_rank_within_constraint == 1, na.rm = TRUE),
    countries_non_dominated = sum(non_dominated_lancet_outcome, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    strategy_label_plot = fct_reorder(strategy_label_plot, median_reduction),
    strategy_overview_plot = fct_reorder(strategy_short_plot, median_reduction)
  )

program_frontier <- program_frontier %>%
  left_join(
    strategy_distribution %>%
      transmute(
        strategy,
        strategy_label_plot_ordered = strategy_label_plot,
        strategy_overview_plot
      ),
    by = "strategy"
  ) %>%
  mutate(
    strategy_label_plot = strategy_label_plot_ordered
  )

readr::write_csv(
  program_frontier %>%
    transmute(
      country,
      strategy,
      strategy_label,
      primary_case_reduction,
      primary_cases_per_100k,
      implementation_intensity,
      not_clearly_surpassed = non_dominated_lancet_outcome,
      lowest_burden_in_programme_only = primary_case_rank_within_constraint == 1
    ),
  model_path("outputs", "tables", "figure2a_programme_effects.csv")
)

p2a <- ggplot() +
  geom_vline(xintercept = 0, linewidth = 0.24, colour = manuscript_colour("pale_grey")) +
  geom_segment(
    data = strategy_distribution,
    aes(x = min_reduction, xend = max_reduction, y = strategy_overview_plot, yend = strategy_overview_plot),
    linewidth = 0.25,
    colour = manuscript_colour("pale_grey")
  ) +
  geom_segment(
    data = strategy_distribution,
    aes(x = q25_reduction, xend = q75_reduction, y = strategy_overview_plot, yend = strategy_overview_plot),
    linewidth = 1.0,
    colour = manuscript_colour("grey"),
    alpha = 0.72
  ) +
  geom_point(
    data = program_frontier,
    aes(primary_case_reduction, strategy_overview_plot, colour = strategy),
    alpha = 0.50,
    size = 1.25,
    position = position_jitter(height = 0.08, width = 0)
  ) +
  geom_point(
    data = strategy_distribution,
    aes(median_reduction, strategy_overview_plot),
    shape = 23,
    fill = manuscript_colour("black"),
    colour = "white",
    stroke = 0.20,
    size = 2.2
  ) +
  scale_x_continuous(labels = label_lancet_percent(accuracy = 1),
                     limits = c(-0.25, 0.5),
                     breaks = c(-0.25, 0, 0.25, 0.5)) +
  scale_colour_manual(values = strategy_colours, guide = "none") +
  labs(
    x = "Case reduction in people aged <18 years",
    y = NULL,
    tag = "a"
  ) +
  theme_lancet_panel(base_size = journal_base_text_size, plot_margin = margin(4, 7, 4, 4), show_y_grid = TRUE) +
  theme(
    axis.text.y = element_text(lineheight = 0.88)
  )

## Panel B: country-by-strategy programme effects ------------------------------

program_heatmap <- program_frontier %>%
  mutate(
    country_label = factor(country_label_text, levels = rev(country_order)),
    strategy_axis = factor(strategy, levels = strategy_order),
    effect_label = lancet_percent(primary_case_reduction, accuracy = 1),
    effect_text_colour = if_else(primary_case_reduction >= 0.28, "white", "black"),
    preferred_in_program_only = primary_case_rank_within_constraint == 1
  )

readr::write_csv(
  program_heatmap %>%
    transmute(
      country,
      strategy,
      strategy_label,
      primary_case_reduction,
      primary_cases_per_100k,
      current_primary_cases_per_100k,
      implementation_intensity,
      primary_case_rank_within_constraint,
      lowest_burden_in_programme_only = preferred_in_program_only,
      not_clearly_surpassed = non_dominated_lancet_outcome
    ),
  model_path("outputs", "tables", "figure2b_programme_country_strategy_matrix.csv")
)

p2b <- ggplot(program_heatmap, aes(x = strategy_axis, y = country_label, fill = primary_case_reduction)) +
  geom_tile(colour = "white", linewidth = lancet_heatmap_tile_linewidth) +
  geom_text(
    aes(label = effect_label, colour = effect_text_colour),
    size = journal_heatmap_cell_text_size_small,
    lineheight = 0.88
  ) +
  geom_tile(
    data = filter(program_heatmap, preferred_in_program_only),
    fill = NA,
    colour = manuscript_colour("black"),
    linewidth = 0.42
  ) +
  scale_x_discrete(
    labels = strategy_heatmap_labels[strategy_order]
  ) +
  scale_fill_reduction(
    midpoint = 0,
    limits = c(-0.25, 0.5),
    breaks = seq(-0.25, 0.5, 0.25),
    labels = label_lancet_percent(accuracy = 1),
    oob = scales::squish,
    name = "Case reduction",
    guide = guide_lancet_colourbar(barwidth = unit(4.2, "cm"),
                                   barheight = unit(0.18, "cm"),
                                   title.position = "left")
  ) +
  scale_colour_identity() +
  labs(x = NULL, y = NULL, tag = "b") +
  theme_lancet_heatmap(base_size = journal_dense_text_size, plot_margin = margin(4, 4, 3, 7)) +
  theme(
    legend.title = element_text(vjust = 1)
  )

## Panel C: lowest-burden programme absolute burden ---------------------------

ranked_program <- program_frontier %>%
  arrange(country, primary_cases_per_100k, implementation_intensity, strategy) %>%
  group_by(country) %>%
  mutate(decision_rank = row_number()) %>%
  ungroup()

selected_program <- ranked_program %>%
  group_by(country, country_label_text, country_label) %>%
  summarise(
    current_cases_per_100k = first(current_primary_cases_per_100k),
    winning_strategy = first(strategy),
    winning_strategy_label = first(strategy_label),
    winning_strategy_short = first(as.character(strategy_short_plot)),
    winning_reduction = first(primary_case_reduction),
    winning_cases_per_100k = first(primary_cases_per_100k),
    runner_up_strategy = nth(strategy, 2),
    runner_up_strategy_label = nth(strategy_label, 2),
    runner_up_reduction = nth(primary_case_reduction, 2),
    runner_up_cases_per_100k = nth(primary_cases_per_100k, 2),
    winner_margin_reduction = winning_reduction - runner_up_reduction,
    winner_margin_cases_per_100k = runner_up_cases_per_100k - winning_cases_per_100k,
    .groups = "drop"
  ) %>%
  mutate(
    country_label = factor(country_label_text, levels = rev(country_order)),
    selected_cases_averted_per_100k = current_cases_per_100k - winning_cases_per_100k,
    selected_strategy_text = stringr::str_replace_all(winning_strategy_short, "\n", " "),
    cases_averted_label = paste0(format_lancet_fixed(selected_cases_averted_per_100k, digits = 0), " fewer"),
    cases_averted_label_x = if_else(winning_cases_per_100k < 120, winning_cases_per_100k + 18, winning_cases_per_100k - 8),
    cases_averted_label_hjust = if_else(winning_cases_per_100k < 120, 0, 1),
    runner_up_excess_label = paste0("+", format_lancet_fixed(winner_margin_cases_per_100k, digits = 1))
  )

selected_strategy_order <- strategy_order[strategy_order %in% selected_program$winning_strategy]
selected_strategy_legend_labels <- selected_strategy_labels[selected_strategy_order]

readr::write_csv(
  selected_program %>%
    transmute(
      country,
      current_cases_per_100k,
      lowest_burden_strategy = winning_strategy,
      lowest_burden_strategy_label = winning_strategy_label,
      lowest_burden_reduction = winning_reduction,
      lowest_burden_cases_per_100k = winning_cases_per_100k,
      cases_averted_per_100k_with_lowest_burden_option = selected_cases_averted_per_100k,
      runner_up_strategy,
      runner_up_strategy_label,
      runner_up_reduction,
      runner_up_cases_per_100k,
      runner_up_excess_reduction = winner_margin_reduction,
      runner_up_excess_percentage_points = 100 * winner_margin_reduction,
      runner_up_excess_cases_per_100k = winner_margin_cases_per_100k
    ),
  model_path("outputs", "tables", "figure2c_programme_selected_burden.csv")
)

p2c <- ggplot(selected_program, aes(y = country_label)) +
  geom_segment(
    aes(x = current_cases_per_100k, xend = winning_cases_per_100k, yend = country_label, colour = winning_strategy),
    linewidth = 0.72,
    lineend = "round"
  ) +
  geom_point(
    aes(x = current_cases_per_100k, shape = "Current practice"),
    fill = "white",
    colour = manuscript_colour("mid_grey"),
    size = 1.55,
    stroke = 0.32
  ) +
  geom_point(
	    aes(x = winning_cases_per_100k, fill = winning_strategy, shape = "Lowest-burden option"),
    colour = manuscript_colour("black"),
    size = 1.85,
    stroke = 0.28
  ) +
  scale_x_continuous(labels = label_lancet_comma(accuracy = 1), breaks = seq(0, 1100, by = 250)) +
  scale_colour_manual(values = strategy_colours, guide = "none") +
  scale_fill_manual(values = strategy_colours, guide = "none") +
  scale_shape_manual(
    values = c("Current practice" = 21, "Lowest-burden option" = 21),
    name = NULL,
    guide = guide_legend(
      order = 2,
      position = 'top',
      override.aes = list(
        fill = c("white", manuscript_colour("black")),
        colour = c(manuscript_colour("mid_grey"), manuscript_colour("black"))
      )
    )
  ) +
  coord_cartesian(xlim = c(0, max(selected_program$current_cases_per_100k, na.rm = TRUE) * 1.05), clip = "off") +
  labs(
    x = "Annual cases per 100 000 people aged <18 years",
    y = NULL,
    tag = "c"
  ) +
  theme_lancet_panel(base_size = journal_dense_text_size, plot_margin = margin(4, 14, 4, 7), show_y_grid = TRUE) +
  theme(
    axis.text.y = element_text(lineheight = 0.88)
  )

## Panel D: runner-up excess burden -------------------------------------------

readr::write_csv(
  selected_program %>%
    transmute(
      country,
      lowest_burden_strategy = winning_strategy,
      lowest_burden_strategy_label = winning_strategy_label,
      runner_up_strategy,
      runner_up_strategy_label,
      lowest_burden_cases_per_100k = winning_cases_per_100k,
      runner_up_cases_per_100k,
      runner_up_excess_cases_per_100k = winner_margin_cases_per_100k,
      runner_up_excess_reduction = winner_margin_reduction,
      runner_up_excess_percentage_points = 100 * winner_margin_reduction
    ),
  model_path("outputs", "tables", "figure2d_programme_runner_up_excess.csv")
)

p2d <- ggplot(
  selected_program,
  aes(winner_margin_cases_per_100k, country_label)
) +
  geom_col(
    aes(fill = winning_strategy),
    width = 0.56,
    alpha = 0.90
  ) +
  geom_text(
    aes(label = runner_up_excess_label),
    hjust = -0.12,
    size = journal_heatmap_cell_text_size_small,
    colour = manuscript_colour("black")
  ) +
  scale_x_continuous(
    breaks = seq(0, 150, by = 50),
    labels = label_lancet_comma(accuracy = 1),
    expand = expansion(mult = c(0, 0.10))
  ) +
  scale_fill_manual(
    values = strategy_colours,
    breaks = selected_strategy_order,
    labels = selected_strategy_legend_labels,
	    name = "Lowest-burden\noption",
    guide = guide_legend(
      nrow = 1,
      title.position = "left",
      keywidth = unit(0.36, "cm"),
      keyheight = unit(0.16, "cm"),
      override.aes = list(alpha = 0.90)
    )
  ) +
  coord_cartesian(xlim = c(0, max(selected_program$winner_margin_cases_per_100k, na.rm = TRUE) * 1.18), clip = "off") +
  labs(
	    x = "Extra cases per 100 000 people aged <18 years\nif runner-up used",
    y = NULL,
    tag = "d"
  ) +
  theme_lancet_panel(base_size = journal_dense_text_size, plot_margin = margin(4, 4, 4, 7), show_y_grid = TRUE) +
  theme(
    axis.text.y = element_text(lineheight = 0.88),
    axis.ticks.y = element_blank(),
    legend.position = "top",
    legend.direction = "horizontal",
    legend.title = element_text(face = "bold", lineheight = 0.90),
    legend.text = element_text(lineheight = 0.92),
    legend.box.margin = margin(0, 0, 1, 0),
    legend.margin = margin(0, 0, 0, 0)
  )

## Compose Figure 2 ------------------------------------------------------------

design <- "
ABB
CCD
"

figure2 <- p2a + free(p2b) + p2c + p2d +
  plot_layout(widths = c(0.46, 0.4, 1.40), design = design, guides = "keep") &
  theme_lancet_tags()

save_main_figure(figure2, "figure_2_country_strategy_prioritization", height = 5.8)
cat("Figure 2 saved.\n")
