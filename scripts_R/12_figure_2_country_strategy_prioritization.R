#!/usr/bin/env Rscript
## Figure 2: implementable programme strategies --------------------------------
## Layout: (a) Programme-only effect distributions
##         (b) Country-by-strategy quantitative programme effects
##         (c) Within-profile rank distributions
##         (d) Runner-up excess burden margins

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
  "maternal_immunization",
  "timeliness_only",
  "cocooning_adjunct",
  "targeted_pep_high_risk",
  "pregnancy_tdap_scaleup",
  "adolescent_booster"
)

winning_strategy_group_order <- c(
  "timeliness_only",
  "maternal_immunization",
  "targeted_pep_high_risk"
)

strategy_labels <- c(
  higher_child_coverage = "Nominal coverage floor",
  timeliness_only = "Routine schedule timeliness",
  maternal_immunization = "Infant-exposure reduction composite",
  cocooning_adjunct = "Close-contact adult adjuncts",
  targeted_pep_high_risk = "Targeted high-risk PEP",
  adolescent_booster = "Adolescent booster scale-up",
  pregnancy_tdap_scaleup = "Pregnancy Tdap scale-up"
)

strategy_overview_labels <- c(
  higher_child_coverage = "Coverage",
  maternal_immunization = "Infant-exposure\nreduction composite",
  timeliness_only = "Routine schedule\ntimeliness",
  cocooning_adjunct = "Close-contact\nadult adjuncts",
  targeted_pep_high_risk = "Targeted high-risk\nPEP",
  adolescent_booster = "Adolescent booster\nscale-up",
  pregnancy_tdap_scaleup = "Pregnancy Tdap\nscale-up"
)

strategy_heatmap_labels <- c(
  higher_child_coverage = "Coverage",
  maternal_immunization = "Infant-exposure\nreduction\ncomposite",
  timeliness_only = "Routine\nschedule\ntimeliness",
  cocooning_adjunct = "Close-contact\nadult\nadjuncts",
  targeted_pep_high_risk = "Targeted\nhigh-risk\nPEP",
  adolescent_booster = "Adolescent\nbooster\nscale-up",
  pregnancy_tdap_scaleup = "Pregnancy\nTdap\nscale-up"
)

selected_strategy_labels <- c(
  higher_child_coverage = "Coverage",
  timeliness_only = "Routine schedule\ntimeliness",
  maternal_immunization = "Infant-exposure\nreduction composite",
  cocooning_adjunct = "Close-contact\nadult adjuncts",
  targeted_pep_high_risk = "Targeted high-risk\nPEP",
  adolescent_booster = "Adolescent booster\nscale-up",
  pregnancy_tdap_scaleup = "Pregnancy Tdap\nscale-up"
)

rank_levels <- 1:6
rank_labels <- c("1st", "2nd", "3rd", "4th", "5th", "6th")
rank_colours <- setNames(
  grDevices::colorRampPalette(manuscript_continuous_burden)(length(rank_labels)),
  rank_labels
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
    strategy_overview_plot = factor(
      strategy,
      levels = rev(strategy_order),
      labels = strategy_overview_labels[rev(strategy_order)]
    )
  ) %>%
  arrange(country, primary_cases_per_100k, implementation_intensity, strategy) %>%
  group_by(country) %>%
  mutate(
    decision_rank = row_number(),
    best_cases_per_100k = first(primary_cases_per_100k),
    best_primary_case_reduction = first(primary_case_reduction),
    margin_to_best_cases_per_100k = primary_cases_per_100k - best_cases_per_100k,
    margin_to_best_percentage_points = 100 * (best_primary_case_reduction - primary_case_reduction),
    near_best_5_cases = margin_to_best_cases_per_100k <= 5,
    cases_averted_per_100k = current_primary_cases_per_100k - primary_cases_per_100k
  ) %>%
  ungroup()

selected_program <- program_frontier %>%
  arrange(country, decision_rank) %>%
  group_by(country, country_label_text) %>%
  summarise(
    current_cases_per_100k = first(current_primary_cases_per_100k),
    winning_strategy = first(strategy),
    winning_strategy_label = first(strategy_label),
    winning_strategy_short = first(as.character(strategy_overview_plot)),
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
    selected_cases_averted_per_100k = current_cases_per_100k - winning_cases_per_100k,
    selected_strategy_text = stringr::str_replace_all(winning_strategy_short, "\n", " "),
    runner_up_excess_label = paste0("+", format_lancet_fixed(winner_margin_cases_per_100k, digits = 1)),
    winning_strategy_group = factor(winning_strategy, levels = winning_strategy_group_order)
  )

heatmap_country_order <- selected_program %>%
  arrange(winning_strategy_group, desc(winner_margin_cases_per_100k), country_label_text) %>%
  pull(country_label_text)

heatmap_group_sizes <- selected_program %>%
  count(winning_strategy_group, name = "profiles") %>%
  filter(!is.na(winning_strategy_group)) %>%
  arrange(winning_strategy_group)

heatmap_group_separators <- length(heatmap_country_order) -
  cumsum(heatmap_group_sizes$profiles)[-nrow(heatmap_group_sizes)] + 0.5

selected_program <- selected_program %>%
  mutate(
    country_label_heatmap = factor(country_label_text, levels = rev(heatmap_country_order)),
    country_label_margin = factor(country_label_text, levels = rev(heatmap_country_order))
  )

selected_strategy_order <- strategy_order[strategy_order %in% selected_program$winning_strategy]
selected_strategy_legend_labels <- stringr::str_replace_all(
  selected_strategy_labels[selected_strategy_order],
  "\n",
  " "
)

## Panel A: programme-only effect distributions --------------------------------

strategy_distribution <- program_frontier %>%
  group_by(strategy, strategy_label, strategy_overview_plot) %>%
  summarise(
    median_reduction = median(primary_case_reduction, na.rm = TRUE),
    q25_reduction = interval_quantile(primary_case_reduction, 0.25),
    q75_reduction = interval_quantile(primary_case_reduction, 0.75),
    min_reduction = min(primary_case_reduction, na.rm = TRUE),
    max_reduction = max(primary_case_reduction, na.rm = TRUE),
    countries_ranked_first = sum(decision_rank == 1, na.rm = TRUE),
    countries_non_dominated = sum(non_dominated_lancet_outcome, na.rm = TRUE),
    .groups = "drop"
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
      primary_case_rank_within_programme_only = decision_rank,
      not_clearly_surpassed = non_dominated_lancet_outcome,
      lowest_burden_in_programme_only = decision_rank == 1
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
    aes(
      x = q25_reduction,
      xend = q75_reduction,
      y = strategy_overview_plot,
      yend = strategy_overview_plot,
      linetype = "IQR"
    ),
    linewidth = 1.0,
    colour = manuscript_colour("grey"),
    alpha = 0.72
  ) +
  geom_point(
    data = program_frontier,
    aes(primary_case_reduction, strategy_overview_plot, colour = strategy, shape = "Profiles"),
    alpha = 0.50,
    size = 1.25,
    position = position_jitter(height = 0.08, width = 0)
  ) +
  geom_point(
    data = strategy_distribution,
    aes(median_reduction, strategy_overview_plot, shape = "Median"),
    fill = manuscript_colour("black"),
    colour = "white",
    stroke = 0.20,
    size = 2.2
  ) +
  scale_x_continuous(labels = label_lancet_percent(accuracy = 1),
                     limits = c(-0.10, 0.5),
                     breaks = c(0, 0.25, 0.5)) +
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
        linewidth = 1.0,
        alpha = 0.72
      )
    )
  ) +
  labs(
    x = "Case reduction in people aged <18 years",
    y = NULL,
    tag = "a"
  ) +
  theme_lancet_panel(base_size = journal_base_text_size, plot_margin = margin(4, 7, 4, 4), show_y_grid = TRUE) +
  theme(
    axis.text.y = element_text(lineheight = 0.88),
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

## Panel B: country-by-strategy programme effects ------------------------------

program_heatmap <- program_frontier %>%
  mutate(
    country_label = factor(country_label_text, levels = rev(heatmap_country_order)),
    strategy_axis = factor(strategy, levels = strategy_order),
    effect_label = lancet_percent(primary_case_reduction, accuracy = 1),
    effect_text_colour = if_else(primary_case_reduction >= 0.28, "white", lancet_text_colour),
    preferred_in_program_only = decision_rank == 1
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
      primary_case_rank_within_programme_only = decision_rank,
      margin_to_best_cases_per_100k,
      margin_to_best_percentage_points,
      near_best_5_cases,
      lowest_burden_in_programme_only = preferred_in_program_only,
      not_clearly_surpassed = non_dominated_lancet_outcome
    ),
  model_path("outputs", "tables", "figure2b_programme_country_strategy_matrix.csv")
)

p2b <- ggplot(program_heatmap, aes(x = strategy_axis, y = country_label, fill = primary_case_reduction)) +
  geom_tile(colour = "white", linewidth = lancet_heatmap_tile_linewidth) +
  geom_hline(
    yintercept = heatmap_group_separators,
    linewidth = 0.48,
    colour = "white"
  ) +
  geom_tile(
    data = filter(program_heatmap, preferred_in_program_only),
    fill = NA,
    colour = manuscript_colour("black"),
    linewidth = 0.30
  ) +
  geom_text(
    aes(label = effect_label, colour = effect_text_colour),
    size = journal_heatmap_cell_text_size_small,
    lineheight = 0.88
  ) +
  scale_x_discrete(
    labels = strategy_heatmap_labels[strategy_order]
  ) +
  scale_fill_reduction(
    midpoint = 0,
    limits = c(-0.10, 0.5),
    breaks = c(-0.10, 0, 0.25, 0.5),
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

## Panel C: within-profile rank distribution -----------------------------------

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

rank_distribution <- program_frontier %>%
  count(strategy, decision_rank, name = "profile_count") %>%
  tidyr::complete(strategy = strategy_order, decision_rank = rank_levels, fill = list(profile_count = 0L)) %>%
  mutate(
    strategy_label = strategy_labels[strategy],
    rank_label = factor(decision_rank, levels = rank_levels, labels = rank_labels),
    strategy_y = as.numeric(factor(strategy, levels = rev(strategy_order))),
    segment_text_colour = if_else(decision_rank %in% c(1L, 6L), "white", lancet_text_colour)
  ) %>%
  arrange(strategy, decision_rank) %>%
  group_by(strategy) %>%
  mutate(
    xmin = cumsum(lag(profile_count, default = 0L)),
    xmax = xmin + profile_count,
    xmid = (xmin + xmax) / 2,
    segment_label = if_else(profile_count > 0, as.character(profile_count), NA_character_),
    rank_total_profiles = sum(profile_count)
  ) %>%
  ungroup()

readr::write_csv(
  rank_distribution %>%
    transmute(
      strategy,
      strategy_label,
      within_profile_rank = decision_rank,
      rank_label = as.character(rank_label),
      profile_count,
      total_profiles = rank_total_profiles
    ),
  model_path("outputs", "tables", "figure2c_programme_rank_distribution.csv")
)

p2c <- ggplot(rank_distribution) +
  geom_rect(
    aes(
      xmin = xmin,
      xmax = xmax,
      ymin = strategy_y - 0.32,
      ymax = strategy_y + 0.32,
      fill = rank_label
    ),
    colour = "white",
    linewidth = 0.18
  ) +
  geom_text(
    data = filter(rank_distribution, profile_count > 0),
    aes(x = xmid, y = strategy_y, label = segment_label, colour = segment_text_colour),
    size = journal_heatmap_cell_text_size_small,
    lineheight = 0.90,
    show.legend = FALSE
  ) +
  scale_x_continuous(
    breaks = seq(0, 10, by = 2),
    labels = label_lancet_number(accuracy = 1),
    expand = expansion(mult = c(0, 0.02))
  ) +
  scale_y_continuous(
    breaks = seq_along(rev(strategy_order)),
    labels = unname(strategy_overview_labels[rev(strategy_order)]),
    expand = expansion(add = c(0.45, 0.45))
  ) +
  scale_fill_manual(
    values = rank_colours,
    breaks = rank_labels,
    drop = FALSE,
    name = "Rank",
    guide = guide_legend(
      nrow = 1,
      title.position = "left",
      keywidth = unit(0.28, "cm"),
      keyheight = unit(0.18, "cm")
    )
  ) +
  scale_colour_identity() +
  coord_cartesian(xlim = c(0, 10), clip = "off") +
  labs(
    x = "Profiles (n=10)",
    y = NULL,
    tag = "c"
  ) +
  theme_lancet_panel(base_size = journal_base_text_size, plot_margin = margin(4, 8, 4, 7), show_y_grid = TRUE) +
  theme(
    axis.text.y = element_text(lineheight = 0.88),
    legend.position = "top",
    legend.justification = "right",
    legend.box.just = "right",
    legend.direction = "horizontal",
    legend.title = element_text(face = "bold", lineheight = 0.90),
    legend.text = element_text(lineheight = 0.92),
    legend.box.margin = margin(0, 0, 1, 0),
    legend.margin = margin(0, 0, 0, 0),
    legend.spacing.x = unit(2, "pt")
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
  aes(winner_margin_cases_per_100k, country_label_margin)
) +
  annotate(
    "rect",
    xmin = 0,
    xmax = 5,
    ymin = -Inf,
    ymax = Inf,
    fill = manuscript_colour("light_grey"),
    alpha = 0.45
  ) +
  geom_vline(xintercept = 5, linewidth = 0.24, linetype = "dashed", colour = manuscript_colour("grey")) +
  geom_vline(xintercept = 25, linewidth = 0.20, linetype = "dotted", colour = manuscript_colour("grey")) +
  geom_text(
    data = data.frame(
      label_x = 2.5,
      country_label_margin = factor(heatmap_country_order[[1]], levels = rev(heatmap_country_order)),
      label_text = "near-tie \u22645"
    ),
    aes(x = label_x, y = country_label_margin, label = label_text),
    inherit.aes = FALSE,
    size = lancet_pt_to_geom_size(7.2),
    fontface = "bold",
    colour = manuscript_colour("grey"),
    hjust = 0.5,
    vjust = -2.8
  ) +
  geom_segment(
    aes(x = 0, xend = winner_margin_cases_per_100k, yend = country_label_margin, colour = winning_strategy),
    linewidth = 0.58,
    lineend = "round",
    alpha = 0.90
  ) +
  geom_point(
    aes(fill = winning_strategy),
    shape = 21,
    size = 2.3,
    stroke = 0.22,
    colour = "white"
  ) +
  geom_text(
    aes(label = runner_up_excess_label),
    hjust = -0.12,
    size = journal_heatmap_cell_text_size_small,
    colour = manuscript_colour("black")
  ) +
  scale_x_continuous(
    breaks = c(0, 5, 25, 50, 100),
    labels = label_lancet_comma(accuracy = 1),
    expand = expansion(mult = c(0, 0.10))
  ) +
  scale_fill_manual(
    values = strategy_colours,
    breaks = selected_strategy_order,
    labels = selected_strategy_legend_labels,
    name = "Lowest-burden option",
    guide = guide_legend(
      ncol = 1,
      title.position = "top",
      keywidth = unit(0.36, "cm"),
      keyheight = unit(0.46, "cm"),
      override.aes = list(alpha = 0.90)
    )
  ) +
  scale_colour_manual(values = strategy_colours, guide = "none") +
  scale_y_discrete(limits = rev(heatmap_country_order)) +
  coord_cartesian(xlim = c(0, max(selected_program$winner_margin_cases_per_100k, na.rm = TRUE) * 1.28), clip = "off") +
  labs(
    x = "Extra annualised symptomatic cases per 100 000\npeople aged <18 years if second-ranked strategy used",
    y = NULL,
    tag = "d"
  ) +
  theme_lancet_panel(base_size = journal_dense_text_size, plot_margin = margin(12, 4, 4, 7), show_y_grid = TRUE) +
  theme(
    axis.text.y = element_text(lineheight = 0.88),
    axis.ticks.y = element_blank(),
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

## Compose Figure 2 ------------------------------------------------------------

design <- "
ABB
CCD
"

figure2 <- p2a + free(p2b) + p2c + p2d +
  plot_layout(widths = c(0.55, 0.42, 1.36), design = design, guides = "keep") &
  theme_lancet_tags()

save_main_figure(figure2, "figure_2_country_strategy_prioritization", height = 6.1)
cat("Figure 2 saved.\n")
