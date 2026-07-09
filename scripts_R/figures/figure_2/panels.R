## Figure 2 panels -------------------------------------------------------------

plot_figure_2_panel_a <- function(data) {
  ggplot() +
    geom_vline(xintercept = 0, linewidth = 0.24, colour = manuscript_colour("pale_grey")) +
    geom_segment(
      data = data$strategy_distribution,
      aes(x = min_reduction, xend = max_reduction, y = strategy_overview_plot, yend = strategy_overview_plot),
      linewidth = 0.25,
      colour = manuscript_colour("pale_grey")
    ) +
    geom_segment(
      data = data$strategy_distribution,
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
      data = data$program_frontier,
      aes(primary_case_reduction, strategy_overview_plot, colour = strategy, shape = "Profiles"),
      alpha = 0.50,
      size = 1.25,
      position = position_jitter(height = 0.08, width = 0)
    ) +
    geom_point(
      data = data$strategy_distribution,
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
    scale_shape_profile_median() +
    scale_linetype_iqr(label = "Cross-profile IQR", linewidth = 1.0, alpha = 0.72) +
    labs(
      x = "Case reduction in people aged <18 years",
      y = NULL,
      tag = "a"
    ) +
    theme_lancet_panel(base_size = journal_base_text_size, plot_margin = margin(4, 7, 4, 4), show_y_grid = TRUE) +
    theme(
      axis.text.y = element_text(lineheight = 0.88)
    ) +
    theme_lancet_inside_legend(
      key_height = unit(0.46, "cm"),
      title_lineheight = 0.90,
      text_lineheight = 1.16,
      text_margin = margin(t = 3, b = 3),
      spacing_y = unit(8, "pt"),
      box_margin = margin(0, 0, 1, 0),
      legend_margin = margin(3, 0, 3, 0)
    )
}

plot_figure_2_panel_b <- function(data) {
  ggplot(
    data$selected_program,
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
        country_label_margin = factor(data$heatmap_country_order[[1]], levels = rev(data$heatmap_country_order)),
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
      breaks = data$selected_strategy_order,
      labels = data$selected_strategy_legend_labels,
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
    scale_y_discrete(limits = rev(data$heatmap_country_order)) +
    coord_cartesian(xlim = c(0, max(data$selected_program$winner_margin_cases_per_100k, na.rm = TRUE) * 1.28), clip = "off") +
    labs(
      x = "Extra annualised symptomatic cases per 100 000\npeople aged <18 years if second-ranked strategy used",
      y = NULL,
      tag = "b"
    ) +
    theme_lancet_panel(base_size = journal_dense_text_size, plot_margin = margin(12, 4, 4, 7), show_y_grid = TRUE) +
    theme(
      axis.text.y = element_text(lineheight = 0.88),
      axis.ticks.y = element_blank(),
    ) +
    theme_lancet_inside_legend(
      key_height = unit(0.46, "cm"),
      title_lineheight = 0.90,
      text_lineheight = 1.16,
      text_margin = margin(t = 3, b = 3),
      spacing_y = unit(8, "pt"),
      box_margin = margin(0, 0, 1, 0),
      legend_margin = margin(3, 0, 3, 0)
    )
}

plot_figure_2_panel_c <- function(data) {
  ggplot(data$program_heatmap, aes(x = strategy_axis, y = country_label, fill = primary_case_reduction)) +
    geom_tile(colour = "white", linewidth = lancet_heatmap_tile_linewidth) +
    geom_hline(
      yintercept = data$heatmap_group_separators,
      linewidth = 0.48,
      colour = "white"
    ) +
    geom_tile(
      data = filter(data$program_heatmap, preferred_in_program_only),
      fill = NA,
      colour = manuscript_colour("black"),
      linewidth = 0.30
    ) +
    geom_text(
      aes(label = effect_label, colour = effect_text_colour),
      size = journal_heatmap_cell_text_size_small - 0.30,
      lineheight = 0.82
    ) +
    scale_x_discrete(
      labels = data$strategy_heatmap_labels[data$strategy_order]
    ) +
    scale_fill_reduction(
      midpoint = 0,
      limits = c(-0.10, 0.5),
      breaks = seq(-0.10, 0.50, by = 0.10),
      labels = label_lancet_percent(accuracy = 1),
      oob = scales::squish,
      name = "Median [95% PI] of reduction",
      guide = guide_lancet_colourbar(
        barwidth = unit(0.30, "cm"),
        barheight = unit(5.0, "cm"),
        title.position = "left",
        title.hjust = 0
      )
    ) +
    scale_colour_identity() +
    labs(x = NULL, y = NULL, tag = "c") +
    theme_lancet_heatmap(
      base_size = journal_dense_text_size,
      plot_margin = margin(4, 4, 3, 7),
      legend_position = "right",
      legend_direction = "vertical"
    ) +
    theme(
      legend.title = element_text(angle = 90, hjust = 0, vjust = 0.5, lineheight = 0.92),
      legend.text = element_text(lineheight = 0.95),
      legend.title.position = "left"
    )
}

plot_figure_2_rank_distribution <- function(data) {
  ggplot(data$rank_distribution) +
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
      data = filter(data$rank_distribution, profile_count > 0),
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
      breaks = seq_along(rev(data$strategy_order)),
      labels = unname(data$strategy_overview_labels[rev(data$strategy_order)]),
      expand = expansion(add = c(0.45, 0.45))
    ) +
    scale_fill_manual(
      values = data$rank_colours,
      breaks = data$rank_labels,
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
}

plot_figure_2_panels <- function(data) {
  list(
    a = plot_figure_2_panel_a(data),
    b = plot_figure_2_panel_b(data),
    c = plot_figure_2_panel_c(data),
    rank_distribution = plot_figure_2_rank_distribution(data)
  )
}
