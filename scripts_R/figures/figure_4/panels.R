## Figure 4 panels -------------------------------------------------------------

figure_4_scale_colour_vaccine_profiles <- function(data, guide = "none") {
  scale_colour_vaccine_profile(data$vaccine_outcome_colours, data$vaccine_legend_levels, guide = guide)
}

plot_figure_4_panel_a <- function(data) {
  ggplot() +
    geom_vline(xintercept = 0, linewidth = 0.24, colour = manuscript_colour("pale_grey")) +
    geom_point(
      data = data$mechanism_profile_points,
      aes(reduction_vs_baseline, mechanism_y_metric, colour = metric),
      shape = 16,
      size = 1.18,
      alpha = 0.52,
      position = position_jitter(width = 0, height = 0.075, seed = 4)
    ) +
    geom_point(
      data = data$mechanism_delta_summary,
      aes(median_reduction_vs_baseline, mechanism_y_metric, colour = metric, fill = metric),
      shape = 23,
      size = 2.35,
      stroke = 0.38
    ) +
    geom_point(
      data = data$mechanism_point_legend,
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
      breaks = data$mechanism_y_breaks$mechanism_y,
      labels = data$mechanism_y_breaks$mechanism_axis_label,
      expand = expansion(add = c(0.28, 0.28))
    ) +
    scale_colour_manual(
      values = data$mechanism_outcome_colours[data$mechanism_outcome_breaks],
      breaks = data$mechanism_outcome_breaks,
      labels = unname(data$mechanism_outcome_labels[data$mechanism_outcome_breaks]),
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
          colour = unname(data$mechanism_outcome_colours[data$mechanism_outcome_breaks]),
          fill = NA,
          size = 2.1,
          alpha = 1
        )
      )
    ) +
    scale_fill_manual(
      values = data$mechanism_outcome_colours,
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
      panel.grid.major.x = element_line(linewidth = 0.16, colour = lancet_grid_light_colour)
    ) +
    theme_lancet_inside_legend(
      key_height = unit(0.42, "cm"),
      box = "vertical",
      box_just = "left",
      title_lineheight = 0.90,
      title_hjust = 0,
      text_lineheight = 1.12,
      text_hjust = 0,
      text_margin = margin(t = 2, b = 2),
      spacing_x = unit(4, "pt"),
      background_fill = "#FFFFFFE8",
      legend_margin = margin(2, 3, 2, 3)
    )
}

plot_figure_4_panel_b <- function(data) {
  pct <- label_lancet_percent(accuracy = 1)

  ggplot(data$resistance_management, aes(primary_case_reduction, resistant_reduction_plot)) +
    geom_hline(yintercept = 0, linewidth = 0.22, colour = "grey72") +
    geom_vline(xintercept = 0, linewidth = 0.22, colour = "grey72") +
    geom_abline(slope = 1, intercept = 0, linewidth = 0.22, linetype = "22", colour = manuscript_colour("mid_grey")) +
    geom_point(
      data = filter(data$resistance_management, resistance_reduction_estimable),
      aes(colour = current_resistant_infections_plot),
      shape = 16,
      size = 2.45,
      alpha = 0.78
    ) +
    geom_point(
      data = filter(data$resistance_management, !resistance_reduction_estimable),
      aes(shape = "Not estimable at output threshold"),
      size = 2.35,
      colour = manuscript_colour("mid_grey"),
      fill = "white",
      stroke = 0.35,
      alpha = 0.82
    ) +
    ggrepel::geom_text_repel(
      data = data$p4b_label_data,
      aes(label = country_code),
      size = lancet_pt_to_geom_size(6.6),
      fontface = "bold",
      seed = 4,
      box.padding = 0.22,
      point.padding = 0.10,
      force = 3.0,
      force_pull = 0.35,
      max.iter = 10000,
      max.time = 1,
      nudge_x = data$p4b_label_data$label_nudge_x,
      nudge_y = data$p4b_label_data$label_nudge_y,
      xlim = c(-0.015, 0.84),
      ylim = c(0.02, 0.82),
      segment.size = 0.12,
      min.segment.length = 0,
      max.overlaps = Inf,
      show.legend = FALSE
    ) +
    scale_x_continuous(labels = pct, breaks = seq(0, 0.8, by = 0.2)) +
    scale_y_continuous(labels = pct, breaks = seq(0, 0.8, by = 0.2)) +
    scale_colour_gradientn(
      colours = c(manuscript_colour("light_grey"), manuscript_colour("pale_orange"), data$colours$resistance),
      trans = "log10",
      breaks = data$pal_breaks,
      limits = data$pal_limits,
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
      values = c("Not estimable at output threshold" = 21),
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
      panel.grid.major.x = element_line(linewidth = 0.16, colour = lancet_grid_light_colour)
    ) +
    theme_lancet_inside_legend(
      direction = "horizontal",
      key_width = unit(0.34, "cm"),
      title_size = journal_compact_text_size - 0.2,
      title_lineheight = 0.9,
      text_size = journal_compact_text_size - 0.5,
      background_fill = "#FFFFFFE8",
      legend_margin = margin(2, 3, 2, 3)
    )
}

plot_figure_4_panel_c <- function(data) {
  pct <- label_lancet_percent(accuracy = 1)

  ggplot(data$preference_summary, aes(resistance_weight_lambda, countries_preferred, colour = strategy_label)) +
    geom_step(linewidth = 0.72, direction = "hv", alpha = 0.88) +
    geom_point(
      data = filter(data$preference_summary, resistance_weight_lambda %in% seq(0, 1, by = 0.10)),
      size = 1.35,
      alpha = 0.90
    ) +
    geom_segment(
      data = data$p4c_threshold_tick_data,
      aes(
        x = minimum_lambda_for_resistance_guided_preferred,
        xend = minimum_lambda_for_resistance_guided_preferred,
        y = threshold_tick_ymin,
        yend = -0.18
      ),
      inherit.aes = FALSE,
      linewidth = 0.24,
      colour = manuscript_colour("mid_grey")
    ) +
    geom_text(
      data = data$p4c_threshold_label_data,
      aes(
        x = minimum_lambda_for_resistance_guided_preferred,
        y = threshold_label_y,
        label = threshold_label
      ),
      inherit.aes = FALSE,
      size = lancet_pt_to_geom_size(5.8),
      fontface = "bold",
      angle = 90,
      hjust = 0.5,
      vjust = 0.5
    ) +
    geom_segment(
      data = data$p4c_curve_label_data,
      aes(
        x = resistance_weight_lambda,
        xend = label_x,
        y = countries_preferred,
        yend = label_y,
        colour = strategy_label
      ),
      inherit.aes = FALSE,
      linewidth = 0.18,
      alpha = 0.70,
      show.legend = FALSE
    ) +
    geom_label(
      data = data$p4c_curve_label_data,
      aes(x = label_x, y = label_y, label = curve_label, colour = strategy_label),
      inherit.aes = FALSE,
      fill = "#FFFFFFE8",
      linewidth = 0,
      label.padding = unit(0.07, "lines"),
      label.r = unit(0.05, "lines"),
      size = lancet_pt_to_geom_size(5.8),
      fontface = "bold",
      lineheight = 0.88,
      show.legend = FALSE
    ) +
    scale_x_continuous(
      labels = pct,
      breaks = seq(0, 1, by = 0.20),
      expand = expansion(mult = c(0.02, 0.04))
    ) +
    scale_y_continuous(
      breaks = seq(0, 10, by = 2),
      limits = c(-1.08, 10.3),
      expand = expansion(mult = c(0.01, 0.03))
    ) +
    scale_colour_manual(
      values = data$preference_colours,
      guide = "none"
    ) +
    coord_cartesian(xlim = c(0, 1), ylim = c(-1.08, 10.3), clip = "off") +
    labs(
      x = "Weight on resistant-infection reduction",
      y = "Profiles with lowest weighted metric\n(out of 10)",
      tag = "c"
    ) +
    theme_lancet_panel(base_size = journal_compact_text_size, plot_margin = margin(4, 5, 4, 4), show_x_grid = TRUE, show_y_grid = TRUE) +
    theme(
      panel.grid.major.x = element_line(linewidth = 0.16, colour = lancet_grid_light_colour),
      axis.text.x = element_text(size = journal_compact_text_size, angle = 0, hjust = 0.5, vjust = 0.5)
    ) +
    theme_lancet_inside_legend(
      position = c(1, 0.5),
      justification = c(1, 0.5),
      key_height = unit(0.42, "cm"),
      box = "vertical",
      box_just = "left",
      title_lineheight = 0.90,
      title_hjust = 0,
      text_lineheight = 1.12,
      text_hjust = 0,
      text_margin = margin(t = 2, b = 2),
      spacing_x = unit(4, "pt"),
      background_fill = "#FFFFFFE8",
      legend_margin = margin(2, 3, 2, 3)
    )
}

plot_figure_4_panel_d <- function(data) {
  pct <- label_lancet_percent(accuracy = 1)

  ggplot(data$implementation_plot, aes(uptake_plot, median_reduction)) +
    geom_hline(yintercept = 0, linewidth = 0.22, colour = "grey72") +
    geom_line(
      aes(colour = implementation_group, group = implementation_group),
      linewidth = 0.42,
      alpha = 0.74
    ) +
    geom_linerange(
      aes(ymin = q25_reduction_plot, ymax = q75_reduction_plot, colour = implementation_group),
      linewidth = 0.34,
      alpha = 0.60,
      show.legend = FALSE
    ) +
    geom_point(
      aes(colour = implementation_group),
      size = 1.80,
      shape = 16,
      alpha = 0.92,
      stroke = 0.52
    ) +
    facet_wrap(
      vars(metric),
      nrow = 1,
      labeller = as_labeller(c(
        "Cases aged <18 years" = "Cases aged <18 years",
        "Resistant infections" = "Resistant infections"
      ))
    ) +
    scale_x_continuous(
      labels = pct,
      breaks = c(0, 0.5, 1),
      expand = expansion(mult = c(0.02, 0.03))
    ) +
    scale_y_continuous(labels = pct, breaks = c(-1.0, -0.5, 0, 0.5, 0.8)) +
    scale_colour_manual(
      values = c(
        "Guided only" = data$colours$implementation_guided,
        "+ restored PEP" = data$colours$implementation_pep_restored,
        "Lower reach" = data$colours$implementation_lower_reach
      ),
      name = "Implementation group",
      guide = guide_legend(
        title.position = "left",
        title.hjust = 0,
        nrow = 1,
        byrow = TRUE,
        override.aes = list(shape = 16, alpha = 1, size = 2.1)
      )
    ) +
    coord_cartesian(xlim = c(-0.07, 1.17), ylim = c(data$reduction_axis_floor, data$reduction_axis_ceiling), clip = "on") +
    labs(
      x = "Testing/treatment uptake",
      y = "Change vs guided-only pathway",
      tag = "d"
    ) +
    theme_lancet_panel(base_size = journal_compact_text_size, plot_margin = margin(3, 5, 4, 4), show_y_grid = TRUE) +
    theme(
      panel.grid.major.x = element_line(linewidth = 0.16, colour = lancet_grid_light_colour),
      legend.position = "top",
      legend.direction = "horizontal",
      legend.justification = "center",
      legend.box.just = "center",
      legend.title = element_text(face = "bold", size = journal_compact_text_size - 0.2, lineheight = 0.90),
      legend.text = element_text(size = journal_compact_text_size - 0.2, lineheight = 1.0),
      legend.key.width = unit(0.32, "cm"),
      legend.key.height = unit(0.24, "cm"),
      legend.margin = margin(0, 0, 1, 0),
      legend.spacing.x = unit(2, "pt"),
      axis.text.x = element_text(size = journal_compact_text_size - 0.5),
      strip.background = element_blank(),
      strip.text = element_text(face = "bold", size = journal_compact_text_size, lineheight = 0.92)
    )
}

plot_figure_4_future_residual <- function(data) {
  ggplot() +
    geom_point(
      data = data$vaccine_residual,
      aes(primary_cases_per_100k, scenario_label, colour = scenario_label, shape = "Profiles"),
      size = 1.18,
      alpha = 0.52,
      position = position_jitter(width = 0, height = 0.075, seed = 4)
    ) +
    geom_point(
      data = data$vaccine_burden_summary,
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
    scale_y_discrete(labels = data$vaccine_profile_axis_labels) +
    figure_4_scale_colour_vaccine_profiles(data) +
    scale_shape_profile_median() +
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
      panel.grid.major.x = element_line(linewidth = 0.16, colour = lancet_grid_light_colour)
    ) +
    theme_lancet_inside_legend(
      position = c(0.01, 1),
      justification = c(0, 1),
      key_height = unit(0.46, "cm"),
      title_lineheight = 0.90,
      text_lineheight = 1.16,
      text_margin = margin(t = 3, b = 3),
      spacing_y = unit(8, "pt"),
      box_margin = margin(1, 0, 0, 0),
      legend_margin = margin(3, 0, 3, 0)
    )
}

plot_figure_4_future_veinf <- function(data) {
  ggplot(
    data$veinf_attainment_plot_summary,
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
      values = data$veinf_seed_colours,
      breaks = names(data$veinf_seed_colours),
      labels = names(data$veinf_seed_colours),
      name = "Resistant seeding level",
      guide = guide_legend(nrow = 1, byrow = TRUE, override.aes = list(linewidth = 0.8, size = 2.0))
    ) +
    scale_linetype_manual(
      values = data$veinf_seed_linetypes,
      breaks = names(data$veinf_seed_linetypes),
      labels = names(data$veinf_seed_linetypes),
      name = "Resistant seeding level"
    ) +
    scale_shape_manual(
      values = data$veinf_seed_shapes,
      breaks = names(data$veinf_seed_shapes),
      labels = names(data$veinf_seed_shapes),
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
}

plot_figure_4_panels <- function(data) {
  list(
    a = plot_figure_4_panel_a(data),
    b = plot_figure_4_panel_b(data),
    c = plot_figure_4_panel_c(data),
    d = plot_figure_4_panel_d(data),
    future_residual = plot_figure_4_future_residual(data),
    future_veinf = plot_figure_4_future_veinf(data)
  )
}
