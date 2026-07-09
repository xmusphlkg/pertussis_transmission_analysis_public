## Figure 1 panels -------------------------------------------------------------

figure_1_age_group_colours <- function() {
  c(
    "Infant" = palette_discrete_primary_9[[7]],
    "Children" = palette_discrete_primary_9[[4]],
    "Adolescent" = palette_discrete_primary_9[[8]]
  )
}

figure_1_age_group_display_labels <- function() {
  c(
    "Infant" = "Infants <1 y",
    "Children" = "Children 1-9 y",
    "Adolescent" = "Adolescents 10-17 y"
  )
}

figure_1_endpoint_colours <- function() {
  c(
    "Reports" = palette_discrete_secondary_9[[7]],
    "Symptomatic" = palette_discrete_secondary_9[[6]],
    "Infections" = palette_discrete_secondary_9[[3]]
  )
}

figure_1_endpoint_shapes <- function() {
  c(
    "Reports" = 16,
    "Symptomatic" = 21,
    "Infections" = 17
  )
}

plot_figure_1_panel_a <- function(data) {
  ggplot(
    data$regional_focus %>% filter(positive_rate(reported_incidence_per_100k)),
    aes(year, reported_incidence_per_100k, colour = region)
  ) +
    geom_line(linewidth = 0.58, alpha = 0.92) +
    geom_point(data = data$regional_focus_latest, size = 1.15) +
    ggrepel::geom_text_repel(
      data = data$regional_focus_latest,
      aes(label = region_label),
      size = lancet_direct_label_size,
      fontface = "bold",
      nudge_x = 0.55,
      min.segment.length = 0,
      segment.size = 0.15,
      direction = "y",
      hjust = 0,
      show.legend = FALSE
    ) +
    scale_colour_manual(values = region_colours, guide = "none") +
    scale_x_continuous(breaks = seq(2000, 2024, by = 4), expand = expansion(mult = c(0.01, 0.14))) +
    scale_y_log10(breaks = c(0.3, 1, 3, 10, 30), labels = label_lancet_number(accuracy = 0.1)) +
    coord_cartesian(xlim = c(2000, 2025), ylim = c(0.25, 50)) +
    labs(x = "Year", y = "Reported incidence per 100 000/year (log)", tag = "a") +
    theme_lancet(base_size = journal_base_text_size)
}

plot_figure_1_panel_b <- function(data) {
  baseline_endpoint_colours <- figure_1_endpoint_colours()
  baseline_endpoint_shapes <- figure_1_endpoint_shapes()
  p1d_x_breaks <- pretty(c(0, max(data$baseline_endpoint_long$rate_per_100k, na.rm = TRUE)), n = 5)
  p1d_x_breaks <- p1d_x_breaks[p1d_x_breaks >= 0]
  p1d_x_limits <- range(p1d_x_breaks)

  ggplot() +
    geom_segment(
      data = data$endpoint_spans,
      aes(x = min_rate_per_100k, xend = max_rate_per_100k, y = country_y_base, yend = country_y_base),
      linewidth = 0.42,
      colour = manuscript_colour("pale_grey"),
      lineend = "round"
    ) +
    geom_point(
      data = data$baseline_endpoint_long,
      aes(rate_per_100k, country_y, colour = outcome, shape = outcome, fill = outcome),
      size = 1.95,
      alpha = 0.88,
      stroke = 0.36
    ) +
    scale_x_continuous(
      breaks = p1d_x_breaks,
      labels = label_lancet_comma(accuracy = 1),
      expand = expansion(mult = c(0.01, 0.04))
    ) +
    scale_y_continuous(
      breaks = seq_along(rev(data$country_order)),
      labels = rev(data$country_order),
      expand = expansion(add = c(0.35, 0.35))
    ) +
    scale_colour_manual(values = baseline_endpoint_colours, guide = "none") +
    scale_fill_manual(values = baseline_endpoint_colours, guide = "none") +
    scale_shape_manual(
      values = baseline_endpoint_shapes,
      name = "Outcome",
      guide = guide_legend(
        ncol = 1,
        byrow = TRUE,
        title.position = "top",
        title.hjust = 0,
        keywidth = unit(0.34, "cm"),
        keyheight = unit(0.46, "cm"),
        override.aes = list(
          colour = unname(baseline_endpoint_colours[names(baseline_endpoint_shapes)]),
          fill = unname(baseline_endpoint_colours[names(baseline_endpoint_shapes)]),
          alpha = 1,
          size = 2.1
        )
      )
    ) +
    coord_cartesian(xlim = p1d_x_limits, clip = "off") +
    labs(
      x = "Annualised rate per 100 000 people aged <18 years,\n2025–50",
      y = NULL,
      tag = "b"
    ) +
    theme_lancet(base_size = journal_compact_text_size) +
    theme_lancet_inside_legend(
      text_size = journal_compact_text_size,
      title_size = journal_compact_text_size,
      key_width = unit(0.34, "cm"),
      key_height = unit(0.46, "cm"),
      position = c(1, 0.01),
      justification = c(1, 0)
    ) +
    theme(
      legend.title = element_text(lineheight = 0.90),
      legend.text = element_text(lineheight = 1.16, margin = margin(t = 3, b = 3)),
      legend.spacing.y = unit(8, "pt"),
      legend.box.margin = margin(0, 0, 1, 0),
      legend.margin = margin(3, 0, 3, 0),
      axis.ticks.y = element_blank()
    )
}

plot_figure_1_panel_c <- function(data) {
  ggplot(data$decision_map, aes(primary_cases_per_100k, infant_hospitalizations_per_100k)) +
    geom_vline(
      xintercept = data$decision_medians$median_primary_cases_per_100k[[1]],
      linewidth = 0.28,
      linetype = "dashed",
      colour = manuscript_colour("mid_grey")
    ) +
    geom_hline(
      yintercept = data$decision_medians$median_infant_hospitalizations_per_100k[[1]],
      linewidth = 0.28,
      linetype = "dashed",
      colour = manuscript_colour("mid_grey")
    ) +
    geom_point(aes(fill = who_region), shape = 21, size = 2.35, colour = manuscript_colour("black"), stroke = 0.25, alpha = 0.9) +
    ggrepel::geom_text_repel(
      aes(label = country_code),
      size = journal_point_label_text_size,
      fontface = "bold",
      min.segment.length = 0,
      segment.size = 0.14,
      box.padding = 0.18,
      point.padding = 0.12,
      max.overlaps = Inf,
      show.legend = FALSE
    ) +
    scale_fill_manual(values = c(region_colours, "Other" = manuscript_colour("mid_grey")), guide = "none") +
    scale_x_continuous(
      breaks = pretty_breaks(n = 4),
      labels = label_lancet_comma(accuracy = 1),
      expand = expansion(mult = c(0.08, 0.12))
    ) +
    scale_y_continuous(
      breaks = pretty_breaks(n = 4),
      labels = label_lancet_comma(accuracy = 1),
      expand = expansion(mult = c(0.08, 0.12))
    ) +
    labs(
      x = "Symptomatic cases per 100 000 people aged <18 years,\n2025–50",
      y = "Infant hospitalisations per 100 000 infants,\n2025–50",
      tag = "c"
    ) +
    theme_lancet_panel(
      base_size = journal_compact_text_size,
      plot_margin = margin(5, 5, 4, 5),
      show_x_grid = TRUE,
      show_y_grid = TRUE
    )
}

plot_figure_1_panel_d <- function(data) {
  age_group_colours <- figure_1_age_group_colours()
  age_group_display_labels <- figure_1_age_group_display_labels()
  p1c_x_breaks <- pretty(c(0, max(tapply(
    data$baseline_composition$symptomatic_cases_per_100k_under18,
    data$baseline_composition$country,
    sum,
    na.rm = TRUE
  ))), n = 5)
  p1c_x_breaks <- p1c_x_breaks[p1c_x_breaks >= 0]

  ggplot(data$baseline_composition, aes(symptomatic_cases_per_100k_under18, country_label, fill = age_group)) +
    geom_col(width = 0.64, colour = "white", linewidth = 0.18, position = position_stack(reverse = TRUE)) +
    scale_x_continuous(
      labels = label_lancet_comma(accuracy = 1),
      breaks = p1c_x_breaks,
      expand = expansion(mult = c(0, 0.03))
    ) +
    scale_fill_manual(
      values = age_group_colours,
      labels = age_group_display_labels,
      guide = guide_legend(
        ncol = 1,
        byrow = TRUE,
        title.position = "top",
        title.hjust = 0,
        keywidth = unit(0.36, "cm"),
        keyheight = unit(0.46, "cm")
      )
    ) +
    coord_cartesian(xlim = range(p1c_x_breaks), clip = "off") +
    labs(
      x = "Symptomatic cases per 100 000 people aged <18 years,\n2025–50",
      y = NULL,
      fill = "Age group",
      tag = "d"
    ) +
    theme_lancet(base_size = journal_compact_text_size) +
    theme_lancet_inside_legend(
      text_size = journal_compact_text_size,
      title_size = journal_compact_text_size,
      key_width = unit(0.36, "cm"),
      key_height = unit(0.46, "cm"),
      position = c(1, 0.01),
      justification = c(1, 0)
    ) +
    theme(
      legend.title = element_text(lineheight = 0.90),
      legend.text = element_text(lineheight = 1.16, margin = margin(t = 3, b = 3)),
      legend.spacing.y = unit(8, "pt"),
      legend.box.margin = margin(0, 0, 1, 0),
      legend.margin = margin(3, 0, 3, 0),
      axis.ticks.y = element_blank()
    )
}

plot_figure_1_panels <- function(data) {
  list(
    a = plot_figure_1_panel_a(data),
    b = plot_figure_1_panel_b(data),
    c = plot_figure_1_panel_c(data),
    d = plot_figure_1_panel_d(data)
  )
}
