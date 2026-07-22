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

figure_1_decision_map_label_offsets <- function() {
  tibble::tibble(
    country_label_text = c(
      "China", "Thailand", "Japan", "South Africa", "Brazil",
      "United States", "Sweden", "New Zealand", "United Kingdom", "Australia"
    ),
    label_nudge_x = c(-18, -20, -14, 14, 16, 18, 18, -12, 12, 12),
    label_nudge_y = c(-14, 4, 12, -12, -6, 12, 22, -16, 18, 8)
  )
}

figure_1_region_label_positions <- function() {
  tibble::tibble(
    region_key = c(
      "European Region", "Western Pacific Region", "Global",
      "Region of the Americas", "African Region", "South-East Asia Region"
    ),
    label_x = c(2024.35, 2024.35, 2024.35, 2024.35, 2024.35, 2024.35),
    label_y = c(39.0, 23.5, 14.3, 5.6, 0.83, 0.41)
  )
}

plot_figure_1_panel_a <- function(data) {
  regional_labels <- data$regional_focus_latest %>%
    mutate(region_key = as.character(region)) %>%
    left_join(figure_1_region_label_positions(), by = "region_key")

  ggplot(
    data$regional_focus %>% filter(positive_rate(reported_incidence_per_100k)),
    aes(year, reported_incidence_per_100k, colour = region)
  ) +
    geom_line(linewidth = 0.58, alpha = 0.92) +
    geom_point(data = data$regional_focus_latest, size = 1.15) +
    geom_text(
      data = regional_labels,
      aes(x = label_x, y = label_y, label = region_label),
      size = lancet_direct_label_size,
      fontface = "bold",
      hjust = 0,
      lineheight = 0.9,
      show.legend = FALSE
    ) +
    scale_colour_manual(values = region_colours, guide = "none") +
    scale_x_continuous(breaks = seq(2000, 2024, by = 4), expand = expansion(mult = c(0.01, 0.01))) +
    scale_y_log10(
      breaks = c(0.3, 1, 3, 10, 30, 50),
      labels = label_lancet_number(accuracy = 0.1),
      expand = expansion(mult = c(0, 0))
    ) +
    coord_cartesian(xlim = c(2000, 2029.5), ylim = c(0.25, 50)) +
    labs(x = "Year", y = "Reported incidence per 100 000 (log)", tag = "a") +
    theme_lancet(base_size = journal_base_text_size)
}

plot_figure_1_panel_b <- function(data) {
  baseline_endpoint_colours <- figure_1_endpoint_colours()
  baseline_endpoint_shapes <- figure_1_endpoint_shapes()
  baseline_endpoint_order <- names(baseline_endpoint_shapes)
  endpoint_interval_cap <- 0.075
  panel_b_x_values <- c(
    0,
    data$baseline_endpoint_long$rate_per_100k,
    data$endpoint_intervals$interval_upper_per_100k
  )
  p1d_x_breaks <- pretty(panel_b_x_values[is.finite(panel_b_x_values)], n = 5)
  p1d_x_breaks <- p1d_x_breaks[p1d_x_breaks >= 0]
  p1d_x_breaks <- sort(unique(c(p1d_x_breaks, 7000)))
  p1d_x_limits <- range(p1d_x_breaks)

  ggplot() +
    geom_segment(
      data = data$endpoint_spans,
      aes(x = min_rate_per_100k, xend = max_rate_per_100k, y = country_y_base, yend = country_y_base),
      linewidth = 0.42,
      colour = manuscript_colour("pale_grey"),
      lineend = "round"
    ) +
    geom_segment(
      data = data$endpoint_intervals,
      aes(
        x = interval_lower_per_100k,
        xend = interval_upper_per_100k,
        y = country_y,
        yend = country_y,
        colour = outcome
      ),
      linewidth = 0.68,
      alpha = 0.92,
      lineend = "butt"
    ) +
    geom_segment(
      data = data$endpoint_intervals,
      aes(
        x = interval_lower_per_100k,
        xend = interval_lower_per_100k,
        y = country_y - endpoint_interval_cap,
        yend = country_y + endpoint_interval_cap,
        colour = outcome
      ),
      linewidth = 0.46,
      alpha = 0.92,
      lineend = "butt"
    ) +
    geom_segment(
      data = data$endpoint_intervals,
      aes(
        x = interval_upper_per_100k,
        xend = interval_upper_per_100k,
        y = country_y - endpoint_interval_cap,
        yend = country_y + endpoint_interval_cap,
        colour = outcome
      ),
      linewidth = 0.46,
      alpha = 0.92,
      lineend = "butt"
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
      limits = baseline_endpoint_order,
      breaks = baseline_endpoint_order,
      labels = c(
        "Reports" = "Reports",
        "Symptomatic" = "Symptomatic",
        "Infections" = "Infections"
      ),
      name = "Outcome",
      guide = guide_legend(
        ncol = 1,
        byrow = TRUE,
        title.position = "top",
        title.hjust = 0,
        keywidth = unit(0.34, "cm"),
        keyheight = unit(0.46, "cm"),
        override.aes = list(
          colour = unname(baseline_endpoint_colours[baseline_endpoint_order]),
          fill = unname(baseline_endpoint_colours[baseline_endpoint_order]),
          alpha = 1,
          size = 2.1
        )
      )
    ) +
    coord_cartesian(xlim = p1d_x_limits, clip = "off") +
    labs(
      x = "Annual <18 index per 100 000",
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
  decision_map <- data$decision_map %>%
    left_join(figure_1_decision_map_label_offsets(), by = "country_label_text") %>%
    arrange(primary_cases_per_100k, infant_hospitalizations_per_100k)
  panel_upper <- ceiling(
    max(
      c(decision_map$primary_cases_per_100k, decision_map$infant_hospitalizations_per_100k),
      na.rm = TRUE
    ) / 200
  ) * 200
  panel_limits <- c(0, panel_upper)
  panel_breaks <- seq(0, panel_upper, by = 200)

  ggplot(decision_map, aes(primary_cases_per_100k, infant_hospitalizations_per_100k)) +
    geom_abline(
      intercept = 0,
      slope = 1,
      linewidth = 0.28,
      linetype = "22",
      colour = manuscript_colour("mid_grey")
    ) +
    geom_point(aes(fill = who_region), shape = 21, size = 2.35, colour = manuscript_colour("black"), stroke = 0.25, alpha = 0.9) +
    ggrepel::geom_text_repel(
      aes(label = country_label_text),
      size = journal_heatmap_cell_text_size,
      fontface = "bold",
      seed = 7,
      nudge_x = decision_map$label_nudge_x,
      nudge_y = decision_map$label_nudge_y,
      box.padding = 0.22,
      point.padding = 0.10,
      force = 1.6,
      force_pull = 0.28,
      max.iter = 10000,
      max.time = 1.5,
      segment.size = 0.12,
      segment.alpha = 0.85,
      min.segment.length = 0,
      max.overlaps = Inf,
      direction = "both",
      show.legend = FALSE
    ) +
    scale_fill_manual(values = c(region_colours, "Other" = manuscript_colour("mid_grey")), guide = "none") +
    scale_x_continuous(
      limits = panel_limits,
      breaks = panel_breaks,
      labels = label_lancet_comma(accuracy = 1),
      expand = expansion(mult = c(0, 0))
    ) +
    scale_y_continuous(
      limits = panel_limits,
      breaks = panel_breaks,
      labels = label_lancet_comma(accuracy = 1),
      expand = expansion(mult = c(0, 0))
    ) +
    coord_equal(xlim = panel_limits, ylim = panel_limits, clip = "off") +
    labs(
      x = "<18 cases per 100 000",
      y = "Infant hospitalisations per 100 000",
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
  country_totals <- data$baseline_composition %>%
    group_by(country_label) %>%
    summarise(
      symptomatic_cases_per_100k_under18 = sum(symptomatic_cases_per_100k_under18),
      .groups = "drop"
    )
  p1d_x_breaks <- pretty(
    c(0, country_totals$symptomatic_cases_per_100k_under18),
    n = 5
  )
  p1d_x_breaks <- p1d_x_breaks[p1d_x_breaks >= 0]
  p1d_x_limits <- range(p1d_x_breaks)

  ggplot(data$baseline_composition, aes(symptomatic_cases_per_100k_under18, country_label, fill = age_group)) +
    geom_col(width = 0.64, colour = "white", linewidth = 0.18, position = position_stack(reverse = TRUE)) +
    scale_x_continuous(
      limits = p1d_x_limits,
      labels = label_lancet_comma(accuracy = 1),
      breaks = p1d_x_breaks,
      expand = expansion(mult = c(0, 0))
    ) +
    scale_y_discrete(limits = rev(data$country_order), drop = FALSE) +
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
    labs(
      x = "<18 symptomatic cases per 100 000",
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
