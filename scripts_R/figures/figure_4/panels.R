## Figure 4 panels -------------------------------------------------------------

# Figure contract --------------------------------------------------------------
# Core conclusion: resistance-guided management separates broad paediatric
# morbidity from resistant-infection control, while future vaccine mechanisms
# lower residual burden across most endpoints.
# Archetype: asymmetric quantitative grid, matching Figures 2 and 3.
# Panel a: paired country-level contrast for the two decision-defining outcomes.
# Panel b: cross-profile median and IQR across all seven endpoints.
# Panel c: full vaccine-mechanism surface, with median and IQR but no profile count.

figure_4_endpoint_display <- function(data) {
  unname(data$endpoint_axis_labels[as.character(data$endpoint_levels)])
}

figure_4_management_key_outcomes <- function(data) {
  outcome_labels <- c(
    "All <18 cases" = "<18 symptomatic cases",
    "Resistant infections" = "Resistant infections"
  )

  data$management_pairwise %>%
    filter(as.character(endpoint) %in% names(outcome_labels)) %>%
    mutate(
      outcome = factor(
        unname(outcome_labels[as.character(endpoint)]),
        levels = unname(outcome_labels)
      )
    ) %>%
    select(country, country_label, outcome, residual_index) %>%
    pivot_wider(names_from = outcome, values_from = residual_index)
}

plot_figure_4_panel_a <- function(data) {
  key_outcomes <- figure_4_management_key_outcomes(data)
  key_outcomes_long <- key_outcomes %>%
    pivot_longer(
      cols = c("<18 symptomatic cases", "Resistant infections"),
      names_to = "outcome",
      values_to = "residual_index"
    ) %>%
    mutate(outcome = factor(outcome, levels = c("<18 symptomatic cases", "Resistant infections")))

  outcome_colours <- c(
    "<18 symptomatic cases" = manuscript_colour("mid_grey"),
    "Resistant infections" = manuscript_colour("blue")
  )
  outcome_shapes <- c("<18 symptomatic cases" = 21, "Resistant infections" = 23)

  ggplot() +
    geom_vline(xintercept = 1, linewidth = 0.30, linetype = "dashed", colour = manuscript_colour("mid_grey")) +
    geom_segment(
      data = key_outcomes,
      aes(
        x = `<18 symptomatic cases`, xend = `Resistant infections`,
        y = country_label, yend = country_label
      ),
      linewidth = 0.38,
      colour = manuscript_colour("light_grey")
    ) +
    geom_point(
      data = key_outcomes_long,
      aes(residual_index, country_label, colour = outcome, shape = outcome),
      fill = "white", size = 2.05, stroke = 0.32
    ) +
    scale_x_continuous(
      breaks = seq(0.4, 1.1, by = 0.1),
      labels = label_lancet_percent(accuracy = 1),
      expand = expansion(mult = c(0.03, 0.06))
    ) +
    scale_colour_manual(values = outcome_colours, name = "Outcome") +
    scale_shape_manual(values = outcome_shapes, name = "Outcome") +
    labs(
      x = paste0(
        "Burden under resistance-guided management\n",
        "relative to routine schedule timeliness (%)"
      ),
      y = NULL,
      tag = "a"
    ) +
    theme_lancet_panel(
      base_size = journal_dense_text_size,
      plot_margin = margin(4, 5, 4, 4),
      show_x_grid = TRUE,
      show_y_grid = TRUE
    ) +
    theme(
      panel.grid.major.x = element_line(linewidth = 0.16, colour = lancet_grid_light_colour),
      legend.position = "top",
      legend.direction = "horizontal",
      legend.title.position = "left",
      legend.justification = "right",
      legend.margin = margin(0, 0, 1, 0),
      legend.key.width = unit(0.34, "cm")
    )
}

plot_figure_4_panel_b <- function(data) {
  endpoint_labels <- figure_4_endpoint_display(data)
  endpoint_colours <- stats::setNames(
    palette_discrete_7,
    as.character(data$endpoint_levels)
  )
  management_points <- data$management_pairwise %>%
    mutate(
      endpoint_display = factor(
        unname(data$endpoint_axis_labels[as.character(endpoint)]),
        levels = rev(endpoint_labels)
      )
    )
  management_summary <- data$management_summary %>%
    mutate(
      endpoint_display = factor(
        unname(data$endpoint_axis_labels[as.character(endpoint)]),
        levels = rev(endpoint_labels)
      )
    )

  ggplot() +
    geom_vline(xintercept = 1, linewidth = 0.24, colour = manuscript_colour("pale_grey")) +
    geom_segment(
      data = management_summary,
      aes(
        x = q25_residual_index,
        xend = q75_residual_index,
        y = endpoint_display,
        yend = endpoint_display,
        linetype = "IQR"
      ),
      linewidth = 1.05,
      colour = manuscript_colour("grey"),
      alpha = 0.74
    ) +
    geom_point(
      data = management_points,
      aes(
        residual_index,
        endpoint_display,
        colour = endpoint,
        shape = "Profiles"
      ),
      alpha = 0.50,
      size = 1.15,
      position = position_jitter(height = 0.075, width = 0)
    ) +
    geom_point(
      data = management_summary,
      aes(median_residual_index, endpoint_display, shape = "Median"),
      fill = manuscript_colour("black"),
      colour = "white",
      stroke = 0.20,
      size = 2.15
    ) +
    scale_x_continuous(
      breaks = seq(0.4, 1.4, by = 0.2),
      labels = label_lancet_percent(accuracy = 1)
    ) +
    scale_colour_manual(values = endpoint_colours, guide = "none") +
    scale_shape_profile_median() +
    scale_linetype_iqr() +
    coord_cartesian(xlim = c(0.35, 1.40), clip = "off") +
    labs(
      x = paste0(
        "Burden under resistance-guided management\n",
        "relative to routine schedule timeliness (%)"
      ),
      y = NULL,
      tag = "b"
    ) +
    theme_lancet_panel(
      base_size = journal_dense_text_size,
      plot_margin = margin(4, 4, 4, 6),
      show_x_grid = TRUE,
      show_y_grid = TRUE
    ) +
    theme(
      panel.grid.major.x = element_line(linewidth = 0.16, colour = lancet_grid_light_colour)
    ) +
    theme_lancet_inside_legend(
      position = c(1, 0.01),
      justification = c(1, 0),
      key_height = unit(0.34, "cm"),
      title_lineheight = 0.90,
      text_lineheight = 1.16,
      text_margin = margin(t = 0, b = 0),
      spacing_y = unit(0, "pt"),
      box_margin = margin(0, 0, 1, 0),
      legend_margin = margin(3, 0, 3, 0)
    )
}

plot_figure_4_panel_c <- function(data) {
  scenario_axis_labels <- c(
    "No vaccine" = "No vaccine",
    "Current aP-like" = "Current\naP-like",
    "Infection blocking" = "Infection\nblocking",
    "Transmission blocking" = "Transmission\nblocking",
    "High-blocking target" = "High-blocking\ntarget"
  )
  endpoint_axis_labels <- data$endpoint_axis_labels
  endpoint_axis_labels[c(
    "Infant hospitalisations",
    "Adolescent cases",
    "All <18 cases",
    "Resistant infections"
  )] <- c(
    "Infant\nhospitalisations",
    "Adolescent\ncases",
    "<18 symptomatic\ncases",
    "Resistant\ninfections"
  )
  vaccine_summary <- data$vaccine_summary_matrix %>%
    mutate(
      log2_median_residual = log2(median_residual_index),
      interval_collapsed = dplyr::near(q25_residual_index, median_residual_index) &
        dplyr::near(q75_residual_index, median_residual_index),
      summary_label = if_else(
        as.character(scenario_label) == "Current aP-like",
        "Reference",
        if_else(
          interval_collapsed,
          lancet_percent(median_residual_index, accuracy = 0.1),
          paste0(
            lancet_percent(median_residual_index, accuracy = 0.1), "\n[",
            lancet_percent(q25_residual_index, accuracy = 0.1), "–",
            lancet_percent(q75_residual_index, accuracy = 0.1), "]"
          )
        )
      ),
      summary_text_colour = if_else(
        median_residual_index < 0.62 | median_residual_index > 2.05,
        "white",
        lancet_text_colour
      )
    )
  panel_c_colourbar_values <- c(0.30, 0.50, 0.70, 1.00, 1.50, 2.00, 3.00, 4.00)
  panel_c_colourbar_breaks <- log2(panel_c_colourbar_values)
  panel_c_colourbar_limits <- range(panel_c_colourbar_breaks)
  panel_c_colourbar_labels <- label_lancet_percent(accuracy = 1)(panel_c_colourbar_values)

  ggplot(
    vaccine_summary,
    aes(endpoint, scenario_label, fill = log2_median_residual)
  ) +
    geom_tile(colour = "white", linewidth = lancet_heatmap_tile_linewidth) +
    geom_vline(xintercept = c(3.5, 5.5, 6.5), colour = "white", linewidth = 1.0) +
    geom_hline(yintercept = 3.5, colour = "white", linewidth = 1.0) +
    geom_text(
      aes(label = summary_label, colour = summary_text_colour),
      size = journal_heatmap_cell_text_size,
      lineheight = 0.84
    ) +
    scale_x_discrete(labels = endpoint_axis_labels) +
    scale_y_discrete(labels = scenario_axis_labels) +
    scale_fill_gradient2(
      low = manuscript_colour("blue"),
      mid = manuscript_colour("light_grey"),
      high = manuscript_colour("vermillion"),
      midpoint = 0,
      limits = panel_c_colourbar_limits,
      breaks = panel_c_colourbar_breaks,
      labels = panel_c_colourbar_labels,
      oob = scales::squish,
      name = "Median burden relative to current aP-like (%)",
      guide = guide_lancet_colourbar(
        barwidth = unit(0.30, "cm"),
        barheight = unit(5.2, "cm"),
        title.position = "left",
        title.hjust = 0
      )
    ) +
    scale_colour_identity() +
    labs(x = NULL, y = NULL, tag = "c") +
    theme_lancet_panel(
      base_size = journal_dense_text_size,
      plot_margin = margin(4, 5, 4, 4),
      show_x_grid = TRUE,
      show_y_grid = TRUE
    ) +
    theme(
      panel.grid.major.x = element_line(linewidth = 0.16, colour = lancet_grid_light_colour),
      axis.text.x = element_text(
        face = "plain",
        size = journal_dense_text_size - 0.9
      ),
      axis.text.y = element_text(
        size = journal_dense_text_size - 0.2,
        lineheight = 0.90
      ),
      legend.title = element_text(angle = 90, hjust = 0, vjust = 0.5),
      legend.title.position = "left",
      legend.position = "right",
      legend.direction = "vertical"
    )
}

plot_figure_4_panels <- function(data) {
  list(
    a = plot_figure_4_panel_a(data),
    b = plot_figure_4_panel_b(data),
    c = plot_figure_4_panel_c(data)
  )
}
