## Figure 2 panels -------------------------------------------------------------

figure_2_visual_required_columns <- function(data, columns, panel_name) {
  missing_columns <- setdiff(columns, names(data))
  if (length(missing_columns) > 0L) {
    stop(
      panel_name, " is missing visual-contract column(s): ",
      paste(missing_columns, collapse = ", "),
      call. = FALSE
    )
  }
  invisible(data)
}

figure_2_profile_limits <- function(data) {
  profile_order <- as.character(data$profile_order)
  if (length(profile_order) == 0L ||
      anyNA(profile_order) ||
      any(!nzchar(profile_order))) {
    stop("Figure 2 requires a complete top-to-bottom profile_order.", call. = FALSE)
  }
  rev(profile_order)
}

figure_2_prepare_profile_factor <- function(panel_data, data, panel_name) {
  if (!"profile_factor" %in% names(panel_data)) {
    if (!"country_label" %in% names(panel_data)) {
      stop(panel_name, " requires profile_factor or country_label.", call. = FALSE)
    }
    panel_data$profile_factor <- panel_data$country_label
  }
  panel_data$profile_factor <- factor(
    as.character(panel_data$profile_factor),
    levels = figure_2_profile_limits(data)
  )
  if (anyNA(panel_data$profile_factor)) {
    stop(panel_name, " contains profiles outside profile_order.", call. = FALSE)
  }
  panel_data
}

figure_2_visual_strategy_labels <- function(data) {
  strategy_order <- as.character(data$strategy_order)
  labels <- data$strategy_short_labels
  if (is.null(labels)) labels <- data$strategy_labels
  if (is.null(names(labels)) && length(labels) == length(strategy_order)) {
    names(labels) <- strategy_order
  }
  resolved <- unname(labels[strategy_order])
  resolved[is.na(resolved) | !nzchar(resolved)] <- strategy_order[
    is.na(resolved) | !nzchar(resolved)
  ]
  stats::setNames(resolved, strategy_order)
}

figure_2_strategy_palette <- function(data) {
  palette <- data$strategy_colours
  strategy_order <- as.character(data$strategy_order)
  if (is.null(names(palette)) && length(palette) == length(strategy_order)) {
    names(palette) <- strategy_order
  }
  palette <- palette[strategy_order]
  if (length(palette) != length(strategy_order) ||
      anyNA(palette) ||
      any(!nzchar(palette))) {
    stop("Figure 2 requires one named colour for every strategy.", call. = FALSE)
  }
  palette
}

plot_figure_2_panel_a <- function(data) {
  figure_2_visual_required_columns(
    data$panel_a,
    c(
      "country", "scenario", "scenario_label",
      "relative_reduction_percent", "profile_factor"
    ),
    "Figure 2 panel a"
  )
  panel_a <- figure_2_prepare_profile_factor(
    data$panel_a,
    data,
    "Figure 2 panel a"
  )
  scenario_order <- c("coverage_floor_only", "timeliness_only")
  panel_a$scenario <- factor(
    as.character(panel_a$scenario),
    levels = scenario_order
  )
  if (anyNA(panel_a$scenario) ||
      any(!is.finite(panel_a$relative_reduction_percent))) {
    stop("Figure 2 panel a contains an invalid delivery-lever result.", call. = FALSE)
  }
  connector_data <- panel_a %>%
    group_by(country, profile_factor) %>%
    summarise(
      xmin = min(relative_reduction_percent),
      xmax = max(relative_reduction_percent),
      .groups = "drop"
    )
  scenario_labels <- stats::setNames(
    c("Coverage floor only", "Timeliness only"),
    scenario_order
  )
  scenario_colours <- c(
    coverage_floor_only = manuscript_colour("grey"),
    timeliness_only = manuscript_colour("blue")
  )
  scenario_shapes <- c(
    coverage_floor_only = 22,
    timeliness_only = 21
  )

  ggplot(panel_a, aes(x = relative_reduction_percent, y = profile_factor)) +
    geom_vline(
      xintercept = 0,
      linewidth = 0.30,
      linetype = "dashed",
      colour = manuscript_colour("mid_grey")
    ) +
    geom_segment(
      data = connector_data,
      aes(x = xmin, xend = xmax, y = profile_factor, yend = profile_factor),
      inherit.aes = FALSE,
      linewidth = 0.48,
      colour = manuscript_colour("light_grey"),
      lineend = "round"
    ) +
    geom_point(
      aes(fill = scenario, shape = scenario),
      size = 2.55,
      stroke = 0.34,
      colour = lancet_text_colour
    ) +
    scale_fill_manual(
      values = scenario_colours,
      breaks = scenario_order,
      labels = scenario_labels,
      name = "Configured lever"
    ) +
    scale_shape_manual(
      values = scenario_shapes,
      breaks = scenario_order,
      labels = scenario_labels,
      name = "Configured lever"
    ) +
    scale_x_continuous(
      limits = c(-5, 30),
      breaks = c(-5, 0, 10, 20, 30),
      labels = label_lancet_number(accuracy = 1, suffix = "%"),
      expand = expansion(mult = 0)
    ) +
    scale_y_discrete(
      limits = figure_2_profile_limits(data),
      drop = FALSE,
      expand = expansion(add = 0.55)
    ) +
    labs(
      x = "Reduction in <18 symptomatic-case index",
      y = NULL,
      tag = "a"
    ) +
    theme_lancet_panel(
      base_size = journal_dense_text_size,
      plot_margin = margin(3, 5, 4, 7),
      show_x_grid = FALSE,
      show_y_grid = TRUE
    ) +
    theme(
      axis.text.y = element_text(
        colour = lancet_text_colour,
        size = journal_country_axis_text_size - 0.15,
        lineheight = 0.95
      ),
      axis.title.x = element_text(
        size = journal_dense_text_size - 0.35,
        lineheight = 0.94
      ),
      legend.position = "top",
      legend.direction = "horizontal",
      legend.justification = "left",
      legend.title = element_text(size = journal_dense_text_size - 0.5),
      legend.text = element_text(size = journal_dense_text_size - 0.65),
      legend.key.width = unit(0.30, "cm"),
      legend.spacing.x = unit(0.08, "cm"),
      legend.margin = margin(0, 0, 0, 0),
      legend.box.margin = margin(0, 0, 0, 0)
    ) +
    guides(
      fill = guide_legend(
        nrow = 1,
        byrow = TRUE,
        title.position = "left",
        title.hjust = 0,
        override.aes = list(shape = unname(scenario_shapes))
      ),
      shape = "none"
    )
}

plot_figure_2_panel_b <- function(data) {
  figure_2_visual_required_columns(
    data$panel_b,
    c(
      "country_label", "reference_strategy",
      "reference_choice_retained", "regret_pp_q95", "setting_count"
    ),
    "Figure 2 panel b"
  )
  panel_b <- data$panel_b
  if (any(!is.finite(panel_b$reference_choice_retained)) ||
      any(!is.finite(panel_b$regret_pp_q95)) ||
      any(panel_b$setting_count != data$rank_setting_count)) {
    stop("Figure 2 panel b contains an invalid fragility summary.", call. = FALSE)
  }
  observed_strategies <- as.character(data$strategy_order)[
    as.character(data$strategy_order) %in% panel_b$reference_strategy
  ]
  strategy_labels <- figure_2_visual_strategy_labels(data)
  strategy_palette <- figure_2_strategy_palette(data)
  panel_b_x_breaks <- pretty(panel_b$reference_choice_retained, n = 5)
  panel_b_y_breaks <- pretty(panel_b$regret_pp_q95, n = 5)

  ggplot(
    panel_b,
    aes(
      x = reference_choice_retained,
      y = regret_pp_q95,
      fill = reference_strategy
    )
  ) +
    geom_hline(
      yintercept = 0,
      linewidth = 0.30,
      colour = manuscript_colour("mid_grey")
    ) +
    geom_point(
      shape = 21,
      size = 2.75,
      stroke = 0.34,
      colour = lancet_text_colour
    ) +
    ggrepel::geom_text_repel(
      aes(label = country_label),
      seed = 20260718,
      size = journal_heatmap_cell_text_size,
      fontface = "bold",
      family = lancet_font_family,
      colour = lancet_text_colour,
      box.padding = 0.30,
      point.padding = 0.24,
      min.segment.length = 0,
      segment.size = lancet_direct_label_segment_linewidth,
      segment.color = manuscript_colour("mid_grey"),
      max.overlaps = Inf,
      max.time = 2,
      direction = "both",
      show.legend = FALSE
    ) +
    scale_fill_manual(
      values = strategy_palette,
      breaks = observed_strategies,
      labels = unname(strategy_labels[observed_strategies]),
      name = "Reference choice",
      drop = FALSE
    ) +
    scale_x_continuous(
      limits = range(panel_b_x_breaks),
      breaks = panel_b_x_breaks,
      labels = label_lancet_number(accuracy = 1),
      expand = expansion(mult = 0)
    ) +
    scale_y_continuous(
      limits = range(panel_b_y_breaks),
      breaks = panel_b_y_breaks,
      labels = label_lancet_number(accuracy = 1),
      expand = expansion(mult = c(0.14, 0.04))
    ) +
    coord_cartesian(clip = "off") +
    labs(
      x = "Settings retaining the reference choice (out of 128)",
      y = paste0(
        "95th-percentile regret\n",
        "(percentage points of current-practice burden)"
      ),
      tag = "b"
    ) +
    theme_lancet_panel(
      base_size = journal_dense_text_size,
      plot_margin = margin(3, 9, 4, 6),
      show_x_grid = FALSE,
      show_y_grid = TRUE
    ) +
    theme(
      axis.title.x = element_text(
        size = journal_dense_text_size - 0.55,
        lineheight = 0.93
      ),
      axis.title.y = element_text(
        size = journal_dense_text_size - 0.55,
        lineheight = 0.93
      ),
      legend.position = "top",
      legend.direction = "horizontal",
      legend.justification = "left",
      legend.title = element_text(size = journal_dense_text_size - 0.5),
      legend.text = element_text(size = journal_dense_text_size - 0.65),
      legend.key.width = unit(0.30, "cm"),
      legend.spacing.x = unit(0.08, "cm"),
      legend.margin = margin(0, 0, 0, 0),
      legend.box.margin = margin(0, 0, 0, 0)
    ) +
    guides(
      fill = guide_legend(
        nrow = 1,
        byrow = TRUE,
        title.position = "left",
        title.hjust = 0,
        override.aes = list(shape = 21, size = 2.3)
      )
    )
}

plot_figure_2_panel_c <- function(data) {
  figure_2_visual_required_columns(
    data$panel_c,
    c(
      "country", "strategy", "effect_label", "effect_text_colour",
      "deterministic_relative_reduction", "preferred_in_program_only"
    ),
    "Figure 2 panel c"
  )
  panel_c <- figure_2_prepare_profile_factor(
    data$panel_c,
    data,
    "Figure 2 panel c"
  )
  panel_c$strategy <- factor(
    as.character(panel_c$strategy),
    levels = as.character(data$strategy_order)
  )
  if (anyNA(panel_c$strategy) ||
      any(!is.finite(panel_c$deterministic_relative_reduction)) ||
      anyNA(panel_c$preferred_in_program_only)) {
    stop("Figure 2 panel c contains an invalid effect-matrix value.", call. = FALSE)
  }
  matrix_labels <- figure_2_visual_strategy_labels(data)
  matrix_labels <- stats::setNames(
    stringr::str_wrap(unname(matrix_labels), width = 15),
    names(matrix_labels)
  )
  panel_c_colourbar_breaks <- pretty(
    range(panel_c$deterministic_relative_reduction),
    n = 5
  )
  panel_c_colourbar_limits <- range(panel_c_colourbar_breaks)

  ggplot(
    panel_c,
    aes(
      x = strategy,
      y = profile_factor,
      fill = deterministic_relative_reduction
    )
  ) +
    geom_tile(
      width = 0.96,
      height = 0.92,
      colour = "white",
      linewidth = lancet_heatmap_tile_linewidth
    ) +
    geom_text(
      aes(label = effect_label, colour = effect_text_colour),
      size = journal_heatmap_cell_text_size,
      lineheight = 0.88,
      family = lancet_font_family,
      show.legend = FALSE
    ) +
    scale_fill_reduction(
      midpoint = 0,
      limits = panel_c_colourbar_limits,
      breaks = panel_c_colourbar_breaks,
      labels = label_lancet_percent(accuracy = 1),
      oob = scales::squish,
      name = "Under-18 case reduction: estimate [95% CI]",
      guide = guide_lancet_colourbar(
        barwidth = unit(0.30, "cm"),
        barheight = unit(6.0, "cm"),
        title.position = "left",
        title.hjust = 0
      )
    ) +
    scale_colour_identity() +
    scale_x_discrete(
      limits = as.character(data$strategy_order),
      labels = matrix_labels,
      drop = FALSE,
      expand = expansion(add = 0)
    ) +
    scale_y_discrete(
      limits = figure_2_profile_limits(data),
      drop = FALSE,
      expand = expansion(add = 0)
    ) +
    labs(
      x = "Programme strategy",
      y = NULL,
      tag = "c"
    ) +
    theme_lancet_panel(
      base_size = journal_dense_text_size,
      plot_margin = margin(3, 5, 4, 7),
      show_x_grid = FALSE,
      show_y_grid = FALSE
    ) +
    theme(
      axis.text.x = element_text(
        size = journal_dense_text_size - 0.35,
        lineheight = 0.88,
        margin = margin(t = 2)
      ),
      axis.text.y = element_text(
        colour = lancet_text_colour,
        size = journal_country_axis_text_size,
        lineheight = 0.95
      ),
      axis.title.x = element_text(
        size = journal_dense_text_size - 0.2,
        margin = margin(t = 4)
      ),
      legend.position = "right",
      legend.direction = "vertical",
      legend.title = element_text(angle = 90, hjust = 0, vjust = 0.5),
      legend.title.position = "left"
    )
}

plot_figure_2_panels <- function(data) {
  list(
    a = plot_figure_2_panel_a(data),
    b = plot_figure_2_panel_b(data),
    c = plot_figure_2_panel_c(data)
  )
}
