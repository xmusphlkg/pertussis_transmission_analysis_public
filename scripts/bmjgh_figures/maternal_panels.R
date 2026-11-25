# Figure 3 and Supplementary Figure S3 share the same endpoint and setting order.
source("scripts/bmjgh_figures/theme.R")
maternal_source <- "outputs/core2026/07_manuscript/model_study_revision/results_revision/maternal_coverage_infant_outcomes.csv"
maternal_baselines <- c(China = 0, Japan = 0, Sweden = .5, Thailand = .5, Australia = .6,
                        England = .73, Brazil = .75, New_Zealand = .75)

draw_maternal_coverage <- function(number, route_name, caption, ncol = 2, height = 218, pretty_axes = FALSE) {
  all_rows <- read.csv(maternal_source)
  rows <- subset(all_rows, route == route_name, select = country:yll_q975)
  stopifnot(nrow(rows) == 46, all(rows$nasal_coverage == .9), all(rows$profile_id == "b050_a050_d03_point"),
            all(rows$maternal_baseline == maternal_baselines[rows$country]),
            all(rows$yll_q025 <= rows$yll_q975), !anyDuplicated(rows[c("country", "maternal_coverage", "horizon")]))
  by_site <- split(rows, factor(rows$country, levels = names(maternal_baselines)))
  panels <- lapply(by_site, function(d) {
    d$window <- factor(d$horizon, levels = c(20, 30))
    site <- gsub("_", " ", d$country[1])
    stopifnot(all(table(d$maternal_coverage, d$window) == 1))
    coverage <- 100 * d$maternal_coverage
    span <- diff(range(coverage))
    offset <- ifelse(d$horizon == 20, -.014, .014) * span
    d$display_coverage <- coverage + offset
    title <- sprintf("%s (baseline %g%%)", site, d$maternal_baseline[1] * 100)
    y_labels <- scales::label_number(big.mark = ",")
    if (pretty_axes) {
      # Keep both offset horizons and their error-bar caps inside 0–100% coverage.
      inset <- .04 * span
      d$display_coverage <- pmin(100 - inset, pmax(inset, coverage)) + offset
      half_cap <- .006 * span
      x_breaks <- pretty(range(d$display_coverage) + c(-half_cap, half_cap), n = 3)
      y_breaks <- pretty(c(0, d$infant_yll_averted, d$yll_q025, d$yll_q975), n = 4)
      axes <- list(
        scale_x_continuous(breaks = x_breaks, limits = range(x_breaks), expand = c(0, 0)),
        scale_y_continuous(breaks = y_breaks, limits = range(y_breaks), expand = c(0, 0),
                           labels = y_labels))
    } else {
      axes <- list(
        scale_x_continuous(breaks = sort(unique(coverage)), expand = expansion(mult = .075)),
        scale_y_continuous(labels = y_labels, n.breaks = 4, expand = expansion(mult = .075)))
    }
    if (ncol == 4) title <- sub(" (baseline", "\n(baseline", title, fixed = TRUE)
    panel <- ggplot(d, aes(display_coverage, infant_yll_averted, group = window,
                  colour = window, shape = window)) +
      geom_hline(yintercept = 0, linewidth = .32, colour = "#8C989F") +
      geom_line(linewidth = .42, alpha = .72) +
      geom_errorbar(aes(ymin = yll_q025, ymax = yll_q975), width = .012 * span, linewidth = .48) +
      geom_point(size = 2.05) +
      scale_colour_manual(NULL, values = horizon_colours, labels = c("20 years", "30 years")) +
      scale_shape_manual(NULL, values = horizon_shapes, labels = c("20 years", "30 years")) +
      axes +
      labs(title = title,
           y = "Discounted infant YLL averted", x = "Maternal vaccination coverage (%)") +
      theme(axis.text = element_text(size = 8), plot.title = element_text(size = 9),
            panel.grid.major.x = element_blank(),
            plot.margin = margin(5, 9, 5, 4), legend.text = element_text(size = 8))
    if (route_name == "standalone" && site == "Brazil") {
      near_zero <- subset(d, horizon == 30)
      labels <- sprintf("%g%%   %+.2f [%+.2f, %+.2f]", 100 * near_zero$maternal_coverage,
                        near_zero$infant_yll_averted, near_zero$yll_q025, near_zero$yll_q975)
      panel <- panel + annotate("text", x = mean(range(coverage)), y = .38 * max(y_breaks),
                                label = paste(c("30-year YLL [95% UI]", labels), collapse = "\n"),
                                size = 7.3, size.unit = "pt", lineheight = 1.05)
    }
    panel
  })
  names(panels) <- names(by_site) <- LETTERS[1:8]
  panels <- Map(function(panel, letter) {
    panel + labs(title = paste(letter, panel$labels$title, sep = ": "))
  }, panels, names(panels))
  composite <- wrap_plots(panels, ncol = ncol, guides = "collect", axis_titles = "collect_x") &
    labs(y = NULL) & theme(legend.position = "bottom")
  y_label <- wrap_elements(full = grid::textGrob(
    "Discounted infant YLL averted", rot = 90,
    gp = grid::gpar(fontsize = 8, fontfamily = "Arial", col = "#29353D")))
  composite <- (y_label | composite) + plot_layout(widths = c(.045, 1))
  input_files <- c(maternal_source, "scripts/bmjgh_figures/maternal_panels.R")
  save_figure(number, composite, panels, by_site, caption, height = height, width = 190)
}
