## Extended Data Figure 11 assembly -------------------------------------------

assemble_extended_data_figure_11 <- function(panels) {
  wrap_plots(
    panels$a, panels$b, panels$c,
    panels$d, panels$e, free(panels$f),
    ncol = 3,
    guides = "keep"
  ) +
    plot_layout(widths = c(1.15, 1.08, 1), heights = c(1.03, 1)) +
    plot_annotation(tag_levels = "A") &
    (theme(plot.margin = margin(4, 4, 4, 4)) + theme_lancet_tags())
}
