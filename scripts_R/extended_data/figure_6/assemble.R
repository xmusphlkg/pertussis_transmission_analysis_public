## Extended Data Figure 6 assembly --------------------------------------------

assemble_extended_data_figure_6 <- function(panels) {
  free(panels$b) + free(panels$c) + free(panels$d) +
    plot_layout(
      design = "AB\nCC",
      guides = "keep",
      widths = c(1.12, 0.88),
      heights = c(0.92, 1.08)
    ) +
    plot_annotation(tag_levels = "A") &
    (theme(plot.margin = margin(5, 3, 3, 5)) + theme_lancet_tags())
}
