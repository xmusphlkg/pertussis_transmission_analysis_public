## Extended Data Figure 5 assembly --------------------------------------------

assemble_extended_data_figure_5 <- function(panels) {
  free(panels$b) + free(panels$c) + free(panels$d) +
    plot_layout(design = "AA\nBC", guides = "keep", heights = c(1.22, 1), widths = c(1.05, 0.95)) +
    plot_annotation(tag_levels = "A") &
    (theme(plot.margin = margin(3, 3, 3, 3)) + theme_lancet_tags())
}
