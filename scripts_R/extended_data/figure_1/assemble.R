## Extended Data Figure 1 assembly --------------------------------------------

assemble_extended_data_figure_1 <- function(panels) {
  free(panels$a) + free(panels$b) + panels$c + free(panels$d) +
    plot_layout(ncol = 2, guides = "keep", widths = c(0.9, 1.1), heights = c(0.92, 1.08)) +
    plot_annotation(tag_levels = "A") &
    (theme(plot.margin = margin(3, 3, 3, 3)) + theme_lancet_tags())
}
