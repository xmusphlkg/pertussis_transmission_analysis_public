## Extended Data Figure 4 assembly --------------------------------------------

assemble_extended_data_figure_4 <- function(panels) {
  free(panels$a) + free(panels$b) + panels$c + free(panels$d) +
    plot_layout(design = "AA\nBB\nCD", guides = "keep", heights = c(0.82, 0.82, 1.0)) +
    plot_annotation(tag_levels = "A") &
    (theme(plot.margin = margin(3, 3, 3, 3)) + theme_lancet_tags())
}
