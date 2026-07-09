## Extended Data Figure 7 assembly --------------------------------------------

assemble_extended_data_figure_7 <- function(panels) {
  free(panels$a) + free(panels$b) + free(panels$c) +
    plot_layout(design = "AC\nBB\nBB", guides = "keep", widths = c(0.95, 1.05), heights = c(0.95, 1, 1)) +
    plot_annotation(tag_levels = "A") &
    (theme(plot.margin = margin(4, 5, 4, 4)) + theme_lancet_tags())
}
