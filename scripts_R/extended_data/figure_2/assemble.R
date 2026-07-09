## Extended Data Figure 2 assembly --------------------------------------------

assemble_extended_data_figure_2 <- function(panels) {
  free(panels$a) + free(panels$b) +
    free(panels$c) + panels$d + panels$e + panels$f +
    panels$g + free(panels$h) +
    plot_layout(design = "AAAA\nBBBB\nCDEF\nGGHH", guides = "keep", heights = c(1.0, 1.0, 0.78, 1.0)) &
    (theme(plot.margin = margin(5, 3, 3, 5)) + theme_lancet_tags())
}
