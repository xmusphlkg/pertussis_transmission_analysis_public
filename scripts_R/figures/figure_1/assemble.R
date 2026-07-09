## Figure 1 assembly -----------------------------------------------------------

assemble_figure_1 <- function(panels) {
  free(panels$a) + panels$b + panels$c + panels$d +
    plot_layout(widths = c(0.98, 1.02)) &
    theme_lancet_tags()
}
