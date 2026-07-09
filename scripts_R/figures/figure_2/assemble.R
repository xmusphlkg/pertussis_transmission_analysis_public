## Figure 2 assembly -----------------------------------------------------------

assemble_figure_2 <- function(panels) {
  design <- "
AB
CC
"

  panels$a + panels$b + free(panels$c) +
    plot_layout(heights = c(0.88, 1.18), design = design, guides = "keep") &
    theme_lancet_tags()
}
