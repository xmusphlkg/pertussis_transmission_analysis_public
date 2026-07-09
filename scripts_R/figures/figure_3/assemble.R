## Figure 3 assembly -----------------------------------------------------------

assemble_figure_3 <- function(panels) {
  design <- "
AA
BC
"

  free(panels$a) + panels$b + panels$c +
    plot_layout(design = design, heights = c(0.98, 0.92)) &
    theme_lancet_tags()
}
