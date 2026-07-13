## Figure 4 assembly -----------------------------------------------------------

assemble_figure_4 <- function(panels) {
  design <- "
AAB
CDD
EEF
"

  panels$a + panels$b + free(panels$c) + free(panels$d) +
    free(panels$future_residual) + free(panels$future_veinf) +
    plot_layout(
      design = design,
      widths = c(0.500, 0.265, 0.735),
      heights = c(1.00, 1.02, 0.92),
      guides = "keep"
    ) &
    theme_lancet_tags()
}
