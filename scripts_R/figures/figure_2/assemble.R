## Figure 2 assembly -----------------------------------------------------------

assemble_figure_2 <- function(panels) {
  required_panels <- c("a", "b", "c")
  missing_panels <- setdiff(required_panels, names(panels))
  if (length(missing_panels) > 0L) {
    stop(
      "Figure 2 assembly is missing panel(s): ",
      paste(missing_panels, collapse = ", "),
      call. = FALSE
    )
  }

  design <- "
AB
CC
"

  # The delivery-lever contrast and consequence-aware fragility audit introduce
  # the decision logic; the complete effect surface remains the full-width hero.
  # The shared export contract supplies the 183 mm page width.
  panels$a + panels$b + free(panels$c) +
    plot_layout(
      design = design,
      heights = c(0.95, 1.08),
      guides = "keep"
    ) &
    theme_lancet_tags()
}
