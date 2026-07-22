## Figure 4 assembly -----------------------------------------------------------

assemble_figure_4 <- function(panels) {
  required_panels <- c("a", "b", "c")
  missing_panels <- setdiff(required_panels, names(panels))
  if (length(missing_panels) > 0L) {
    stop(
      "Figure 4 assembly is missing panel(s): ",
      paste(missing_panels, collapse = ", "),
      call. = FALSE
    )
  }

  design <- "
AB
CC
"

  free(panels$a) + panels$b + free(panels$c) +
    plot_layout(design = design, heights = c(1.12, 0.96), guides = "keep") &
    theme_lancet_tags()
}
