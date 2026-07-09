## Extended Data Figure 10 assembly -------------------------------------------

assemble_extended_data_figure_10 <- function(panels) {
  layout <- "
AB
CC
DE
"

  panels$a + panels$b + free(panels$c) + panels$d + panels$e +
    plot_layout(design = layout, heights = c(0.90, 1.30, 1.00), guides = "keep") +
    plot_annotation(tag_levels = "A") &
    (theme(plot.margin = margin(4, 4, 4, 4)) + theme_lancet_tags())
}
