## Extended Data Figure 8 assembly --------------------------------------------

assemble_extended_data_figure_8 <- function(panels) {
  panels$a + panels$b + panels$c + panels$d +
    plot_layout(ncol = 2, guides = "collect") +
    plot_annotation(tag_levels = "A") &
    (theme(
      legend.position = "bottom",
      legend.box = "horizontal",
      plot.margin = margin(3, 3, 3, 3)
    ) + theme_lancet_tags())
}
