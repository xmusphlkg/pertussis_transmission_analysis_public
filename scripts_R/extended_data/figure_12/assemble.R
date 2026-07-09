## Extended Data Figure 12 assembly -------------------------------------------

assemble_extended_data_figure_12 <- function(panels) {
  ((panels$a | panels$b | panels$d) / (panels$e | panels$c | panels$f)) +
    plot_layout(guides = "keep", widths = c(1, 1, 1), heights = c(1, 1)) +
    plot_annotation(tag_levels = "A") &
    (theme(plot.margin = margin(4, 4, 4, 4)) + theme_lancet_tags())
}
