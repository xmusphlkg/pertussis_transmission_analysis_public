## Extended Data Figure 9 assembly --------------------------------------------

assemble_extended_data_figure_9 <- function(panels) {
  layout <- "
ABC
DEF
GGG
"

  wrap_plots(
    A = free(panels$a), B = panels$b, C = panels$c,
    D = panels$d, E = free(panels$e), F = panels$f,
    G = free(panels$g),
    design = layout,
    guides = "keep"
  ) +
    plot_layout(heights = c(1, 1, 0.76)) +
    plot_annotation(tag_levels = "A") &
    (theme(plot.margin = margin(4, 4, 4, 4)) + theme_lancet_tags())
}
