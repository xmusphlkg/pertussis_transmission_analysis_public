args <- commandArgs(FALSE)
file_arg <- sub("^--file=", "", args[grepl("^--file=", args)])
script_dir <- if (length(file_arg) > 0) dirname(normalizePath(file_arg[[1]])) else file.path(getwd(), "scripts_R")
source(file.path(script_dir, "10_shared.R"))

schematic_theme <- function() {
  theme_void(base_family = lancet_font_family) +
    theme(
      plot.margin = margin(2, 3, 2, 3)
    )
}

node_fill <- c(
  susceptible = manuscript_colour("cream"),
  infection = manuscript_colour("pale_orange"),
  treated = manuscript_colour("lavender"),
  immunity = manuscript_colour("pale_green"),
  note = manuscript_colour("light_grey")
)

draw_edges <- function(edges, colour = manuscript_colour("grey"), linetype = "solid") {
  geom_segment(
    data = edges,
    aes(x = x, y = y, xend = xend, yend = yend),
    arrow = arrow(length = unit(1.45, "mm"), type = "closed"),
    linewidth = 0.30,
    lineend = "round",
    colour = colour,
    linetype = linetype,
    inherit.aes = FALSE
  )
}

draw_compartment_nodes <- function(nodes) {
  list(
    geom_rect(
      data = nodes,
      aes(xmin = x - w / 2, xmax = x + w / 2, ymin = y - h / 2, ymax = y + h / 2, fill = fill),
      colour = manuscript_colour("black"),
      linewidth = 0.24,
      inherit.aes = FALSE
    ),
    geom_text(
      data = nodes,
      aes(x = x, y = y, label = label),
      parse = TRUE,
      size = 2.45,
      family = lancet_font_family,
      inherit.aes = FALSE
    )
  )
}

draw_plain_nodes <- function(nodes, size = 1.95) {
  list(
    geom_rect(
      data = nodes,
      aes(xmin = x - w / 2, xmax = x + w / 2, ymin = y - h / 2, ymax = y + h / 2, fill = fill),
      colour = manuscript_colour("black"),
      linewidth = 0.24,
      inherit.aes = FALSE
    ),
    geom_text(
      data = nodes,
      aes(x = x, y = y, label = label),
      size = size,
      lineheight = 0.88,
      family = lancet_font_family,
      inherit.aes = FALSE
    )
  )
}

math_label <- function(x, y, label, size = 1.8, hjust = 0.5, vjust = 0.5) {
  annotate(
    "text",
    x = x,
    y = y,
    label = label,
    parse = TRUE,
    size = size,
    hjust = hjust,
    vjust = vjust,
    family = lancet_font_family
  )
}

scale_schematic_fill <- scale_fill_identity()

schematic <- {
  nodes <- tribble(
    ~label, ~x, ~y, ~w, ~h, ~fill,
    "S[i*','*o]", 0.85, 2.05, 1.12, 0.58, node_fill["susceptible"],
    "E[i*','*k*','*o]", 2.35, 2.05, 1.14, 0.58, node_fill["infection"],
    "I[i*','*k*','*o]^sym", 3.92, 2.78, 1.28, 0.60, node_fill["infection"],
    "I[i*','*k*','*o]^asym", 3.92, 1.32, 1.28, 0.60, node_fill["infection"],
    "T[i*','*k*','*o]", 5.55, 2.05, 1.12, 0.58, node_fill["treated"],
    "R[i]", 7.05, 2.05, 0.96, 0.58, node_fill["immunity"],
    "W[i]", 8.55, 2.05, 0.96, 0.58, node_fill["immunity"],
    "S[i*','*U]", 10.05, 2.05, 1.12, 0.58, node_fill["susceptible"]
  )
  edges <- tribble(
    ~x, ~y, ~xend, ~yend,
    1.45, 2.05, 1.74, 2.05,
    2.96, 2.18, 3.25, 2.58,
    2.96, 1.92, 3.24, 1.52,
    4.58, 2.72, 4.96, 2.25,
    4.58, 1.38, 4.96, 1.85,
    6.14, 2.05, 6.47, 2.05,
    7.63, 2.05, 7.97, 2.05,
    9.13, 2.05, 9.44, 2.05
  )
  ggplot() +
    draw_edges(edges) +
    geom_curve(
      aes(x = 4.56, y = 3.02, xend = 6.48, yend = 2.32),
      curvature = -0.12,
      arrow = arrow(length = unit(1.25, "mm"), type = "closed"),
      linewidth = 0.26,
      colour = manuscript_colour("mid_grey"),
      inherit.aes = FALSE
    ) +
    geom_curve(
      aes(x = 4.56, y = 1.08, xend = 6.48, yend = 1.78),
      curvature = 0.12,
      arrow = arrow(length = unit(1.25, "mm"), type = "closed"),
      linewidth = 0.26,
      colour = manuscript_colour("mid_grey"),
      inherit.aes = FALSE
    ) +
    geom_curve(
      aes(x = 8.42, y = 2.44, xend = 7.18, yend = 2.44),
      curvature = 0.42,
      arrow = arrow(length = unit(1.35, "mm"), type = "closed"),
      linewidth = 0.28,
      colour = manuscript_colour("mid_grey"),
      inherit.aes = FALSE
    ) +
    draw_compartment_nodes(nodes) +
    scale_schematic_fill +
    coord_cartesian(xlim = c(0.05, 10.70), ylim = c(0.72, 3.15), clip = "off") +
    schematic_theme()
}

origin_effects <- tibble(
  origin = c(
    "Unvaccinated", "Maternal", "Dose 1 recent", "Dose 1 waned",
    "Dose 2 recent", "Dose 2 waned", "Dose 3+ recent", "Dose 3+ waned"
  ),
  relative_effect = c(0.00, 0.75, 0.45, 0.45 * 0.35, 0.75, 0.75 * 0.35, 1.00, 0.35)
) %>%
  mutate(origin = factor(origin, levels = origin))

p_ed3c <- origin_effects %>%
  ggplot(aes(relative_effect, origin, fill = relative_effect)) +
  geom_col(width = 0.66) +
  geom_text(aes(label = percent(relative_effect, accuracy = 1)), hjust = -0.12, size = 2) +
  scale_x_continuous(labels = percent_format(accuracy = 1), expand = expansion(mult = c(0, 0.16))) +
  scale_fill_fraction(guide = "none") +
  labs(x = "Relative vaccine-effect weight", y = NULL) +
  theme_lancet()

extended3 <- free(schematic) + free(p_ed3c) +
  plot_layout(design = "A\nB", guides = "keep", heights = c(1.25, 0.75)) +
  plot_annotation(tag_levels = "A") &
  (theme(plot.margin = margin(3, 3, 3, 3)) + theme_lancet_tags())

save_appendix_figure(extended3, "extended_data_figure_3_model_structure", height = 6.8)
