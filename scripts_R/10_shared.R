args <- commandArgs(FALSE)
file_arg <- sub("^--file=", "", args[grepl("^--file=", args)])
find_scripts_dir <- function(start_dir) {
  candidate <- normalizePath(start_dir, mustWork = FALSE)
  for (i in seq_len(8)) {
    if (file.exists(file.path(candidate, "_helpers.R"))) {
      return(candidate)
    }
    parent <- dirname(candidate)
    if (identical(parent, candidate)) {
      break
    }
    candidate <- parent
  }
  normalizePath(file.path(getwd(), "scripts_R"), mustWork = FALSE)
}
script_dir <- if (length(file_arg) > 0) {
  find_scripts_dir(dirname(normalizePath(file_arg[[1]], mustWork = FALSE)))
} else {
  find_scripts_dir(file.path(getwd(), "scripts_R"))
}
source(file.path(script_dir, "_helpers.R"))

suppressPackageStartupMessages({
  library(grid)
  library(ggrepel)
})

figures_dir <- model_path("outputs", "figures")
appendix_dir <- model_path("outputs", "appendix")
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(appendix_dir, recursive = TRUE, showWarnings = FALSE)

lancet_min_image_width <- 107 / 25.4
lancet_double_width <- 183 / 25.4
lancet_single_width <- lancet_min_image_width
lancet_submission_dpi <- 300

save_figure_to_dir <- function(plot,
                               dir,
                               filename,
                               width = lancet_double_width,
                               height = 7.5,
                               dpi = lancet_submission_dpi) {
  pdf_path <- file.path(dir, paste0(filename, ".pdf"))
  png_path <- file.path(dir, paste0(filename, ".png"))
  ggsave(pdf_path, plot, width = width, height = height, device = cairo_pdf, bg = "white", limitsize = FALSE)
  ggsave(png_path, plot, width = width, height = height, dpi = dpi, bg = "white", limitsize = FALSE)
}

save_main_figure <- function(plot, filename, width = lancet_double_width, height = 7.5, dpi = lancet_submission_dpi) {
  save_figure_to_dir(plot, figures_dir, filename, width = width, height = height, dpi = dpi)
}

save_appendix_figure <- function(plot, filename, width = lancet_double_width, height = 7.5, dpi = lancet_submission_dpi) {
  save_figure_to_dir(plot, appendix_dir, filename, width = width, height = height, dpi = dpi)
}

# Lancet-style figure exports: Times-family text, 8-10 pt labels, and 300 dpi PNGs.
lancet_font_family <- Sys.getenv("LANCET_FONT_FAMILY", "Times New Roman")
lancet_decimal_mark <- "\u00B7"
lancet_text_colour <- "#232323"
lancet_grid_colour <- "#D9D9D9"
lancet_grid_light_colour <- "#ECECEC"

lancet_pt_to_geom_size <- function(pt) {
  pt / ggplot2::.pt
}

lancet_midline_decimal <- function(x) {
  gsub("\\.", lancet_decimal_mark, x)
}

format_lancet_fixed <- function(x, digits = 1) {
  lancet_midline_decimal(sprintf(paste0("%.", digits, "f"), x))
}

label_lancet_number <- function(accuracy = 1, big.mark = "", suffix = "", ...) {
  label_number(
    accuracy = accuracy,
    big.mark = big.mark,
    decimal.mark = lancet_decimal_mark,
    suffix = suffix,
    ...
  )
}

label_lancet_comma <- function(accuracy = 1, ...) {
  label_lancet_number(accuracy = accuracy, big.mark = ",", ...)
}

label_lancet_percent <- function(accuracy = 1, ...) {
  percent_format(accuracy = accuracy, decimal.mark = lancet_decimal_mark, ...)
}

lancet_percent <- function(x, accuracy = 1, ...) {
  percent(x, accuracy = accuracy, decimal.mark = lancet_decimal_mark, ...)
}

lancet_figure_text_size_pt <- 8
lancet_figure_annotation_size_pt <- 7
lancet_figure_tag_size_pt <- 10

journal_base_text_size <- lancet_figure_text_size_pt
journal_compact_text_size <- lancet_figure_text_size_pt
journal_dense_text_size <- lancet_figure_text_size_pt
journal_country_axis_text_size <- lancet_figure_text_size_pt
journal_strip_text_size <- lancet_figure_text_size_pt
journal_table_cell_text_size <- lancet_pt_to_geom_size(lancet_figure_annotation_size_pt)
journal_heatmap_cell_text_size <- lancet_pt_to_geom_size(lancet_figure_annotation_size_pt)
journal_heatmap_cell_text_size_small <- lancet_pt_to_geom_size(lancet_figure_annotation_size_pt)
journal_point_label_text_size <- lancet_pt_to_geom_size(lancet_figure_annotation_size_pt)
journal_direct_label_text_size <- lancet_pt_to_geom_size(lancet_figure_annotation_size_pt)
journal_tag_base_size <- lancet_figure_tag_size_pt
lancet_axis_linewidth <- 0.30
lancet_tick_linewidth <- 0.30
lancet_grid_linewidth <- 0.18
lancet_heatmap_tile_linewidth <- 0.18
lancet_direct_label_size <- journal_direct_label_text_size
lancet_direct_label_face <- "bold"
lancet_direct_label_segment_linewidth <- 0.15

theme_lancet <- function(base_size = journal_base_text_size,
                         base_family = lancet_font_family,
                         show_x_grid = FALSE,
                         show_y_grid = TRUE) {
  theme_classic(base_size = base_size, base_family = base_family) +
    theme(
      axis.line = element_line(linewidth = lancet_axis_linewidth, colour = lancet_text_colour),
      axis.ticks = element_line(linewidth = lancet_tick_linewidth, colour = lancet_text_colour),
      axis.ticks.length = unit(2, "pt"),
      axis.text = element_text(colour = lancet_text_colour, size = base_size, lineheight = 0.95),
      axis.title = element_text(colour = lancet_text_colour, face = "bold", size = base_size + 0.3),
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.box = "vertical",
      legend.justification = "center",
      legend.title = element_text(colour = lancet_text_colour, face = "bold", size = base_size, lineheight = 0.95),
      legend.text = element_text(colour = lancet_text_colour, size = base_size, lineheight = 0.95),
      legend.key = element_rect(fill = "white", colour = NA),
      legend.key.size = unit(0.28, "cm"),
      legend.spacing.x = unit(3, "pt"),
      legend.spacing.y = unit(1, "pt"),
      panel.background = element_rect(fill = "white", colour = NA),
      panel.border = element_blank(),
      panel.grid.major.x = if (show_x_grid) {
        element_line(linewidth = lancet_grid_linewidth, colour = lancet_grid_light_colour)
      } else {
        element_blank()
      },
      panel.grid.major.y = if (show_y_grid) {
        element_line(linewidth = lancet_grid_linewidth, colour = lancet_grid_colour)
      } else {
        element_blank()
      },
      panel.grid.minor = element_blank(),
      plot.title = element_blank(),
      plot.subtitle = element_blank(),
      plot.caption = element_blank(),
      plot.background = element_rect(fill = "white", colour = NA),
      plot.tag = element_text(face = "bold", size = journal_tag_base_size, colour = lancet_text_colour),
      plot.tag.position = c(0, 1),
      strip.background = element_blank(),
      strip.text = element_text(face = "bold", colour = lancet_text_colour, size = base_size, margin = margin(2, 0, 2, 0)),
      plot.margin = margin(4, 4, 4, 4)
    )
}

# Compact variant for heatmaps, table-like panels, and dense multipanel figures.
theme_lancet_compact <- function(base_size = journal_compact_text_size, base_family = lancet_font_family) {
  theme_lancet(base_size = base_size, base_family = base_family, show_x_grid = FALSE, show_y_grid = FALSE) +
    theme(
      axis.text = element_text(colour = lancet_text_colour, lineheight = 0.95),
      axis.title = element_text(colour = lancet_text_colour, face = "bold"),
      legend.title = element_text(colour = lancet_text_colour, face = "bold", lineheight = 0.95),
      legend.text = element_text(colour = lancet_text_colour, lineheight = 0.95)
    )
}

theme_lancet_panel <- function(base_size = journal_compact_text_size,
                               base_family = lancet_font_family,
                               plot_margin = margin(3, 3, 3, 3),
                               show_x_grid = FALSE,
                               show_y_grid = TRUE) {
  theme_lancet(
    base_size = base_size,
    base_family = base_family,
    show_x_grid = show_x_grid,
    show_y_grid = show_y_grid
  ) +
    theme(
      axis.text = element_text(colour = lancet_text_colour, lineheight = 0.95),
      axis.title = element_text(colour = lancet_text_colour, face = "bold"),
      legend.title = element_text(colour = lancet_text_colour, face = "bold", lineheight = 0.95),
      legend.text = element_text(colour = lancet_text_colour, lineheight = 0.95),
      strip.text = element_text(face = "bold", colour = lancet_text_colour),
      plot.margin = plot_margin
    )
}

theme_lancet_heatmap <- function(base_size = journal_dense_text_size,
                                  base_family = lancet_font_family,
                                  plot_margin = margin(4, 4, 4, 4),
                                  x_angle = 0,
                                  x_hjust = 0.5,
                                  x_vjust = 1,
                                  x_size = base_size,
                                  y_size = base_size,
                                  x_face = "plain",
                                  y_face = "plain",
                                  legend_position = "bottom",
                                  legend_direction = "horizontal") {
  theme_lancet_panel(
    base_size = base_size,
    base_family = base_family,
    plot_margin = plot_margin,
    show_x_grid = FALSE,
    show_y_grid = FALSE
  ) +
    theme(
      axis.line = element_blank(),
      axis.ticks = element_blank(),
      axis.text.x = element_text(
        angle = x_angle,
        hjust = x_hjust,
        vjust = x_vjust,
        size = x_size,
        face = x_face,
        colour = lancet_text_colour,
        lineheight = 0.90,
        margin = margin(t = 2)
      ),
      axis.text.y = element_text(
        size = y_size,
        face = y_face,
        colour = lancet_text_colour,
        lineheight = 0.95
      ),
      panel.grid = element_blank(),
      legend.position = legend_position,
      legend.direction = legend_direction,
      legend.title.position = "left",
      legend.title = element_text(colour = lancet_text_colour, face = "bold", lineheight = 0.95),
      legend.text = element_text(colour = lancet_text_colour, lineheight = 0.95)
    )
}

theme_lancet_tags <- function(base_size = journal_tag_base_size) {
  theme(
    plot.tag = element_text(
      family = lancet_font_family,
      face = "bold",
      size = base_size,
      colour = lancet_text_colour
    ),
    plot.tag.position = c(0, 1)
  )
}

theme_lancet_inside_legend <- function(text_size = journal_base_text_size,
                                       title_size = text_size,
                                       key_width = unit(0.28, "cm"),
                                       key_height = unit(0.2, "cm"),
                                       position = c(0.999, 0.001),
                                       justification = c(1, 0),
                                       direction = "vertical",
                                       box = NULL,
                                       box_just = NULL,
                                       title_face = "bold",
                                       title_lineheight = 0.95,
                                       title_hjust = NULL,
                                       text_lineheight = 0.95,
                                       text_hjust = NULL,
                                       text_margin = NULL,
                                       spacing_x = unit(3, "pt"),
                                       spacing_y = unit(1, "pt"),
                                       background_fill = "#FFFFFFE6",
                                       box_margin = margin(0, 0, 0, 0),
                                       legend_margin = margin(0, 0, 0, 0)) {
  theme_args <- list(
    legend.position = "inside",
    legend.position.inside = position,
    legend.justification = justification,
    legend.direction = direction,
    legend.box = box,
    legend.box.just = box_just,
    legend.background = element_rect(fill = background_fill, colour = NA),
    legend.title = element_text(
      family = lancet_font_family,
      colour = lancet_text_colour,
      face = title_face,
      size = title_size,
      lineheight = title_lineheight,
      hjust = title_hjust
    ),
    legend.text = element_text(
      family = lancet_font_family,
      colour = lancet_text_colour,
      size = text_size,
      lineheight = text_lineheight,
      hjust = text_hjust,
      margin = text_margin
    ),
    legend.key.width = key_width,
    legend.key.height = key_height,
    legend.spacing.x = spacing_x,
    legend.spacing.y = spacing_y,
    legend.box.margin = box_margin,
    legend.margin = legend_margin
  )
  do.call(theme, theme_args[!vapply(theme_args, is.null, logical(1))])
}

set_lancet_text_defaults <- function() {
  ggplot2::theme_set(theme_lancet())
  ggplot2::update_geom_defaults("text", list(
    family = lancet_font_family,
    size = journal_point_label_text_size,
    colour = lancet_text_colour
  ))
  ggplot2::update_geom_defaults("label", list(
    family = lancet_font_family,
    size = journal_point_label_text_size,
    colour = lancet_text_colour
  ))
  ggplot2::update_geom_defaults("text_repel", list(
    family = lancet_font_family,
    size = journal_point_label_text_size,
    colour = lancet_text_colour
  ))
  invisible(TRUE)
}

set_lancet_text_defaults()

guide_lancet_colourbar <- function(barwidth = unit(2.8, "cm"),
                                   barheight = unit(0.18, "cm"),
                                   title.position = "top",
                                   ...) {
  guide_colourbar(
    barwidth = barwidth,
    barheight = barheight,
    title.position = title.position,
    ...
  )
}

okabe_ito <- c(
  "#000000", "#E69F00", "#56B4E9", "#009E73",
  "#F0E442", "#0072B2", "#D55E00", "#CC79A7"
)

palette_discrete_primary_9 <- c(
  "#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF",
  "#FDAF91FF", "#AD002AFF", "#ADB6B6FF", "#1B1919FF"
)

palette_discrete_secondary_9 <- c(
  "#18B587FF", "#99CCD9FF", "#FF0069FF", "#202B2AFF", "#2AC0E7FF",
  "#24576FFF", "#6E422CFF", "#F0926FFF", "#E59EC0FF"
)

# Backward-compatible palette aliases used by older figure scripts.
palette_discrete_9 <- palette_discrete_primary_9
palette_discrete_alt_9 <- palette_discrete_secondary_9
palette_discrete_7 <- palette_discrete_primary_9[seq_len(7)]
palette_discrete_10 <- c(palette_discrete_secondary_9, palette_discrete_primary_9[[9]])

palette_continuous_orange_blue <- c(
  "#B43C3CFF", "#DFA05BFF", "#F4F1EAFF", "#8EB7BFFF", "#2D6F8EFF"
)

palette_continuous_teal_magenta <- c(
  "#0B5D66FF", "#74A69FFF", "#F4F1EAFF", "#C28A9DFF", "#8E2B5BFF"
)

shared_palettes <- list(
  discrete = list(
    primary_9 = palette_discrete_primary_9,
    secondary_9 = palette_discrete_secondary_9,
    seven = palette_discrete_7,
    ten = palette_discrete_10
  ),
  continuous = list(
    orange_blue = palette_continuous_orange_blue,
    teal_magenta = palette_continuous_teal_magenta
  )
)

manuscript_colours <- c(
  black = "#232323",
  grey = "#52575C",
  mid_grey = "#74787D",
  pale_grey = "#DCD7CE",
  light_grey = "#F4F1EA",
  pale_orange = "#EBC8A2",
  pale_green = "#D7E5D8",
  blue = palette_discrete_primary_9[[1]],
  sky = palette_discrete_secondary_9[[2]],
  green = palette_discrete_primary_9[[3]],
  orange = palette_discrete_primary_9[[6]],
  vermillion = palette_discrete_primary_9[[2]],
  magenta = palette_discrete_secondary_9[[3]],
  yellow = "#C3A05A",
  brown = palette_discrete_secondary_9[[7]],
  tan = palette_discrete_secondary_9[[8]],
  sand = "#D5BE84",
  cream = "#E9D9A6",
  lavender = palette_discrete_secondary_9[[9]],
  purple = palette_discrete_primary_9[[5]]
)

manuscript_colour <- function(name) {
  unname(manuscript_colours[[name]])
}

manuscript_discrete_core <- c(
  deep_teal = palette_discrete_primary_9[[4]],
  orange = palette_discrete_primary_9[[6]],
  blue = palette_discrete_primary_9[[1]],
  vermillion = palette_discrete_primary_9[[2]],
  green = palette_discrete_primary_9[[3]],
  purple = palette_discrete_primary_9[[5]],
  grey = palette_discrete_primary_9[[8]]
)

manuscript_discrete_extended <- c(
  vermillion = palette_discrete_primary_9[[2]],
  sky = palette_discrete_secondary_9[[2]],
  green = palette_discrete_secondary_9[[1]],
  blue = palette_discrete_secondary_9[[6]],
  salmon = palette_discrete_secondary_9[[8]],
  lavender = palette_discrete_secondary_9[[9]],
  mint = palette_discrete_secondary_9[[5]],
  red = palette_discrete_primary_9[[7]],
  brown = palette_discrete_secondary_9[[7]],
  tan = palette_discrete_primary_9[[6]]
)

manuscript_continuous_fraction <- palette_continuous_orange_blue

manuscript_continuous_burden <- palette_continuous_teal_magenta

manuscript_continuous_rank <- c(
  manuscript_colour("blue"),
  manuscript_colour("sky"),
  manuscript_colour("cream"),
  manuscript_colour("orange"),
  manuscript_colour("vermillion")
)

manuscript_continuous_reduction <- c(
  manuscript_colour("vermillion"),
  manuscript_colour("light_grey"),
  manuscript_colour("blue")
)

manuscript_binary_colours <- c(
  `FALSE` = manuscript_colour("pale_grey"),
  `TRUE` = manuscript_colour("blue")
)

manuscript_programme_colours <- c(
  "Current" = manuscript_colour("mid_grey"),
  "Timeliness" = manuscript_colour("blue"),
  "Nominal coverage floor" = manuscript_colour("mid_grey"),
  "Infant exposure" = manuscript_colour("green"),
  "Infant-exposure strategy" = manuscript_colour("green"),
  "Infant-exposure composite" = manuscript_colour("green"),
  "Direct antibody only" = manuscript_colour("sky"),
  "Adolescent" = manuscript_colour("purple"),
  "Adolescent booster" = manuscript_colour("purple"),
  "Pregnancy Tdap" = manuscript_colour("orange"),
  "Close-contact adjunct" = manuscript_discrete_core[["deep_teal"]],
  "Targeted PEP" = manuscript_colour("vermillion"),
  "Resistance mgmt" = manuscript_colour("vermillion"),
  "Resistance-guided care" = manuscript_colour("vermillion"),
  "High transmission blocking" = manuscript_colour("purple"),
  "Combined stress test" = manuscript_colour("black"),
  "Combined strategy" = manuscript_colour("black")
)

manuscript_diagnostic_colours <- c(
  "Window cells top 2" = manuscript_colour("green"),
  "Infant-age cells top 2" = manuscript_colour("magenta"),
  "Top 2" = manuscript_colour("sky"),
  "Within 10% of best" = manuscript_colour("vermillion")
)

manuscript_event_colours <- c(
  "Other cells" = manuscript_colour("mid_grey"),
  "Low infant-event cells" = manuscript_colour("vermillion")
)

manuscript_resistance_layer_colours <- c(
  "No resistance-guided care" = manuscript_colour("mid_grey"),
  "Includes resistance-guided care" = manuscript_colour("blue")
)

manuscript_temporal_colours <- c(
  "Burn-in" = manuscript_colour("sky"),
  "NPI contact shock" = manuscript_colour("orange")
)

manuscript_stochastic_colours <- c(
  "High household" = manuscript_colour("blue"),
  "Homogeneous" = manuscript_colour("green"),
  "Setting clustered" = manuscript_colour("magenta")
)

manuscript_metric_colours <- c(
  "All infections" = manuscript_colour("blue"),
  "Reported cases" = manuscript_colour("vermillion"),
  "Infant cases" = manuscript_colour("green"),
  "Treated cases" = manuscript_colour("blue"),
  "PEP-averted cases" = manuscript_colour("vermillion"),
  "Sensitive" = manuscript_colour("blue"),
  "Resistant" = manuscript_colour("vermillion")
)

manuscript_coverage_colours <- c(
  "DTP1" = manuscript_colour("blue"),
  "DTP3" = manuscript_colour("green"),
  "Maternal" = manuscript_colour("vermillion")
)

manuscript_evidence_colours <- c(
  "Measured isolate fraction" = manuscript_colour("blue"),
  "Conservative anchor" = manuscript_colour("orange")
)

manuscript_resistance_scenario_colours <- c(
  "Country timeline" = manuscript_colour("black"),
  "Low" = manuscript_colour("sky"),
  "Moderate" = manuscript_colour("green"),
  "High" = manuscript_colour("orange"),
  "Very high" = manuscript_colour("vermillion")
)

manuscript_vaccine_profile_colours <- c(
  "No vaccine" = manuscript_colour("black"),
  "Current aP" = manuscript_colour("vermillion"),
  "Inf.-blocking" = manuscript_colour("green"),
  "Trans.-blocking" = manuscript_colour("blue"),
  "Upper-bound" = manuscript_colour("magenta")
)

manuscript_country_focus_colours <- c(
  "China" = manuscript_colour("magenta"),
  "Japan" = manuscript_colour("green"),
  "Australia" = manuscript_colour("orange")
)

manuscript_source_domain_colours <- c(
  "Resistance evidence" = manuscript_colour("blue"),
  "Country input data" = manuscript_colour("green"),
  "Clinical/model assumptions" = manuscript_colour("orange"),
  "Other" = manuscript_colour("magenta")
)

manuscript_file_metric_colours <- c(
  "Files" = manuscript_colour("blue"),
  "Disk footprint, MB" = manuscript_colour("vermillion")
)

manuscript_availability_colours <- c(
  "Available" = manuscript_colour("green"),
  "Missing" = manuscript_colour("pale_grey")
)

manuscript_architecture_colours <- c(
  "Age groups" = manuscript_colour("blue"),
  "Strains" = manuscript_colour("green"),
  "Immune/dose origins" = manuscript_colour("orange"),
  "Compartments per age" = manuscript_colour("magenta"),
  "ODE state variables" = manuscript_colour("purple")
)

manuscript_compartment_colours <- c(
  "Susceptible origins" = manuscript_colour("blue"),
  "Exposed states" = manuscript_colour("green"),
  "Infectious states" = manuscript_colour("vermillion"),
  "Treated states" = manuscript_colour("orange"),
  "Natural/waned immunity" = manuscript_colour("purple")
)

manuscript_route_matrix_colours <- c(
  `FALSE` = manuscript_colour("light_grey"),
  `TRUE` = manuscript_colour("blue")
)

manuscript_parameter_audit_colours <- c(
  "PSA varied" = manuscript_colour("green"),
  "Grid override" = manuscript_colour("orange"),
  "Excluded or fixed" = manuscript_colour("pale_grey")
)

manuscript_fitness_group_colours <- c(
  "Fitness cost" = manuscript_colour("blue"),
  "Neutral" = manuscript_colour("mid_grey"),
  "Advantage" = manuscript_colour("vermillion"),
  "Fitness cost (0.85)" = manuscript_colour("blue"),
  "Neutral (1.00)" = manuscript_colour("mid_grey"),
  "Advantage (1.10)" = manuscript_colour("vermillion")
)

scale_fill_fraction <- function(..., colours = manuscript_continuous_fraction) {
  scale_fill_gradientn(colours = colours, ...)
}

scale_fill_resistance_fraction <- function(..., colours = manuscript_continuous_fraction) {
  scale_fill_fraction(..., colours = colours)
}

scale_fill_infant_burden <- function(..., colours = manuscript_continuous_burden) {
  scale_fill_gradientn(colours = colours, ...)
}

scale_fill_rank <- function(..., colours = manuscript_continuous_rank) {
  scale_fill_gradientn(colours = colours, ...)
}

scale_fill_reduction <- function(...,
                                 low = manuscript_continuous_reduction[[1]],
                                 mid = manuscript_continuous_reduction[[2]],
                                 high = manuscript_continuous_reduction[[3]]) {
  scale_fill_gradient2(low = low, mid = mid, high = high, ...)
}

palette_for_labels <- function(palette, labels, reference_labels = NULL) {
  label_values <- as.character(labels)
  if (!is.null(reference_labels) && !is.null(names(labels))) {
    lookup <- as.character(reference_labels[names(labels)])
  } else {
    lookup <- label_values
  }
  setNames(unname(palette[lookup]), label_values)
}

interval_quantile <- function(x, prob) {
  x <- x[is.finite(x)]
  if (length(x) == 0) {
    return(NA_real_)
  }
  as.numeric(stats::quantile(x, probs = prob, na.rm = TRUE, names = FALSE))
}

interval_label <- function(median, low, high, formatter = label_lancet_number(accuracy = 1)) {
  paste0(formatter(median), "\n[", formatter(low), "–", formatter(high), "]")
}

source(file.path(script_dir, "10_plot_helpers.R"))

# WHO-region-first display order used across manuscript and appendix figures.
# Thailand is a South-East Asia Region profile, not a Western Pacific profile.
country_region_levels <- c(
  "European Region",
  "Western Pacific Region",
  "South-East Asia Region",
  "Region of the Americas",
  "African Region"
)
country_who_regions <- c(
  Sweden = "European Region",
  United_Kingdom = "European Region",
  Australia = "Western Pacific Region",
  China = "Western Pacific Region",
  Japan = "Western Pacific Region",
  New_Zealand = "Western Pacific Region",
  Thailand = "South-East Asia Region",
  Brazil = "Region of the Americas",
  United_States = "Region of the Americas",
  South_Africa = "African Region"
)
country_levels <- names(country_who_regions)
country_label_levels <- stringr::str_replace_all(country_levels, "_", " ")

# Fixed order for the nine calibrated profiles in all main-figure country axes.
main_figure_country_levels <- country_levels[country_levels != "South_Africa"]
main_figure_country_label_levels <- stringr::str_replace_all(
  main_figure_country_levels,
  "_",
  " "
)

main_figure_country_order <- function(countries) {
  labels <- unique(stringr::str_replace_all(as.character(countries), "_", " "))
  c(
    main_figure_country_label_levels[
      main_figure_country_label_levels %in% labels
    ],
    sort(setdiff(labels, main_figure_country_label_levels))
  )
}
country_codes <- c(
  Australia = "AUS",
  China = "CHN",
  Japan = "JPN",
  New_Zealand = "NZL",
  South_Africa = "ZAF",
  Sweden = "SWE",
  United_Kingdom = "GBR",
  United_States = "USA",
  Brazil = "BRA",
  Thailand = "THA"
)

format_country <- function(x) {
  stringr::str_replace_all(x, "_", " ")
}

add_country_label <- function(df) {
  df %>%
    mutate(
      country = stringr::str_replace_all(country, " ", "_"),
      country_label = factor(format_country(country), levels = country_label_levels)
    )
}

age_labels <- c(
  infant_0_2m = "0–2 m",
  infant_3_11m = "3–11 m",
  child_1_4y = "1–4 y",
  child_5_9y = "5–9 y",
  adolescent_10_17y = "10–17 y",
  young_adult_18_39y = "18–39 y",
  middle_adult_40_64y = "40–64 y",
  elderly_65plus = "65+ y"
)

vaccine_levels <- c(
  "no_vaccine", "symptom_protective", "infection_blocking",
  "transmission_blocking", "next_generation"
)
vaccine_labels <- c(
  no_vaccine = "No vaccine",
  symptom_protective = "Current aP profile",
  infection_blocking = "Infection-blocking",
  transmission_blocking = "Transmission-blocking",
  next_generation = "Upper-bound transmission-blocking"
)

resistance_levels <- c("country_timeline", "low", "moderate", "high", "very_high")
resistance_labels <- c(
  country_timeline = "Country timeline",
  low = "Low",
  moderate = "Moderate",
  high = "High",
  very_high = "Very high"
)

intervention_levels <- c(
  "higher_child_coverage", "adolescent_booster",
  "pregnancy_tdap_scaleup", "cocooning_adjunct",
  "maternal_immunization", "targeted_pep_high_risk",
  "resistance_guided_treatment",
  "next_generation_vaccine", "combined_strategy"
)
intervention_labels <- c(
  higher_child_coverage = "Nominal coverage floor",
  adolescent_booster = "Adolescent booster",
  pregnancy_tdap_scaleup = "Pregnancy Tdap scale-up",
  cocooning_adjunct = "Close-contact adjunct",
  maternal_immunization = "Infant-exposure\npackage",
  targeted_pep_high_risk = "Targeted high-risk PEP",
  resistance_guided_treatment = "Resistance-guided management",
  next_generation_vaccine = "High transmission-blocking\nvaccine target",
  combined_strategy = "Combined strategy"
)

reporting_levels <- c(
  "low", "medium", "high", "age_biased", "time_varying",
  "infant_high_adult_very_low", "infant_moderate_adult_minimal",
  "enhanced_surveillance", "adult_focused_improvement", "china_passive_system"
)
reporting_labels <- c(
  low = "Low",
  medium = "Medium",
  high = "High",
  age_biased = "Age-biased",
  time_varying = "Time-varying",
  infant_high_adult_very_low = "Infant high,\nadult very low",
  infant_moderate_adult_minimal = "Infant moderate,\nadult minimal",
  enhanced_surveillance = "Enhanced\nsurveillance",
  adult_focused_improvement = "Adult-focused\nimprovement",
  china_passive_system = "China passive\nsystem"
)

metric_labels <- c(
  annualized_infections_per_100k = "All infections",
  annualized_reported_cases_per_100k = "Reported cases",
  annualized_infant_cases_per_100k = "Infant cases",
  relative_reduction_infant_cases = "Infant cases",
  relative_reduction_total_infections = "All infections",
  relative_reduction_reported_cases = "Reported cases",
  relative_reduction_resistant_infections = "Resistant infections"
)

region_colours <- c(
  "Global" = manuscript_colours[["black"]],
  "Western Pacific Region" = manuscript_discrete_core[["blue"]],
  "South-East Asia Region" = manuscript_discrete_core[["purple"]],
  "European Region" = manuscript_discrete_core[["vermillion"]],
  "Region of the Americas" = manuscript_discrete_core[["green"]],
  "African Region" = manuscript_discrete_core[["orange"]]
)

region_short_colours <- c(
  Neutral = manuscript_colours[["light_grey"]],
  WPR = region_colours[["Western Pacific Region"]],
  SEAR = region_colours[["South-East Asia Region"]],
  EUR = region_colours[["European Region"]],
  AMR = region_colours[["Region of the Americas"]],
  AFR = region_colours[["African Region"]],
  Other = manuscript_colours[["mid_grey"]]
)

country_colour_levels <- c(
  "Australia", "China", "Japan", "New_Zealand", "South_Africa",
  "Sweden", "United_Kingdom", "United_States", "Brazil", "Thailand"
)
country_colours <- setNames(
  palette_discrete_10,
  stringr::str_replace_all(country_colour_levels, "_", " ")
)

vaccine_colours <- c(
  "No vaccine" = manuscript_discrete_core[["grey"]],
  "Current aP profile" = manuscript_discrete_core[["vermillion"]],
  "Infection-blocking" = manuscript_discrete_core[["green"]],
  "Transmission-blocking" = manuscript_discrete_core[["blue"]],
  "Upper-bound transmission-blocking" = manuscript_discrete_core[["purple"]]
)

source_history_colours <- c(
  "Unvaccinated" = manuscript_colours[["mid_grey"]],
  "Maternal" = manuscript_colours[["sky"]],
  "Dose 1" = manuscript_colours[["green"]],
  "Dose 2" = manuscript_colours[["orange"]],
  "Dose 3+" = manuscript_colours[["purple"]],
  "Waned" = manuscript_colours[["vermillion"]]
)

metric_colours <- c(
  "Reported cases" = manuscript_discrete_core[["vermillion"]],
  "Infant cases" = manuscript_discrete_core[["green"]],
  "All infections" = manuscript_discrete_core[["blue"]],
  "Resistant infections" = manuscript_discrete_core[["purple"]]
)

resistance_colours <- c(
  "Country timeline" = manuscript_discrete_core[["grey"]],
  "Low" = manuscript_discrete_core[["blue"]],
  "Moderate" = manuscript_discrete_core[["green"]],
  "High" = manuscript_discrete_core[["orange"]],
  "Very high" = manuscript_discrete_core[["vermillion"]]
)

strategy_colours <- c(
  current = manuscript_colours[["mid_grey"]],
  higher_child_coverage = manuscript_colours[["mid_grey"]],
  timeliness_only = manuscript_discrete_core[["blue"]],
  adolescent_booster = manuscript_discrete_core[["orange"]],
  pregnancy_tdap_scaleup = manuscript_discrete_core[["purple"]],
  cocooning_adjunct = manuscript_discrete_core[["deep_teal"]],
  maternal_immunization = manuscript_discrete_core[["green"]],
  targeted_pep_high_risk = manuscript_discrete_core[["vermillion"]],
  resistance_guided_treatment = manuscript_discrete_core[["vermillion"]],
  transmission_blocking_vaccine = manuscript_discrete_core[["deep_teal"]],
  next_generation_vaccine = manuscript_discrete_core[["blue"]],
  combined_strategy = manuscript_colours[["black"]]
)

fitness_colours <- setNames(
  grDevices::colorRampPalette(palette_continuous_teal_magenta)(7),
  c("0.85", "0.90", "0.95", "1.00", "1.05", "1.10", "1.15")
)

bayesian_validity_summary_path <- model_path(
  "outputs", "summaries", "bayesian_convergence_summary.txt"
)

bayesian_posterior_validated <- function(path = bayesian_validity_summary_path) {
  if (!file.exists(path)) {
    return(FALSE)
  }
  text <- paste(readLines(path, warn = FALSE), collapse = "\n")
  grepl("All parameters converged:\\s*True", text)
}

use_bayesian_posterior_outputs <- bayesian_posterior_validated()
allow_efigure_summary_fallback <- Sys.getenv("PERTUSSIS_ALLOW_EFIGURE_SUMMARY_FALLBACK") == "1"

require_efigure_timeseries <- function(df, figure_label, output_stem) {
  if (nrow(df) > 0 || allow_efigure_summary_fallback) {
    return(invisible(df))
  }
  stop(
    figure_label,
    " publication rendering requires outputs/simulations/",
    output_stem,
    " time-series data from `make simulate`. Set ",
    "PERTUSSIS_ALLOW_EFIGURE_SUMMARY_FALLBACK=1 only for exploratory summary-only rendering.",
    call. = FALSE
  )
}

if (!isTRUE(getOption("pertussis.skip_shared_data", FALSE))) {
baseline <- read_model_table(model_path("outputs", "summaries", "country_scenarios_summary")) %>%
  add_country_label()
vaccine_summary <- read_model_table(model_path("outputs", "summaries", "vaccine_scenarios_summary")) %>%
  add_country_label() %>%
  mutate(
    scenario = factor(scenario, levels = vaccine_levels),
    scenario_label = factor(vaccine_labels[as.character(scenario)], levels = vaccine_labels[vaccine_levels])
  )
resistance_summary <- read_model_table(model_path("outputs", "summaries", "resistance_scenarios_summary")) %>%
  add_country_label() %>%
  mutate(
    scenario = factor(scenario, levels = resistance_levels),
    scenario_label = factor(resistance_labels[as.character(scenario)], levels = resistance_labels[resistance_levels])
  )
intervention_summary <- read_model_table(model_path("outputs", "summaries", "intervention_scenarios_summary")) %>%
  add_country_label() %>%
  mutate(
    scenario = factor(scenario, levels = c("current", intervention_levels)),
    scenario_label = factor(
      c(current = "Current", intervention_labels)[as.character(scenario)],
      levels = c("Current", intervention_labels[intervention_levels])
    )
  )
grid_summary <- read_model_table(model_path("outputs", "summaries", "fitness_resistance_grid_summary")) %>%
  add_country_label() %>%
  mutate(
    grid_VE_inf = as.numeric(grid_VE_inf),
    grid_resistance_prevalence = as.numeric(ifelse("grid_resistance_prevalence" %in% names(.), grid_resistance_prevalence, grid_fitness_R))
  )
reporting_summary <- read_model_table(model_path("outputs", "summaries", "reporting_scenarios_summary")) %>%
  add_country_label() %>%
  mutate(
    scenario = factor(scenario, levels = reporting_levels),
    scenario_label = factor(reporting_labels[as.character(scenario)], levels = reporting_labels[reporting_levels])
  )
sensitivity_summary <- read_model_table(model_path("outputs", "summaries", "sensitivity_runs_summary"))
fitness_summary <- read_model_table_optional(model_path("outputs", "summaries", "fitness_resistance_grid_summary"))
if (nrow(fitness_summary) > 0) {
  fitness_summary <- fitness_summary %>%
    add_country_label() %>%
    mutate(
      grid_fitness_R = as.numeric(grid_fitness_R),
      grid_VE_inf = as.numeric(grid_VE_inf)
    )
}
fitness_benefit_summary <- read_model_table_optional(
  model_path("outputs", "summaries", "fitness_resistance_grid_posterior_benefit_summary")
)
if (nrow(fitness_benefit_summary) > 0) {
  fitness_benefit_summary <- fitness_benefit_summary %>%
    add_country_label() %>%
    mutate(
      fitness_group = factor(
        fitness_group,
        levels = c("Fitness cost (0.85)", "Neutral (1.00)", "Advantage (1.10)")
      ),
      grid_fitness_R = as.numeric(grid_fitness_R),
      low_grid_VE_inf = as.numeric(low_grid_VE_inf),
      high_grid_VE_inf = as.numeric(high_grid_VE_inf),
      median_relative_benefit = as.numeric(median_relative_benefit),
      q025_relative_benefit = as.numeric(q025_relative_benefit),
      q975_relative_benefit = as.numeric(q975_relative_benefit),
      posterior_draws = as.integer(posterior_draws),
      uncertainty_source = as.character(uncertainty_source),
      uncertainty_scope = if ("uncertainty_scope" %in% names(.)) as.character(uncertainty_scope) else NA_character_
    )
}
fitness_benefit_parameter_diagnostics <- read_model_table_optional(
  model_path("outputs", "summaries", "fitness_resistance_grid_posterior_sample_diagnostics")
)
if (nrow(fitness_benefit_parameter_diagnostics) > 0) {
  fitness_benefit_parameter_diagnostics <- fitness_benefit_parameter_diagnostics %>%
    add_country_label() %>%
    mutate(
      posterior_draws = as.integer(posterior_draws),
      unique_values = as.integer(unique_values),
      mean = as.numeric(mean),
      sd = as.numeric(sd),
      coefficient_of_variation = as.numeric(coefficient_of_variation),
      q025 = as.numeric(q025),
      median = as.numeric(median),
      q975 = as.numeric(q975),
      varies_within_country = as.logical(varies_within_country),
      grid_override_in_fig3d = as.logical(grid_override_in_fig3d),
      uncertainty_source = as.character(uncertainty_source)
    )
}
fitness_benefit_psa_summary <- read_model_table_optional(
  model_path("outputs", "tables", "fitness_resistance_grid_psa_benefit_summary")
)
if (nrow(fitness_benefit_psa_summary) > 0) {
  fitness_benefit_psa_summary <- fitness_benefit_psa_summary %>%
    add_country_label() %>%
    mutate(
      fitness_group = factor(
        fitness_group,
        levels = c("Fitness cost (0.85)", "Neutral (1.00)", "Advantage (1.10)")
      ),
      grid_fitness_R = as.numeric(grid_fitness_R),
      low_grid_VE_inf = as.numeric(low_grid_VE_inf),
      high_grid_VE_inf = as.numeric(high_grid_VE_inf),
      median_relative_benefit = as.numeric(median_relative_benefit),
      q025_relative_benefit = as.numeric(q025_relative_benefit),
      q975_relative_benefit = as.numeric(q975_relative_benefit),
      psa_samples = as.integer(psa_samples),
      uncertainty_source = as.character(uncertainty_source),
      uncertainty_scope = if ("uncertainty_scope" %in% names(.)) as.character(uncertainty_scope) else NA_character_
    )
}
bayesian_summary <- tibble()
if (use_bayesian_posterior_outputs) {
  bayesian_summary <- read_model_table_optional(model_path("outputs", "summaries", "bayesian_uncertainty_figure2c_conditional_summary"))
}
if (nrow(bayesian_summary) > 0) {
  bayesian_summary <- bayesian_summary %>% add_country_label()
}

baseline_order <- main_figure_country_order(as.character(baseline$country_label))

baseline <- baseline %>%
  mutate(country_burden_order = factor(as.character(country_label), levels = rev(baseline_order)))

with_burden_order <- function(df) {
  df %>%
    mutate(country_burden_order = factor(as.character(country_label), levels = rev(baseline_order)))
}

vaccine_summary <- with_burden_order(vaccine_summary)
resistance_summary <- with_burden_order(resistance_summary)
intervention_summary <- with_burden_order(intervention_summary)
grid_summary <- with_burden_order(grid_summary)
reporting_summary <- with_burden_order(reporting_summary)
if (nrow(fitness_summary) > 0) {
  fitness_summary <- with_burden_order(fitness_summary)
}
if (nrow(fitness_benefit_summary) > 0) {
  fitness_benefit_summary <- with_burden_order(fitness_benefit_summary)
}
if (nrow(fitness_benefit_parameter_diagnostics) > 0) {
  fitness_benefit_parameter_diagnostics <- with_burden_order(fitness_benefit_parameter_diagnostics)
}
if (nrow(fitness_benefit_psa_summary) > 0) {
  fitness_benefit_psa_summary <- with_burden_order(fitness_benefit_psa_summary)
}
if (nrow(bayesian_summary) > 0) {
  bayesian_summary <- with_burden_order(bayesian_summary)
}
}
