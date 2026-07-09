## Extended Data Figure 12 render entry point ---------------------------------

if (exists("script_dir", inherits = TRUE) && file.exists(file.path(get("script_dir"), "extended_data", "figure_12", "data.R"))) {
  script_dir <- normalizePath(get("script_dir"), mustWork = FALSE)
  extended_data_figure_12_dir <- file.path(script_dir, "extended_data", "figure_12")
} else {
  extended_data_figure_12_file <- tryCatch(normalizePath(sys.frame(1)$ofile, mustWork = FALSE), error = function(e) NA_character_)
  if (is.na(extended_data_figure_12_file)) {
    extended_data_figure_12_file <- file.path(getwd(), "scripts_R", "extended_data", "figure_12", "render.R")
  }
  extended_data_figure_12_dir <- dirname(extended_data_figure_12_file)
  script_dir <- normalizePath(file.path(extended_data_figure_12_dir, "..", ".."), mustWork = FALSE)
}

source(normalizePath(file.path(extended_data_figure_12_dir, "..", "..", "lib", "bootstrap.R"), mustWork = FALSE))
source(file.path(extended_data_figure_12_dir, "data.R"))
source(file.path(extended_data_figure_12_dir, "source_data.R"))
source(file.path(extended_data_figure_12_dir, "panels.R"))
source(file.path(extended_data_figure_12_dir, "assemble.R"))

render_extended_data_figure_12 <- function(write_source = TRUE, save_plot = TRUE) {
  data <- prepare_extended_data_figure_12_data()
  source_data <- make_extended_data_figure_12_source_data(data)
  if (isTRUE(write_source)) {
    write_extended_data_figure_12_source_data(source_data)
  }

  panels <- plot_extended_data_figure_12_panels(data)
  figure <- assemble_extended_data_figure_12(panels)

  if (isTRUE(save_plot)) {
    save_appendix_figure(
      figure,
      "extended_data_figure_12_fitness_resistance_grid",
      width = lancet_double_width * 1.35,
      height = 7.6
    )
    cat("Extended Data Figure 12 saved.\n")
  }

  invisible(list(
    data = data,
    source_data = source_data,
    panels = panels,
    figure = figure
  ))
}
