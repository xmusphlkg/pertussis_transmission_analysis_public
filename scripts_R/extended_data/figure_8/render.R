## Extended Data Figure 8 render entry point ----------------------------------

if (exists("script_dir", inherits = TRUE) &&
    file.exists(file.path(get("script_dir"), "extended_data", "figure_8", "data.R"))) {
  script_dir <- normalizePath(get("script_dir"), mustWork = FALSE)
  extended_data_figure_8_dir <- file.path(script_dir, "extended_data", "figure_8")
} else {
  extended_data_figure_8_file <- tryCatch(normalizePath(sys.frame(1)$ofile, mustWork = FALSE), error = function(e) NA_character_)
  if (is.na(extended_data_figure_8_file)) {
    extended_data_figure_8_file <- file.path(getwd(), "scripts_R", "extended_data", "figure_8", "render.R")
  }
  extended_data_figure_8_dir <- dirname(extended_data_figure_8_file)
  script_dir <- normalizePath(file.path(extended_data_figure_8_dir, "..", ".."), mustWork = FALSE)
}

source(normalizePath(file.path(extended_data_figure_8_dir, "..", "..", "lib", "bootstrap.R"), mustWork = FALSE))
source(file.path(extended_data_figure_8_dir, "data.R"))
source(file.path(extended_data_figure_8_dir, "source_data.R"))
source(file.path(extended_data_figure_8_dir, "panels.R"))
source(file.path(extended_data_figure_8_dir, "assemble.R"))

render_extended_data_figure_8 <- function(write_source = TRUE, save_plot = TRUE) {
  data <- prepare_extended_data_figure_8_data()
  source_data <- make_extended_data_figure_8_source_data(data)
  if (isTRUE(write_source)) {
    write_extended_data_figure_8_source_data(source_data)
  }

  panels <- plot_extended_data_figure_8_panels(data)
  figure <- assemble_extended_data_figure_8(panels)

  if (isTRUE(save_plot)) {
    save_appendix_figure(figure, "extended_data_figure_8_resistance_hindcast", height = 6.4)
    cat("eFigure 8 (resistance hindcast plausibility checks) saved.\n")
  }

  invisible(list(
    data = data,
    source_data = source_data,
    panels = panels,
    figure = figure
  ))
}
