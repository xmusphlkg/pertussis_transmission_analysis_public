## Figure 4 render entry point -------------------------------------------------

if (exists("script_dir", inherits = TRUE) && file.exists(file.path(get("script_dir"), "figures", "figure_4", "data.R"))) {
  script_dir <- normalizePath(get("script_dir"), mustWork = FALSE)
  figure_4_dir <- file.path(script_dir, "figures", "figure_4")
} else {
  figure_4_file <- tryCatch(normalizePath(sys.frame(1)$ofile, mustWork = FALSE), error = function(e) NA_character_)
  if (is.na(figure_4_file)) {
    figure_4_file <- file.path(getwd(), "scripts_R", "figures", "figure_4", "render.R")
  }
  figure_4_dir <- dirname(figure_4_file)
  script_dir <- normalizePath(file.path(figure_4_dir, "..", ".."), mustWork = FALSE)
}

source(normalizePath(file.path(figure_4_dir, "..", "..", "lib", "bootstrap.R"), mustWork = FALSE))
source(file.path(figure_4_dir, "data.R"))
source(file.path(figure_4_dir, "source_data.R"))
source(file.path(figure_4_dir, "panels.R"))
source(file.path(figure_4_dir, "assemble.R"))

render_figure_4 <- function(write_source = TRUE, save_plot = TRUE) {
  data <- prepare_figure_4_data()
  source_data <- make_figure_4_source_data(data)
  if (isTRUE(write_source)) {
    write_figure_4_source_data(source_data)
  }

  panels <- plot_figure_4_panels(data)
  figure <- assemble_figure_4(panels)

  if (isTRUE(save_plot)) {
    save_main_figure(figure, "figure_4_resistance_conditioned_control", height = 6.35)
    cat("Figure 4 saved.\n")
  }

  invisible(list(
    data = data,
    source_data = source_data,
    panels = panels,
    figure = figure
  ))
}
