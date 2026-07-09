## Figure 1 render entry point -------------------------------------------------

if (exists("script_dir", inherits = TRUE) && file.exists(file.path(get("script_dir"), "figures", "figure_1", "data.R"))) {
  script_dir <- normalizePath(get("script_dir"), mustWork = FALSE)
  figure_1_dir <- file.path(script_dir, "figures", "figure_1")
} else {
  figure_1_file <- tryCatch(normalizePath(sys.frame(1)$ofile, mustWork = FALSE), error = function(e) NA_character_)
  if (is.na(figure_1_file)) {
    figure_1_file <- file.path(getwd(), "scripts_R", "figures", "figure_1", "render.R")
  }
  figure_1_dir <- dirname(figure_1_file)
  script_dir <- normalizePath(file.path(figure_1_dir, "..", ".."), mustWork = FALSE)
}
source(normalizePath(file.path(figure_1_dir, "..", "..", "lib", "bootstrap.R"), mustWork = FALSE))
source(file.path(figure_1_dir, "data.R"))
source(file.path(figure_1_dir, "source_data.R"))
source(file.path(figure_1_dir, "panels.R"))
source(file.path(figure_1_dir, "assemble.R"))

render_figure_1 <- function(write_source = TRUE, save_plot = TRUE) {
  data <- prepare_figure_1_data()
  source_data <- make_figure_1_source_data(data)
  if (isTRUE(write_source)) {
    write_figure_1_source_data(source_data)
  }

  panels <- plot_figure_1_panels(data)
  figure <- assemble_figure_1(panels)

  if (isTRUE(save_plot)) {
    save_main_figure(figure, "figure_1_baseline_heterogeneity", height = 6.2)
    cat("Figure 1 saved.\n")
  }

  invisible(list(
    data = data,
    source_data = source_data,
    panels = panels,
    figure = figure
  ))
}
