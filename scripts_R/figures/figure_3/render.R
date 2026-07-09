## Figure 3 render entry point -------------------------------------------------

if (exists("script_dir", inherits = TRUE) && file.exists(file.path(get("script_dir"), "figures", "figure_3", "data.R"))) {
  script_dir <- normalizePath(get("script_dir"), mustWork = FALSE)
  figure_3_dir <- file.path(script_dir, "figures", "figure_3")
} else {
  figure_3_file <- tryCatch(normalizePath(sys.frame(1)$ofile, mustWork = FALSE), error = function(e) NA_character_)
  if (is.na(figure_3_file)) {
    figure_3_file <- file.path(getwd(), "scripts_R", "figures", "figure_3", "render.R")
  }
  figure_3_dir <- dirname(figure_3_file)
  script_dir <- normalizePath(file.path(figure_3_dir, "..", ".."), mustWork = FALSE)
}

source(normalizePath(file.path(figure_3_dir, "..", "..", "lib", "bootstrap.R"), mustWork = FALSE))
source(file.path(figure_3_dir, "data.R"))
source(file.path(figure_3_dir, "source_data.R"))
source(file.path(figure_3_dir, "panels.R"))
source(file.path(figure_3_dir, "assemble.R"))

render_figure_3 <- function(write_source = TRUE, save_plot = TRUE) {
  data <- prepare_figure_3_data()
  source_data <- make_figure_3_source_data(data)
  if (isTRUE(write_source)) {
    write_figure_3_source_data(source_data)
  }

  panels <- plot_figure_3_panels(data)
  figure <- assemble_figure_3(panels)

  if (isTRUE(save_plot)) {
    save_main_figure(figure, "figure_3_endpoint_dependence", height = 5.45)
    cat("Figure 3 saved.\n")
  }

  invisible(list(
    data = data,
    source_data = source_data,
    panels = panels,
    figure = figure
  ))
}
