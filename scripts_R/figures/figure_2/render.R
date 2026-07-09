## Figure 2 render entry point -------------------------------------------------

if (exists("script_dir", inherits = TRUE) && file.exists(file.path(get("script_dir"), "figures", "figure_2", "data.R"))) {
  script_dir <- normalizePath(get("script_dir"), mustWork = FALSE)
  figure_2_dir <- file.path(script_dir, "figures", "figure_2")
} else {
  figure_2_file <- tryCatch(normalizePath(sys.frame(1)$ofile, mustWork = FALSE), error = function(e) NA_character_)
  if (is.na(figure_2_file)) {
    figure_2_file <- file.path(getwd(), "scripts_R", "figures", "figure_2", "render.R")
  }
  figure_2_dir <- dirname(figure_2_file)
  script_dir <- normalizePath(file.path(figure_2_dir, "..", ".."), mustWork = FALSE)
}

source(normalizePath(file.path(figure_2_dir, "..", "..", "lib", "bootstrap.R"), mustWork = FALSE))
source(file.path(figure_2_dir, "data.R"))
source(file.path(figure_2_dir, "source_data.R"))
source(file.path(figure_2_dir, "panels.R"))
source(file.path(figure_2_dir, "assemble.R"))

render_figure_2 <- function(write_source = TRUE, save_plot = TRUE) {
  data <- prepare_figure_2_data()
  source_data <- make_figure_2_source_data(data)
  if (isTRUE(write_source)) {
    write_figure_2_source_data(source_data)
  }

  panels <- plot_figure_2_panels(data)
  figure <- assemble_figure_2(panels)

  if (isTRUE(save_plot)) {
    save_main_figure(figure, "figure_2_country_strategy_prioritization", height = 6.0)
    cat("Figure 2 saved.\n")
  }

  invisible(list(
    data = data,
    source_data = source_data,
    panels = panels,
    figure = figure
  ))
}
