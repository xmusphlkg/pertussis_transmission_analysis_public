## Extended Data render registry ----------------------------------------------

if (exists("script_dir", inherits = TRUE) && file.exists(file.path(get("script_dir"), "extended_data", "render_extended.R"))) {
  script_dir <- normalizePath(get("script_dir"), mustWork = FALSE)
} else {
  extended_registry_file <- tryCatch(normalizePath(sys.frame(1)$ofile, mustWork = FALSE), error = function(e) NA_character_)
  if (is.na(extended_registry_file)) {
    extended_registry_file <- file.path(getwd(), "scripts_R", "extended_data", "render_extended.R")
  }
  script_dir <- normalizePath(file.path(dirname(extended_registry_file), ".."), mustWork = FALSE)
}

extended_data_renderers <- c(
  extended_data_figure_1 = "render_extended_data_figure_1",
  extended_data_figure_2 = "render_extended_data_figure_2",
  extended_data_figure_4 = "render_extended_data_figure_4",
  extended_data_figure_5 = "render_extended_data_figure_5",
  extended_data_figure_6 = "render_extended_data_figure_6",
  extended_data_figure_7 = "render_extended_data_figure_7",
  extended_data_figure_8 = "render_extended_data_figure_8",
  extended_data_figure_9 = "render_extended_data_figure_9",
  extended_data_figure_10 = "render_extended_data_figure_10",
  extended_data_figure_11 = "render_extended_data_figure_11",
  extended_data_figure_12 = "render_extended_data_figure_12"
)

legacy_extended_data_scripts <- character()

source_extended_data_renderers <- function(figures = names(extended_data_renderers)) {
  missing_figures <- setdiff(figures, names(extended_data_renderers))
  if (length(missing_figures) > 0) {
    stop("Unknown Extended Data figure(s): ", paste(missing_figures, collapse = ", "), call. = FALSE)
  }

  purrr::walk(figures, function(figure_id) {
    module_id <- sub("^extended_data_", "", figure_id)
    source(file.path(script_dir, "extended_data", module_id, "render.R"))
  })

  invisible(figures)
}

render_modular_extended_data <- function(figures = names(extended_data_renderers),
                                         write_source = TRUE,
                                         save_plot = TRUE) {
  source_extended_data_renderers(figures)

  purrr::iwalk(extended_data_renderers[figures], function(renderer_name, figure_id) {
    renderer <- get(renderer_name, mode = "function")
    renderer(write_source = write_source, save_plot = save_plot)
  })

  invisible(figures)
}

render_legacy_extended_data <- function(scripts = legacy_extended_data_scripts) {
  purrr::walk(scripts, function(script) {
    source(file.path(script_dir, script))
  })

  invisible(scripts)
}

render_extended_data <- function(figures = names(extended_data_renderers),
                                 write_source = TRUE,
                                 save_plot = TRUE,
                                 include_legacy = TRUE) {
  render_modular_extended_data(figures, write_source = write_source, save_plot = save_plot)
  if (isTRUE(include_legacy)) {
    render_legacy_extended_data()
  }

  invisible(figures)
}
