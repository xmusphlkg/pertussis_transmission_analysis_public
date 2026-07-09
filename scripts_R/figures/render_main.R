## Main figure render registry -------------------------------------------------

if (exists("script_dir", inherits = TRUE) && file.exists(file.path(get("script_dir"), "figures", "render_main.R"))) {
  script_dir <- normalizePath(get("script_dir"), mustWork = FALSE)
} else {
  main_registry_file <- tryCatch(normalizePath(sys.frame(1)$ofile, mustWork = FALSE), error = function(e) NA_character_)
  if (is.na(main_registry_file)) {
    main_registry_file <- file.path(getwd(), "scripts_R", "figures", "render_main.R")
  }
  script_dir <- normalizePath(file.path(dirname(main_registry_file), ".."), mustWork = FALSE)
}

main_figure_renderers <- c(
  figure_1 = "render_figure_1",
  figure_2 = "render_figure_2",
  figure_3 = "render_figure_3",
  figure_4 = "render_figure_4"
)

source_main_figure_renderers <- function(figures = names(main_figure_renderers)) {
  missing_figures <- setdiff(figures, names(main_figure_renderers))
  if (length(missing_figures) > 0) {
    stop("Unknown main figure(s): ", paste(missing_figures, collapse = ", "), call. = FALSE)
  }

  purrr::walk(figures, function(figure_id) {
    source(file.path(script_dir, "figures", figure_id, "render.R"))
  })

  invisible(figures)
}

render_main_figures <- function(figures = names(main_figure_renderers),
                                write_source = TRUE,
                                save_plot = TRUE) {
  source_main_figure_renderers(figures)

  purrr::iwalk(main_figure_renderers[figures], function(renderer_name, figure_id) {
    renderer <- get(renderer_name, mode = "function")
    renderer(write_source = write_source, save_plot = save_plot)
  })

  invisible(figures)
}
