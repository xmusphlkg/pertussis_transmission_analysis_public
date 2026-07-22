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

write_figure_2_plot_staging_files <- function(
  figure,
  temporary_paths,
  width,
  height,
  dpi
) {
  ggsave(
    temporary_paths[["pdf"]], figure,
    width = width, height = height,
    device = cairo_pdf, bg = "white", limitsize = FALSE
  )
  ggsave(
    temporary_paths[["png"]], figure,
    width = width, height = height,
    dpi = dpi, bg = "white", limitsize = FALSE
  )
  sizes <- file.info(temporary_paths)$size
  if (any(!is.finite(sizes)) || any(sizes <= 0)) {
    stop("Figure 2 temporary export is empty; existing files were preserved.", call. = FALSE)
  }

  invisible(temporary_paths)
}

validate_figure_2_source_staging_files <- function(
  source_data,
  temporary_paths
) {
  sizes <- file.info(temporary_paths)$size
  if (any(!is.finite(sizes)) || any(sizes <= 0)) {
    stop(
      "Figure 2 temporary source-data export is empty; existing files were preserved.",
      call. = FALSE
    )
  }
  for (name in names(source_data)) {
    observed <- readr::read_csv(
      temporary_paths[[name]],
      show_col_types = FALSE,
      progress = FALSE
    )
    expected <- source_data[[name]]
    if (nrow(observed) != nrow(expected) || !identical(names(observed), names(expected))) {
      stop(
        "Figure 2 temporary source-data validation failed for ", name, ".",
        call. = FALSE
      )
    }
  }
  invisible(temporary_paths)
}

replace_figure_2_outputs_transactionally <- function(
  temporary_paths,
  final_paths,
  token
) {
  if (!identical(names(temporary_paths), names(final_paths)) ||
      any(!file.exists(temporary_paths))) {
    stop("Figure 2 replacement received an incomplete staged output set.", call. = FALSE)
  }
  Sys.chmod(temporary_paths, mode = "0664", use_umask = FALSE)
  backup_paths <- paste0(final_paths, ".backup", token)
  had_original <- file.exists(final_paths)
  moved_original <- rep(FALSE, length(final_paths))
  installed <- rep(FALSE, length(final_paths))
  committed <- FALSE

  on.exit({
    if (!committed) {
      unlink(final_paths[installed & file.exists(final_paths)])
      for (index in which(moved_original)) {
        if (file.exists(backup_paths[[index]])) {
          file.rename(backup_paths[[index]], final_paths[[index]])
        }
      }
    }
    unlink(temporary_paths[file.exists(temporary_paths)])
    if (committed) {
      unlink(backup_paths[file.exists(backup_paths)])
    }
  }, add = TRUE)

  for (index in which(had_original)) {
    moved_original[[index]] <- file.rename(
      final_paths[[index]],
      backup_paths[[index]]
    )
    if (!moved_original[[index]]) {
      stop(
        "Figure 2 could not stage an existing output for replacement; rollback started.",
        call. = FALSE
      )
    }
  }
  for (index in seq_along(final_paths)) {
    installed[[index]] <- file.rename(
      temporary_paths[[index]],
      final_paths[[index]]
    )
    if (!installed[[index]]) {
      stop(
        "Figure 2 output replacement failed; rollback started.",
        call. = FALSE
      )
    }
  }
  committed <- TRUE
  invisible(final_paths)
}

export_figure_2_transactionally <- function(
  figure,
  source_data,
  filename,
  width,
  height,
  dpi,
  write_source,
  save_plot
) {
  token <- paste0(".tmp-", Sys.getpid(), "-", format(Sys.time(), "%Y%m%d%H%M%OS6"))
  token <- gsub("[^A-Za-z0-9.-]", "-", token)
  final_paths <- character()
  temporary_paths <- character()
  on.exit(
    unlink(temporary_paths[file.exists(temporary_paths)]),
    add = TRUE
  )

  if (isTRUE(write_source)) {
    source_final <- file.path(
      model_path("outputs", "tables"),
      unname(figure_2_source_filenames)
    )
    names(source_final) <- names(figure_2_source_filenames)
    source_temporary <- file.path(
      dirname(source_final),
      paste0(tools::file_path_sans_ext(basename(source_final)), token, ".csv")
    )
    names(source_temporary) <- names(source_final)
    final_paths <- c(final_paths, source_final)
    temporary_paths <- c(temporary_paths, source_temporary)
    write_figure_2_source_data(
      source_data,
      output_dir = dirname(source_temporary[[1]]),
      filenames = stats::setNames(basename(source_temporary), names(source_temporary))
    )
    validate_figure_2_source_staging_files(source_data, source_temporary)
  }

  if (isTRUE(save_plot)) {
    plot_final <- file.path(figures_dir, paste0(filename, c(".pdf", ".png")))
    names(plot_final) <- c("plot_pdf", "plot_png")
    plot_temporary <- file.path(
      figures_dir,
      paste0(filename, token, c(".pdf", ".png"))
    )
    names(plot_temporary) <- names(plot_final)
    final_paths <- c(final_paths, plot_final)
    temporary_paths <- c(temporary_paths, plot_temporary)
    write_figure_2_plot_staging_files(
      figure,
      stats::setNames(unname(plot_temporary), c("pdf", "png")),
      width = width,
      height = height,
      dpi = dpi
    )
  }

  if (length(final_paths) > 0L) {
    replace_figure_2_outputs_transactionally(
      temporary_paths,
      final_paths,
      token
    )
  }

  invisible(final_paths)
}

render_figure_2 <- function(write_source = TRUE, save_plot = TRUE) {
  data <- prepare_figure_2_data()
  source_data <- make_figure_2_source_data(data)
  panels <- plot_figure_2_panels(data)
  figure <- assemble_figure_2(panels)

  if (isTRUE(save_plot)) {
  }

  if (isTRUE(write_source) || isTRUE(save_plot)) {
    export_figure_2_transactionally(
      figure = figure,
      source_data = source_data,
      filename = "figure_2_country_strategy_prioritization",
      width = lancet_double_width,
      height = 6.45,
      dpi = lancet_submission_dpi,
      write_source = write_source,
      save_plot = save_plot
    )
  }
  if (isTRUE(save_plot)) {
    cat("Figure 2 PDF and PNG preview saved.\n")
  }

  invisible(list(
    data = data,
    source_data = source_data,
    panels = panels,
    figure = figure
  ))
}
