suppressPackageStartupMessages({
  library(tidyverse)
  library(ggplot2)
  library(patchwork)
  library(scales)
  library(viridis)
})

if (exists("script_dir", inherits = TRUE) && file.exists(file.path(get("script_dir"), "_helpers.R"))) {
  helper_file <- normalizePath(file.path(get("script_dir"), "_helpers.R"), mustWork = FALSE)
} else {
  helper_file <- tryCatch(normalizePath(sys.frame(1)$ofile), error = function(e) NA_character_)
}
if (is.na(helper_file)) {
  helper_file <- file.path(getwd(), "scripts_R", "_helpers.R")
}
root_dir <- normalizePath(file.path(dirname(helper_file), ".."), mustWork = FALSE)

model_path <- function(...) {
  file.path(root_dir, ...)
}

if (!exists(".pertussis_model_table_cache", envir = .GlobalEnv, inherits = FALSE)) {
  assign(".pertussis_model_table_cache", new.env(parent = emptyenv()), envir = .GlobalEnv)
}

read_model_table <- function(path_without_suffix) {
  parquet_path <- paste0(path_without_suffix, ".parquet")
  csv_path <- paste0(path_without_suffix, ".csv")
  stem <- basename(path_without_suffix)
  stem <- sub("_summary$", "", stem)
  metadata_path <- model_path("outputs", "metadata", paste0(stem, "_run_metadata.json"))
  if (!file.exists(metadata_path)) {
    stop("Missing run metadata for ", stem, ". Regenerate outputs with the current Python pipeline.")
  }
  cache_key <- normalizePath(path_without_suffix, mustWork = FALSE)
  table_cache <- get(".pertussis_model_table_cache", envir = .GlobalEnv)
  if (exists(cache_key, envir = table_cache, inherits = FALSE)) {
    return(get(cache_key, envir = table_cache, inherits = FALSE))
  }
  if (file.exists(csv_path)) {
    out <- readr::read_csv(csv_path, show_col_types = FALSE)
    assign(cache_key, out, envir = table_cache)
    return(out)
  }
  if (requireNamespace("arrow", quietly = TRUE) && file.exists(parquet_path)) {
    out <- arrow::read_parquet(parquet_path)
    assign(cache_key, out, envir = table_cache)
    return(out)
  }
  stop("Could not find either ", parquet_path, " or ", csv_path)
}

read_model_table_optional <- function(path_without_suffix) {
  parquet_path <- paste0(path_without_suffix, ".parquet")
  csv_path <- paste0(path_without_suffix, ".csv")
  stem <- basename(path_without_suffix)
  stem <- sub("_summary$", "", stem)
  metadata_path <- model_path("outputs", "metadata", paste0(stem, "_run_metadata.json"))
  if (!file.exists(metadata_path) || (!file.exists(csv_path) && !file.exists(parquet_path))) {
    return(tibble())
  }
  read_model_table(path_without_suffix)
}

save_figure <- function(plot, filename, width = 9, height = 6) {
  dir.create(model_path("outputs", "figures"), recursive = TRUE, showWarnings = FALSE)
  pdf_path <- model_path("outputs", "figures", paste0(filename, ".pdf"))
  png_path <- model_path("outputs", "figures", paste0(filename, ".png"))
  ggsave(pdf_path, plot, width = width, height = height, device = cairo_pdf)
  ggsave(png_path, plot, width = width, height = height, dpi = 220)
}

infer_dt <- function(df) {
  times <- sort(unique(df$time))
  if (length(times) < 2) {
    return(1)
  }
  stats::median(diff(times))
}

theme_manuscript <- function() {
  theme_minimal(base_size = 11) +
    theme(
      panel.grid.minor = element_blank(),
      plot.title.position = "plot",
      plot.title = element_text(face = "bold"),
      legend.position = "bottom",
      strip.text = element_text(face = "bold")
    )
}
