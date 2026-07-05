args <- commandArgs(FALSE)
file_arg <- sub("^--file=", "", args[grepl("^--file=", args)])
script_dir <- if (length(file_arg) > 0) dirname(normalizePath(file_arg[[1]])) else file.path(getwd(), "scripts_R")
source(file.path(script_dir, "_helpers.R"))

dir.create(model_path("outputs", "figures"), recursive = TRUE, showWarnings = FALSE)
dir.create(model_path("outputs", "tables"), recursive = TRUE, showWarnings = FALSE)

required <- c(
  "tidyverse",
  "ggplot2",
  "ggrepel",
  "ggridges",
  "patchwork",
  "scales",
  "viridis",
  "cowplot",
  "RColorBrewer",
  "readr",
  "arrow"
)
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]

if (length(missing) > 0) {
  stop("Missing R packages: ", paste(missing, collapse = ", "), ". Run install_R_packages.R if installation is allowed.")
}

message("R setup complete.")
