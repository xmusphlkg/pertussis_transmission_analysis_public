packages <- c(
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
  "arrow",
  "yaml",
  "contactdata"
)

repos <- "https://cloud.r-project.org"
current_r <- as.character(getRversion())
installed <- installed.packages()
installed_built <- installed[, "Built"]
installed_names <- rownames(installed)
installed_idx <- match(packages, installed_names)
present <- !is.na(installed_idx)

missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
built_mismatch <- intersect(
  packages,
  packages[present & installed_built[installed_idx[present]] != current_r]
)
targets <- unique(c(missing, built_mismatch))

if (length(targets) > 0) {
  install.packages(targets, repos = repos)
}

message("R package check complete.")
