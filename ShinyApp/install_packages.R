packages <- c(
  "shiny",
  "bslib",
  "deSolve",
  "yaml",
  "readr",
  "dplyr",
  "tidyr",
  "purrr",
  "ggplot2",
  "plotly",
  "DT",
  "scales"
)

repos <- "https://cloud.r-project.org"
missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) {
  install.packages(missing, repos = repos)
}

message("ShinyApp package check complete.")
