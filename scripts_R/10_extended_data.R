args <- commandArgs(FALSE)
file_arg <- sub("^--file=", "", args[grepl("^--file=", args)])
script_dir <- if (length(file_arg) > 0) dirname(normalizePath(file_arg[[1]])) else file.path(getwd(), "scripts_R")
source(file.path(script_dir, "extended_data", "render_extended.R"))
render_extended_data()
# eFigure 3 now uses the curated hand-drawn SVG source
# outputs/appendix/extended_data_figure_3_model_structure.svg.
# The retired ggplot-based model-structure script is retained for provenance only.
# source(file.path(script_dir, "25_extended_data_11_model_structure.R"))
