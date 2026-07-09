args <- commandArgs(FALSE)
file_arg <- sub("^--file=", "", args[grepl("^--file=", args)])
script_dir <- if (length(file_arg) > 0) dirname(normalizePath(file_arg[[1]])) else file.path(getwd(), "scripts_R")
source(file.path(script_dir, "figures", "render_main.R"))
render_main_figures()
# Figure 5 is an exploratory diagnostic, not part of the current submission set.
# source(file.path(script_dir, "15_figure_5_future_vaccine_mechanisms.R"))
source(file.path(script_dir, "10_extended_data.R"))
