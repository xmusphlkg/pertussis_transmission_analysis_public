args <- commandArgs(FALSE)
file_arg <- sub("^--file=", "", args[grepl("^--file=", args)])
script_dir <- if (length(file_arg) > 0) dirname(normalizePath(file_arg[[1]])) else file.path(getwd(), "scripts_R")
source(file.path(script_dir, "15_extended_data_1_country_inputs.R"))
source(file.path(script_dir, "16_extended_data_2_diagnostics_sensitivity.R"))
# eFigure 3 now uses the curated hand-drawn SVG source
# outputs/appendix/extended_data_figure_3_model_structure.svg.
# The retired ggplot-based model-structure script is retained for provenance only.
# source(file.path(script_dir, "25_extended_data_11_model_structure.R"))
source(file.path(script_dir, "20_extended_data_6_baseline_dynamics.R"))
source(file.path(script_dir, "21_extended_data_7_vaccine_deep_dive.R"))
source(file.path(script_dir, "22_extended_data_8_resistance_dynamics.R"))
source(file.path(script_dir, "24_extended_data_10_intervention_extended.R"))
source(file.path(script_dir, "26_extended_data_13_resistance_hindcast.R"))
source(file.path(script_dir, "27_extended_data_9_scenario_robustness.R"))
source(file.path(script_dir, "28_extended_data_10_resistance_management_policy.R"))
source(file.path(script_dir, "29_extended_data_11_implementation_structural_robustness.R"))
source(file.path(script_dir, "23_extended_data_9_full_grid.R"))
