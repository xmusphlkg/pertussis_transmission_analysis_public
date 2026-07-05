# Retired figure script: not part of the current publication pipeline.
# Set PERTUSSIS_ALLOW_RETIRED_R_FIGURES=1 only to regenerate this provenance-only figure.
if (Sys.getenv("PERTUSSIS_ALLOW_RETIRED_R_FIGURES") != "1") {
  stop(
    "Retired figure script: not part of the current publication pipeline. ",
    "Run scripts_R/10_figure_set.R for current figures, or set ",
    "PERTUSSIS_ALLOW_RETIRED_R_FIGURES=1 to reproduce this provenance-only figure."
  )
}

args <- commandArgs(FALSE)
file_arg <- sub("^--file=", "", args[grepl("^--file=", args)])
script_dir <- if (length(file_arg) > 0) dirname(normalizePath(file_arg[[1]])) else file.path(getwd(), "scripts_R")
source(file.path(script_dir, "10_shared.R"))

## Extended Data Figure 5: model architecture and state-space accounting. ----------------------
## Section: extended-data figure setup.
## Purpose: model architecture and state-space accounting.

architecture_counts <- tibble(
  component = c("Age groups", "Strains", "Immune/dose origins", "Compartments per age", "ODE state variables"),
  count = c(8, 2, 8, 74, 592)
) %>%
  mutate(component = factor(component, levels = component))

p_ed5a <- architecture_counts %>%
  ggplot(aes(count, component, fill = component)) +
  geom_col(width = 0.66, show.legend = FALSE) +
  geom_text(aes(label = count), hjust = -0.14, size = 2.2) +
  scale_x_continuous(labels = label_number(accuracy = 1), expand = expansion(mult = c(0, 0.18))) +
  scale_fill_manual(values = manuscript_architecture_colours) +
  labs(x = "Count", y = NULL) +
  theme_lancet()

compartment_blocks <- tibble(
  block = c("Susceptible origins", "Exposed states", "Infectious states", "Treated states", "Natural/waned immunity"),
  count = c(8, 16, 32, 16, 2),
  detail = c(
    "8 origins",
    "2 strains x 8 origins",
    "2 strains x 2 symptom states x 8 origins",
    "2 strains x 8 origins",
    "R and W states"
  )
) %>%
  mutate(block = factor(block, levels = block))

p_ed5b <- compartment_blocks %>%
  ggplot(aes(block, count, fill = block)) +
  geom_col(width = 0.66, show.legend = FALSE) +
  geom_text(aes(label = count), vjust = -0.25, size = 2.2) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.16))) +
  scale_fill_manual(values = manuscript_compartment_colours) +
  labs(x = NULL, y = "Compartments per age group") +
  theme_lancet() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

effect_matrix <- expand_grid(
  vaccine_effect = c("VE_sus", "VE_sym", "VE_inf", "VE_dur"),
  model_route = c("Susceptibility", "Symptomatic disease", "Onward infectiousness", "Infectious duration")
) %>%
  mutate(active = case_when(
    vaccine_effect == "VE_sus" & model_route == "Susceptibility" ~ TRUE,
    vaccine_effect == "VE_sym" & model_route == "Symptomatic disease" ~ TRUE,
    vaccine_effect == "VE_inf" & model_route == "Onward infectiousness" ~ TRUE,
    vaccine_effect == "VE_dur" & model_route == "Infectious duration" ~ TRUE,
    TRUE ~ FALSE
  ))

p_ed5c <- effect_matrix %>%
  ggplot(aes(model_route, vaccine_effect, fill = active)) +
  geom_tile(colour = "white", linewidth = lancet_heatmap_tile_linewidth) +
  geom_text(aes(label = if_else(active, "direct", "")), size = 2) +
  scale_fill_manual(values = manuscript_route_matrix_colours, guide = "none") +
  labs(x = NULL, y = NULL) +
  theme_lancet_heatmap(x_angle = 30, x_hjust = 1)

origin_effects <- tibble(
  origin = c(
    "Unvaccinated", "Maternal", "Dose 1 recent", "Dose 1 waned",
    "Dose 2 recent", "Dose 2 waned", "Dose 3+ recent", "Dose 3+ waned"
  ),
  relative_effect = c(0.00, 0.75, 0.45, 0.45 * 0.35, 0.75, 0.75 * 0.35, 1.00, 0.35)
) %>%
  mutate(origin = factor(origin, levels = origin))

p_ed5d <- origin_effects %>%
  ggplot(aes(relative_effect, origin, fill = relative_effect)) +
  geom_col(width = 0.66) +
  geom_text(aes(label = percent(relative_effect, accuracy = 1)), hjust = -0.12, size = 2) +
  scale_x_continuous(labels = percent_format(accuracy = 1), expand = expansion(mult = c(0, 0.16))) +
  scale_fill_fraction(guide = "none") +
  labs(x = "Relative vaccine-effect weight", y = NULL) +
  theme_lancet()

extended5 <- ((p_ed5a | p_ed5b) / (p_ed5c | p_ed5d)) +
  plot_layout(guides = "keep") +
  plot_annotation(tag_levels = "A") &
  (theme(plot.margin = margin(4, 4, 4, 4)) + theme_lancet_tags())

save_appendix_figure(extended5, "extended_data_figure_5_model_architecture", height = 8.0)
