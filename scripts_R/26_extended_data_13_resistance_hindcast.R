#!/usr/bin/env Rscript
## Extended Data Figure 8: Resistance Hindcast Validation ----------------------
## Layout: (A) China hindcast | (B) Japan hindcast
##         (C) Australia hindcast | (D) Scoring summary

args <- commandArgs(FALSE)
file_arg <- sub("^--file=", "", args[grepl("^--file=", args)])
script_dir <- if (length(file_arg) > 0) dirname(normalizePath(file_arg[[1]])) else file.path(getwd(), "scripts_R")
source(file.path(script_dir, "10_shared.R"))

suppressPackageStartupMessages({
  library(ggrepel)
})

## Load hindcast results ----------------------
## Section: data loading.
## Purpose: Load hindcast results.

hindcast_csv <- model_path("outputs", "tables", "resistance_hindcast_results.csv")
scores_csv <- model_path("outputs", "tables", "resistance_hindcast_scores.csv")

if (!file.exists(hindcast_csv)) {
  stop(
    "eFigure 8 requires outputs/tables/resistance_hindcast_results.csv. ",
    "Run python -m src_python.simulation.run_resistance_hindcast before plotting."
  )
}
if (!file.exists(scores_csv)) {
  stop(
    "eFigure 8 requires outputs/tables/resistance_hindcast_scores.csv. ",
    "Run python -m src_python.simulation.run_resistance_hindcast before plotting."
  )
}

hindcast <- readr::read_csv(hindcast_csv, show_col_types = FALSE)

# Hindcast fitness colour palette, matching the six values in the current CSV.
fitness_colours <- c(
  "0.85" = manuscript_colour("blue"),
  "0.90" = manuscript_colour("sky"),
  "0.95" = manuscript_colour("cream"),
  "1.00" = manuscript_colour("grey"),
  "1.05" = manuscript_colour("sand"),
  "1.10" = manuscript_colour("vermillion")
)

calendar_year_breaks <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) return(numeric())
  span <- diff(range(x))
  by <- if (span <= 3) 1 else 2
  seq(floor(min(x)), ceiling(max(x)), by = by)
}

## Prepare data ----------------------
## Section: data preparation.
## Purpose: Prepare analysis-ready data objects.

hindcast <- hindcast %>%
  mutate(fitness_label = sprintf("%.2f", fitness_R))

# Observed data points (from the hindcast results where observed_fraction is not NA)
observed_data <- hindcast %>%
  filter(!is.na(observed_fraction)) %>%
  select(country, calendar_year, observed_fraction, observed_lower, observed_upper) %>%
  distinct() %>%
  rename(lower = observed_lower, upper = observed_upper)

## Helper function for country hindcast panel ----------------------
## Section: helper functions.
## Purpose: Helper function for country hindcast panel.

plot_country_hindcast <- function(data, obs, country_name, x_label = "Calendar year") {
  country_data <- data %>% filter(country == country_name)
  country_obs <- obs %>% filter(country == country_name)

  if (nrow(country_data) == 0) {
    return(
      ggplot() +
        annotate("text", x = 0, y = 0, label = paste(country_name, "- no data"), size = 2.4) +
        theme_void()
    )
  }

  # Summarise across time for each fitness value
  country_ts <- country_data %>%
    group_by(fitness_label, calendar_year) %>%
    summarise(
      resistant_fraction = mean(model_resistant_fraction, na.rm = TRUE),
      .groups = "drop"
    )

  ggplot(country_ts, aes(calendar_year, resistant_fraction,
                         colour = fitness_label, group = fitness_label)) +
    geom_line(linewidth = 0.5, alpha = 0.8) +
    # Highlight neutral fitness
    geom_line(
      data = country_ts %>% filter(fitness_label == "1.00"),
      linewidth = 0.9, colour = manuscript_colour("grey")
    ) +
    # Observed data points
    geom_pointrange(
      data = country_obs,
      aes(x = calendar_year, y = observed_fraction,
          ymin = lower, ymax = upper),
      colour = manuscript_colour("vermillion"), size = 0.8, linewidth = 0.5,
      shape = 18, inherit.aes = FALSE
    ) +
    scale_x_continuous(
      breaks = calendar_year_breaks,
      labels = function(x) as.character(as.integer(x))
    ) +
    scale_y_continuous(
      labels = scales::percent_format(accuracy = 1),
      limits = c(0, 1.02)
    ) +
    scale_colour_manual(values = fitness_colours, drop = FALSE) +
    labs(
      x = x_label,
      y = "Resistant fraction",
      colour = expression(italic(f)[R])
    ) +
    theme_lancet() +
    theme(
      legend.position = "right",
      legend.key.size = unit(0.25, "cm")
    )
}

## Panel A: China hindcast ----------------------
## Section: Panel A.
## Purpose: China hindcast.

p8a <- plot_country_hindcast(hindcast, observed_data, "China",
                              x_label = "Calendar year")

## Panel B: Japan hindcast ----------------------
## Section: Panel B.
## Purpose: Japan hindcast.

p8b <- plot_country_hindcast(hindcast, observed_data, "Japan",
                              x_label = "Calendar year")

## Panel C: Australia hindcast ----------------------
## Section: Panel C.
## Purpose: Australia hindcast.

p8c <- plot_country_hindcast(hindcast, observed_data, "Australia",
                              x_label = "Calendar year")

## Panel D: Scoring summary ----------------------
## Section: Panel D.
## Purpose: Scoring summary.

scores <- readr::read_csv(scores_csv, show_col_types = FALSE) %>%
  mutate(
    fitness_label = sprintf("%.2f", fitness_R),
    country_label = factor(country, levels = c("China", "Japan", "Australia"))
  )

# Identify best fitness per country
best_fitness <- scores %>%
  group_by(country) %>%
  filter(mean_absolute_error == min(mean_absolute_error, na.rm = TRUE)) %>%
  ungroup()

p8d <- ggplot(scores, aes(fitness_R, mean_absolute_error, colour = country_label)) +
  geom_line(linewidth = 0.5, alpha = 0.7) +
  geom_point(size = 1.5, alpha = 0.8) +
  # Highlight best-fitting fitness
  geom_point(
    data = best_fitness,
    aes(fitness_R, mean_absolute_error),
    shape = 18, size = 3.5, colour = "black"
  ) +
  geom_text_repel(
    data = best_fitness,
    aes(fitness_R, mean_absolute_error,
        label = paste0(country, "\n(f_R=", fitness_label, ")")),
    size = 2.0, colour = "black",
    nudge_y = 0.02, segment.size = 0.2
  ) +
  scale_colour_manual(values = manuscript_country_focus_colours[c("China", "Japan", "Australia")]) +
  scale_x_continuous(breaks = seq(0.85, 1.10, 0.05)) +
  labs(
    x = expression(italic(f)[R]),
    y = "Mean absolute error\n(modelled vs observed)",
    colour = NULL
  ) +
  theme_lancet() +
  theme(legend.position = "bottom")

## Compose eFigure 8 ----------------------
## Section: figure assembly.
## Purpose: Compose eFigure 8.

extended8 <- p8a + p8b + p8c + p8d +
  plot_layout(ncol = 2, guides = "collect") +
  plot_annotation(tag_levels = "A") &
  (theme(
    legend.position = "bottom",
    legend.box = "horizontal",
    plot.margin = margin(3, 3, 3, 3)
  ) + theme_lancet_tags())

save_appendix_figure(extended8, "extended_data_figure_8_resistance_hindcast", height = 6.4)
cat("eFigure 8 (resistance hindcast plausibility checks) saved.\n")
