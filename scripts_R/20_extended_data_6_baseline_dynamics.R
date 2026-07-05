args <- commandArgs(FALSE)
file_arg <- sub("^--file=", "", args[grepl("^--file=", args)])
script_dir <- if (length(file_arg) > 0) dirname(normalizePath(file_arg[[1]])) else file.path(getwd(), "scripts_R")
source(file.path(script_dir, "10_shared.R"))

## Extended Data Figure 4: baseline temporal dynamics. ----------------------
## Section: extended-data figure setup.
## Purpose: baseline temporal dynamics.

country_sim <- read_model_table_optional(model_path("outputs", "simulations", "country_scenarios"))
require_efigure_timeseries(country_sim, "eFigure 4 baseline temporal dynamics", "country_scenarios")

if (nrow(country_sim) > 0) {
  country_ts <- country_sim %>%
    add_country_label() %>%
    left_join(baseline %>% select(country, infant_population), by = "country") %>%
    mutate(
      country_code = factor(country_codes[country], levels = country_codes[country_levels]),
      age_group = factor(age_group, levels = names(age_labels), labels = age_labels),
      strain_label = factor(str_to_title(strain), levels = c("Sensitive", "Resistant"))
    )

  weekly_country <- country_ts %>%
    group_by(country, country_label, country_code, time) %>%
    summarise(
      simulation_year = first(time) / 365,
      total_population = max(total_population, na.rm = TRUE),
      infant_population = max(infant_population, na.rm = TRUE),
      total_infection_rate = sum(total_infection_rate_per_day, na.rm = TRUE),
      infant_case_rate = sum(infant_case_rate_per_day, na.rm = TRUE),
      resistant_infection_rate = sum(if_else(strain == "resistant", total_infection_rate_per_day, 0), na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      infection_incidence = total_infection_rate / pmax(total_population, 1e-9) * 365 * 1e5,
      infant_case_incidence = infant_case_rate / pmax(infant_population, 1e-9) * 365 * 1e5,
      resistant_fraction = resistant_infection_rate / pmax(total_infection_rate, 1e-12)
    )

  p_ed6a <- weekly_country %>%
    ggplot(aes(simulation_year, infection_incidence)) +
    geom_line(linewidth = 0.25, colour = manuscript_metric_colours["All infections"]) +
    facet_wrap(~country_code, scales = "free_y", nrow = 1) +
    scale_x_continuous(breaks = seq(0, 30, by = 10)) +
    scale_y_continuous(labels = label_number(accuracy = 1)) +
    labs(x = "Simulation year", y = "All infections\nper 100,000/year") +
    theme_lancet() +
    theme(axis.text.x = element_text(size = 5.4))

  p_ed6b <- weekly_country %>%
    ggplot(aes(simulation_year, infant_case_incidence)) +
    geom_line(linewidth = 0.25, colour = manuscript_metric_colours["Infant cases"]) +
    facet_wrap(~country_code, scales = "free_y", nrow = 1) +
    scale_x_continuous(breaks = seq(0, 30, by = 10)) +
    scale_y_continuous(labels = label_number(accuracy = 1)) +
    labs(x = "Simulation year", y = "Infant cases\nper 100,000 infants/year") +
    theme_lancet() +
    theme(axis.text.x = element_text(size = 5.4))

  p_ed6c <- weekly_country %>%
    ggplot(aes(simulation_year, resistant_fraction)) +
    geom_line(linewidth = 0.25, colour = manuscript_metric_colours["Resistant"]) +
    facet_wrap(~country_code, ncol = 4) +
    scale_x_continuous(breaks = seq(0, 30, by = 10)) +
    scale_y_continuous(labels = percent_format(accuracy = 1)) +
    coord_cartesian(ylim = c(0, 1)) +
    labs(x = "Simulation year", y = "Resistant infection\nfraction") +
    theme_lancet()

  age_strain_contribution <- country_ts %>%
    group_by(country_label, age_group, strain_label) %>%
    summarise(infections = sum(total_infections, na.rm = TRUE), .groups = "drop") %>%
    group_by(country_label) %>%
    mutate(country_share = infections / pmax(sum(infections, na.rm = TRUE), 1e-9)) %>%
    ungroup()

  p_ed6d <- age_strain_contribution %>%
    ggplot(aes(age_group, country_label, fill = country_share)) +
    geom_tile(colour = "white", linewidth = lancet_heatmap_tile_linewidth) +
    facet_wrap(~strain_label, nrow = 1) +
    scale_fill_infant_burden(
      labels = label_lancet_percent(accuracy = 1),
      guide = guide_colourbar(
        barwidth = grid::unit(40, "mm"),
        barheight = grid::unit(3, "mm"),
        title.position = "top"
      )
    ) +
    labs(x = "Age group", y = NULL, fill = "Share of all infections") +
    theme_lancet_heatmap(x_angle = 90, x_hjust = 1, x_vjust = 0.5)
} else {
  baseline_snapshot <- baseline %>%
    mutate(country_code = factor(country_codes[country], levels = country_codes[country_levels]))

  p_ed6a <- baseline_snapshot %>%
    ggplot(aes(annualized_infections_per_100k, country_burden_order)) +
    geom_point(size = 1.8, colour = manuscript_metric_colours["All infections"]) +
    scale_x_log10(labels = label_number(accuracy = 1)) +
    labs(x = "All infections per\n100,000/year (log)", y = NULL) +
    theme_lancet()

  p_ed6b <- baseline_snapshot %>%
    ggplot(aes(annualized_infant_cases_per_100k, country_burden_order)) +
    geom_point(size = 1.8, colour = manuscript_metric_colours["Infant cases"]) +
    scale_x_log10(labels = label_number(accuracy = 1)) +
    labs(x = "Infant cases per\n100,000 infants/year (log)", y = NULL) +
    theme_lancet()

  p_ed6c <- baseline_snapshot %>%
    ggplot(aes(y = country_burden_order)) +
    geom_segment(aes(x = resistant_fraction_start, xend = resistant_fraction_end, yend = country_burden_order),
                 linewidth = 0.45, colour = manuscript_metric_colours["Resistant"], alpha = 0.75) +
    geom_point(aes(x = resistant_fraction_start), shape = 21, fill = "white", colour = manuscript_metric_colours["Resistant"], size = 1.5, stroke = 0.3) +
    geom_point(aes(x = resistant_fraction_end), colour = manuscript_metric_colours["Resistant"], size = 1.8) +
    scale_x_continuous(labels = percent_format(accuracy = 1)) +
    coord_cartesian(xlim = c(0, 1)) +
    labs(x = "Resistant fraction (start to end)", y = NULL) +
    theme_lancet()

  origin_share_fallback <- baseline_snapshot %>%
    select(country_label, maternal_origin_infection_share, dose1_origin_infection_share,
           dose2_origin_infection_share, dose3plus_origin_infection_share, waned_origin_infection_share) %>%
    pivot_longer(-country_label, names_to = "origin", values_to = "share") %>%
    mutate(origin = factor(origin,
                           levels = c("maternal_origin_infection_share", "dose1_origin_infection_share",
                                      "dose2_origin_infection_share", "dose3plus_origin_infection_share",
                                      "waned_origin_infection_share"),
                           labels = c("Maternal", "Dose 1", "Dose 2", "Dose 3+", "Waned")))

  p_ed6d <- origin_share_fallback %>%
    ggplot(aes(origin, country_label, fill = share)) +
    geom_tile(colour = "white", linewidth = lancet_heatmap_tile_linewidth) +
    scale_fill_infant_burden(
      labels = label_lancet_percent(accuracy = 1),
      guide = guide_colourbar(
        barwidth = grid::unit(40, "mm"),
        barheight = grid::unit(3, "mm"),
        title.position = "top"
      )
    ) +
    labs(x = "Infection source history", y = NULL, fill = "Share of all infections") +
    theme_lancet_heatmap(x_angle = 90, x_hjust = 1, x_vjust = 0.5)
}

extended6 <- free(p_ed6a) + free(p_ed6b) + p_ed6c + free(p_ed6d) +
  plot_layout(design = "AA\nBB\nCD", guides = "keep", heights = c(0.82, 0.82, 1.0)) +
  plot_annotation(tag_levels = "A") &
  (theme(plot.margin = margin(3, 3, 3, 3)) + theme_lancet_tags())

save_appendix_figure(extended6, "extended_data_figure_4_baseline_dynamics", height = 7)
