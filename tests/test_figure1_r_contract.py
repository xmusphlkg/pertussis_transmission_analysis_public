from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript is unavailable")
def test_r_figure1a_requires_all_six_who_regions() -> None:
    expression = r"""
    source('scripts_R/lib/bootstrap.R')
    source('scripts_R/figures/figure_1/data.R')
    source('scripts_R/figures/figure_1/panels.R')

    expected_regions <- c(
      'European Region',
      'Western Pacific Region',
      'Eastern Mediterranean Region',
      'South-East Asia Region',
      'Region of the Americas',
      'African Region'
    )
    stopifnot(identical(figure_1_selected_regions(), expected_regions))
    stopifnot(setequal(
      names(figure_1_region_display_labels()),
      c('Global', expected_regions)
    ))
    stopifnot(setequal(
      figure_1_region_label_positions()$region_key,
      c('Global', expected_regions)
    ))
    stopifnot(all(c('Global', expected_regions) %in% names(region_colours)))
    """

    subprocess.run(
        ["Rscript", "-e", expression],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript is unavailable")
def test_r_figure1b_requires_audited_intervals_for_all_endpoints_and_draws_them() -> None:
    expression = r"""
    source('scripts_R/lib/bootstrap.R')
    source('scripts_R/figures/figure_1/data.R')
    source('scripts_R/figures/figure_1/panels.R')

    conditional_metadata <- list(
      bootstrap_data_generation = 'conditional_fitted_AR1_path_plus_NB2_measurement',
      conditional_on_fitted_latent_process_path = TRUE,
      latent_process_path_regenerated = FALSE,
      bootstrap_refit = 'country_state_space_MAP_full_refit_per_replicate'
    )
    validate_figure_1b_interval_metadata(conditional_metadata)
    marginal_metadata <- conditional_metadata
    marginal_metadata$bootstrap_data_generation <- 'marginal_AR1_process_plus_NB2_measurement'
    marginal_metadata$conditional_on_fitted_latent_process_path <- FALSE
    marginal_metadata$latent_process_path_regenerated <- TRUE
    failed <- tryCatch({
      validate_figure_1b_interval_metadata(marginal_metadata)
      FALSE
    }, error = function(e) TRUE)
    stopifnot(failed)

    intervals <- expand.grid(
      country = c('A', 'B'),
      outcome = c('Reports', 'Symptomatic', 'Infections'),
      stringsAsFactors = FALSE
    )
    intervals$rate_q025 <- case_when(
      intervals$country == 'A' & intervals$outcome == 'Reports' ~ 10,
      intervals$country == 'A' & intervals$outcome == 'Symptomatic' ~ 80,
      intervals$country == 'A' & intervals$outcome == 'Infections' ~ 400,
      intervals$country == 'B' & intervals$outcome == 'Reports' ~ 20,
      intervals$country == 'B' & intervals$outcome == 'Symptomatic' ~ 180,
      TRUE ~ 700
    )
    intervals$rate_q975 <- case_when(
      intervals$country == 'A' & intervals$outcome == 'Reports' ~ 30,
      intervals$country == 'A' & intervals$outcome == 'Symptomatic' ~ 160,
      intervals$country == 'A' & intervals$outcome == 'Infections' ~ 600,
      intervals$country == 'B' & intervals$outcome == 'Reports' ~ 60,
      intervals$country == 'B' & intervals$outcome == 'Symptomatic' ~ 320,
      TRUE ~ 900
    )
    intervals$bootstrap_replicates <- 1008L
    intervals$interval_type <- '95% parametric-bootstrap confidence interval'
    intervals$confidence_interval_method <- 'percentile_parametric_bootstrap'
    intervals$interval_basis <- paste(
      'Current-practice conditional parametric bootstrap. The accepted fitted',
      'annual latent transmission path is held as the generating trajectory;',
      'NB2 observations are regenerated and the country state-space model is fully refitted.'
    )

    prepared_intervals <- prepare_figure_1b_endpoint_intervals(
      intervals,
      publication_countries = c('A', 'B'),
      bootstrap_replicates = 1024L,
      minimum_successful_replicates = 1000L
    )
    stopifnot(nrow(prepared_intervals) == 6L)
    stopifnot(identical(
      prepared_intervals$interval_lower_per_100k,
      c(10, 80, 400, 20, 180, 700)
    ))
    stopifnot(identical(
      prepared_intervals$interval_upper_per_100k,
      c(30, 160, 600, 60, 320, 900)
    ))

    inconsistent <- intervals
    inconsistent$rate_q975[[1]] <- 5
    failed <- tryCatch({
      prepare_figure_1b_endpoint_intervals(
        inconsistent,
        publication_countries = c('A', 'B'),
        bootstrap_replicates = 1024L,
        minimum_successful_replicates = 1000L
      )
      FALSE
    }, error = function(e) TRUE)
    stopifnot(failed)

    mislabeled <- intervals
    mislabeled$interval_type <- '95% posterior credible interval'
    failed <- tryCatch({
      prepare_figure_1b_endpoint_intervals(
        mislabeled,
        publication_countries = c('A', 'B'),
        bootstrap_replicates = 1024L,
        minimum_successful_replicates = 1000L
      )
      FALSE
    }, error = function(e) TRUE)
    stopifnot(failed)

    outcome_levels <- c('Reports', 'Symptomatic', 'Infections')
    endpoint_long <- expand.grid(
      country = c('A', 'B'),
      outcome = outcome_levels,
      stringsAsFactors = FALSE
    ) %>%
      mutate(
        outcome = factor(outcome, levels = outcome_levels),
        country_y_base = if_else(country == 'A', 2, 1),
        outcome_offset = recode(
          as.character(outcome),
          Reports = -0.16,
          Symptomatic = 0,
          Infections = 0.16
        ),
        country_y = country_y_base + outcome_offset,
        rate_per_100k = case_when(
          country == 'A' & outcome == 'Reports' ~ 20,
          country == 'A' & outcome == 'Symptomatic' ~ 120,
          country == 'A' & outcome == 'Infections' ~ 500,
          country == 'B' & outcome == 'Reports' ~ 40,
          country == 'B' & outcome == 'Symptomatic' ~ 240,
          TRUE ~ 800
        )
      )
    endpoint_spans <- endpoint_long %>%
      group_by(country, country_y_base) %>%
      summarise(
        min_rate_per_100k = min(rate_per_100k),
        max_rate_per_100k = max(rate_per_100k),
        .groups = 'drop'
      )
    plot_intervals <- prepared_intervals %>%
      left_join(
        endpoint_long %>%
          select(country, outcome, country_y, rate_per_100k),
        by = c('country', 'outcome')
      )
    panel_data <- list(
      baseline_endpoint_long = endpoint_long,
      endpoint_spans = endpoint_spans,
      endpoint_intervals = plot_intervals,
      country_order = c('A', 'B')
    )
    panel <- plot_figure_1_panel_b(panel_data)
    built <- ggplot2::ggplot_build(panel)
    stopifnot(length(built$data) == 5L)
    stopifnot(nrow(built$data[[2]]) == 6L)
    stopifnot(all(built$data[[2]]$x == c(10, 80, 400, 20, 180, 700)))
    stopifnot(all(built$data[[2]]$xend == c(30, 160, 600, 60, 320, 900)))
    shape_scale <- built$plot$scales$get_scales('shape')
    stopifnot(identical(as.character(shape_scale$get_breaks()), outcome_levels))
    stopifnot(identical(
      as.character(shape_scale$get_labels()),
      c('Reported cases', 'Symptomatic cases', 'Infections')
    ))
    legend_guide <- built$plot$guides$guides[[1]]
    expected_legend_colours <- unname(
      figure_1_endpoint_colours()[outcome_levels]
    )
    stopifnot(identical(
      legend_guide$params$override.aes$colour,
      expected_legend_colours
    ))
    """

    subprocess.run(
        ["Rscript", "-e", expression],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
