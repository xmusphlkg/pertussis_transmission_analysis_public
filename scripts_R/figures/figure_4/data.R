## Figure 4 data preparation ---------------------------------------------------

figure_4_endpoint_levels <- function(include_resistance = TRUE) {
  endpoints <- c(
    "Infant cases",
    "Infant hospitalisations",
    "Infant deaths",
    "Children cases",
    "Adolescent cases",
    "All <18 cases"
  )
  if (isTRUE(include_resistance)) {
    endpoints <- c(endpoints, "Resistant infections")
  }
  endpoints
}

figure_4_endpoint_axis_labels <- function() {
  c(
    `Infant cases` = "Infant cases",
    `Infant hospitalisations` = "Infant hospitalisations",
    `Infant deaths` = "Infant deaths",
    `Children cases` = "Children cases",
    `Adolescent cases` = "Adolescent cases",
    `All <18 cases` = "<18 symptomatic cases",
    `Resistant infections` = "Resistant infections"
  )
}

figure_4_vaccine_setting_levels <- function() {
  c(
    "no_vaccine",
    "symptom_protective",
    "infection_blocking",
    "transmission_blocking",
    "next_generation"
  )
}

figure_4_vaccine_setting_labels <- function() {
  c(
    no_vaccine = "No vaccine",
    symptom_protective = "Current aP-like",
    infection_blocking = "Infection blocking",
    transmission_blocking = "Transmission blocking",
    next_generation = "High-blocking target"
  )
}

figure_4_management_metric_map <- function() {
  c(
    infant_cases_per_100k = "Infant cases",
    infant_hospitalizations_per_100k = "Infant hospitalisations",
    infant_deaths_per_100k = "Infant deaths",
    child_1_9_cases_per_100k = "Children cases",
    adolescent_cases_per_100k = "Adolescent cases",
    primary_cases_per_100k = "All <18 cases",
    annualized_resistant_infections_per_100k = "Resistant infections"
  )
}

figure_4_vaccine_metric_map <- function() {
  c(
    infant_cases = "Infant cases",
    infant_hospitalizations = "Infant hospitalisations",
    infant_deaths = "Infant deaths",
    child_1_9_cases = "Children cases",
    adolescent_cases = "Adolescent cases",
    child_adolescent_cases = "All <18 cases",
    resistant_infections = "Resistant infections"
  )
}

load_figure_4_inputs <- function() {
  if (!requireNamespace("arrow", quietly = TRUE)) {
    stop("Figure 4 requires the R package 'arrow' to read the detailed vaccine simulations.", call. = FALSE)
  }

  vaccine_event_columns <- c(
    "country", "scenario", "VE_sus", "VE_sym", "VE_inf", "VE_dur",
    "history_vaccine_scenario", "vaccine_transition_design",
    "vaccine_transition_interpretation", "simulation_phase", "prospective_policy",
    "infant_cases", "infant_hospitalizations", "infant_deaths",
    "child_1_9_cases", "adolescent_cases", "child_adolescent_cases"
  )

  list(
    decision_frontier = read_table("lancet_child_adolescent_decision_frontier.csv"),
    vaccine_summary = read_summary("vaccine_scenarios_summary.csv"),
    vaccine_timeseries = arrow::read_parquet(
      model_path("outputs", "simulations", "vaccine_scenarios.parquet"),
      col_select = tidyselect::all_of(vaccine_event_columns),
      as_data_frame = TRUE
    )
  )
}

prepare_figure_4_data <- function(inputs = load_figure_4_inputs()) {
  endpoint_levels <- figure_4_endpoint_levels(include_resistance = TRUE)
  endpoint_axis_labels <- figure_4_endpoint_axis_labels()
  management_metric_map <- figure_4_management_metric_map()
  vaccine_metric_map <- figure_4_vaccine_metric_map()
  vaccine_setting_levels <- figure_4_vaccine_setting_levels()
  vaccine_setting_labels <- figure_4_vaccine_setting_labels()

  decision_frontier <- inputs$decision_frontier %>%
    required_columns(
      c(
        "country", "strategy", "optimization_constraint",
        names(management_metric_map)
      ),
      "Lancet child/adolescent decision frontier"
    ) %>%
    mutate(
      country = stringr::str_replace_all(country, " ", "_"),
      country_name = format_country(country),
      strategy = as.character(strategy),
      across(all_of(names(management_metric_map)), as.numeric)
    )

  management_pairwise <- decision_frontier %>%
    filter(
      optimization_constraint == "program_plus_resistance",
      strategy %in% c("timeliness_only", "resistance_guided_treatment")
    ) %>%
    select(country, country_name, strategy, all_of(names(management_metric_map))) %>%
    distinct(country, strategy, .keep_all = TRUE) %>%
    pivot_longer(
      cols = all_of(names(management_metric_map)),
      names_to = "metric_key",
      values_to = "raw_burden"
    ) %>%
    mutate(endpoint = unname(management_metric_map[metric_key])) %>%
    select(-metric_key) %>%
    pivot_wider(names_from = strategy, values_from = raw_burden) %>%
    rename(
      routine_burden = timeliness_only,
      guided_burden = resistance_guided_treatment
    ) %>%
    mutate(
      residual_index = guided_burden / routine_burden,
      reduction_vs_routine = 1 - residual_index,
      endpoint = factor(endpoint, levels = endpoint_levels),
      residual_label = paste0(format_lancet_fixed(100 * residual_index, 0), "%"),
      residual_text_colour = if_else(
        residual_index < 0.60 | residual_index > 1.13,
        "white",
        lancet_text_colour
      ),
      direction = case_when(
        residual_index < 1 - 1e-10 ~ "Guided lower",
        residual_index > 1 + 1e-10 ~ "Routine lower",
        TRUE ~ "Equal"
      ),
      contrast_definition = "resistance-guided raw burden / routine-timeliness raw burden"
    )

  if (
    nrow(management_pairwise) != 9L * length(endpoint_levels) ||
      any(!is.finite(management_pairwise$residual_index)) ||
      any(management_pairwise$routine_burden <= 0) ||
      any(management_pairwise$guided_burden <= 0)
  ) {
    stop("Figure 4a requires positive guided/routine burdens for seven outcomes in nine profiles.", call. = FALSE)
  }

  available_country_labels <- unique(management_pairwise$country_name)
  management_profile_order <- main_figure_country_order(available_country_labels)
  if (length(management_profile_order) != length(main_figure_country_label_levels) ||
      !setequal(management_profile_order, available_country_labels)) {
    stop("Figure 4 countries do not match the fixed main-figure order.", call. = FALSE)
  }

  management_pairwise <- management_pairwise %>%
    mutate(country_label = factor(country_name, levels = rev(management_profile_order)))

  management_summary <- management_pairwise %>%
    group_by(endpoint) %>%
    summarise(
      profiles_guided_lower = sum(residual_index < 1 - 1e-10),
      profiles_routine_lower = sum(residual_index > 1 + 1e-10),
      median_residual_index = median(residual_index),
      q25_residual_index = interval_quantile(residual_index, 0.25),
      q75_residual_index = interval_quantile(residual_index, 0.75),
      min_residual_index = min(residual_index),
      max_residual_index = max(residual_index),
      .groups = "drop"
    )

  vaccine_summary <- inputs$vaccine_summary %>%
    required_columns(
      c("country", "scenario", "vaccine_scenario", "resistant_infections"),
      "Vaccine-scenario summary"
    ) %>%
    transmute(
      country = stringr::str_replace_all(country, " ", "_"),
      scenario = as.character(scenario),
      resistant_infections = as.numeric(resistant_infections)
    )

  vaccine_event_columns <- setdiff(names(vaccine_metric_map), "resistant_infections")
  vaccine_events <- inputs$vaccine_timeseries %>%
    required_columns(
      c(
        "country", "scenario", "VE_sus", "VE_sym", "VE_inf", "VE_dur",
        "history_vaccine_scenario", "vaccine_transition_design",
        "vaccine_transition_interpretation", "simulation_phase", "prospective_policy",
        vaccine_event_columns
      ),
      "Detailed vaccine-scenario simulations"
    ) %>%
    mutate(
      country = stringr::str_replace_all(country, " ", "_"),
      scenario = as.character(scenario),
      across(all_of(c("VE_sus", "VE_sym", "VE_inf", "VE_dur", vaccine_event_columns)), as.numeric),
      history_vaccine_scenario = as.character(history_vaccine_scenario),
      vaccine_transition_design = as.character(vaccine_transition_design),
      vaccine_transition_interpretation = as.character(vaccine_transition_interpretation),
      simulation_phase = as.character(simulation_phase),
      prospective_policy = as.logical(prospective_policy)
    ) %>%
    group_by(country, scenario) %>%
    summarise(
      across(all_of(vaccine_event_columns), ~ sum(.x, na.rm = TRUE)),
      across(all_of(c("VE_sus", "VE_sym", "VE_inf", "VE_dur")), ~ dplyr::first(.x)),
      across(
        all_of(c(
          "history_vaccine_scenario", "vaccine_transition_design",
          "vaccine_transition_interpretation", "simulation_phase", "prospective_policy"
        )),
        ~ dplyr::first(.x)
      ),
      .groups = "drop"
    ) %>%
    left_join(vaccine_summary, by = c("country", "scenario"))

  if (
    nrow(vaccine_events) != 9L * length(vaccine_setting_levels) ||
      anyDuplicated(vaccine_events[c("country", "scenario")]) > 0L ||
      !setequal(unique(vaccine_events$scenario), vaccine_setting_levels) ||
      any(!is.finite(as.matrix(select(vaccine_events, all_of(names(vaccine_metric_map)))))) ||
      any(as.matrix(select(vaccine_events, all_of(names(vaccine_metric_map)))) <= 0) ||
      !all(vaccine_events$history_vaccine_scenario == "symptom_protective") ||
      !all(vaccine_events$vaccine_transition_design ==
        "immediate_full_mechanism_replacement_at_policy_t0") ||
      !all(vaccine_events$vaccine_transition_interpretation ==
        "mechanism_target_not_cohort_rollout") ||
      !all(vaccine_events$simulation_phase == "prospective_policy") ||
      !all(vaccine_events$prospective_policy)
  ) {
    stop(
      paste(
        "Figure 4b requires complete positive results for all five vaccine settings",
        "in nine profiles branching from one current-aP prospective-policy state."
      ),
      call. = FALSE
    )
  }

  vaccine_parameters <- vaccine_events %>%
    distinct(scenario, VE_sus, VE_sym, VE_inf, VE_dur) %>%
    arrange(match(scenario, vaccine_setting_levels))

  if (nrow(vaccine_parameters) != length(vaccine_setting_levels)) {
    stop("Figure 4b vaccine-parameter definitions are not unique by setting.", call. = FALSE)
  }

  vaccine_outcomes <- vaccine_events %>%
    select(
      country, scenario, history_vaccine_scenario, vaccine_transition_design,
      vaccine_transition_interpretation, simulation_phase, prospective_policy,
      all_of(names(vaccine_metric_map))
    ) %>%
    pivot_longer(
      cols = all_of(names(vaccine_metric_map)),
      names_to = "metric_key",
      values_to = "scenario_burden"
    ) %>%
    mutate(endpoint = unname(vaccine_metric_map[metric_key])) %>%
    select(-metric_key)

  vaccine_current <- vaccine_outcomes %>%
    filter(scenario == "symptom_protective") %>%
    select(country, endpoint, current_burden = scenario_burden)

  vaccine_outcomes <- vaccine_outcomes %>%
    left_join(vaccine_current, by = c("country", "endpoint")) %>%
    mutate(
      residual_index = scenario_burden / current_burden,
      reduction_vs_current = 1 - residual_index,
      scenario_label_text = unname(vaccine_setting_labels[scenario]),
      scenario_label = factor(
        scenario_label_text,
        levels = rev(unname(vaccine_setting_labels[vaccine_setting_levels]))
      ),
      endpoint = factor(endpoint, levels = endpoint_levels),
      above_current = residual_index > 1 + 1e-10,
      contrast_definition = paste(
        "scenario cumulative burden / current aP-like cumulative burden;",
        "all settings branch from the same current-aP state at policy t0"
      )
    )

  vaccine_summary_matrix <- vaccine_outcomes %>%
    group_by(scenario, scenario_label, endpoint) %>%
    summarise(
      profiles_evaluated = n(),
      profiles_above_current = sum(above_current),
      median_residual_index = median(residual_index),
      q25_residual_index = interval_quantile(residual_index, 0.25),
      q75_residual_index = interval_quantile(residual_index, 0.75),
      min_residual_index = min(residual_index),
      max_residual_index = max(residual_index),
      .groups = "drop"
    ) %>%
    mutate(
      residual_label = paste0(
        format_lancet_fixed(100 * median_residual_index, 0), "% [",
        format_lancet_fixed(100 * q25_residual_index, 0), "–",
        format_lancet_fixed(100 * q75_residual_index, 0), "]\n",
        ">100%: ", profiles_above_current, "/", profiles_evaluated
      ),
      residual_text_colour = if_else(
        median_residual_index < 0.62 | median_residual_index > 2.05,
        "white",
        lancet_text_colour
      ),
      heterogeneous_harm_signal = scenario %in%
        c("infection_blocking", "transmission_blocking", "next_generation") &
        profiles_above_current > 0L
    )

  if (
    nrow(vaccine_outcomes) != 9L * length(vaccine_setting_levels) * length(endpoint_levels) ||
      nrow(vaccine_summary_matrix) != length(vaccine_setting_levels) * length(endpoint_levels) ||
      any(!is.finite(vaccine_outcomes$residual_index)) ||
      any(abs(
        vaccine_outcomes$residual_index[vaccine_outcomes$scenario == "symptom_protective"] - 1
      ) > 1e-10)
  ) {
    stop("Figure 4b residual-burden calculations failed their common-current contract.", call. = FALSE)
  }

  list(
    endpoint_levels = endpoint_levels,
    endpoint_axis_labels = endpoint_axis_labels,
    vaccine_setting_levels = vaccine_setting_levels,
    vaccine_setting_labels = vaccine_setting_labels,
    decision_frontier = decision_frontier,
    management_pairwise = management_pairwise,
    management_summary = management_summary,
    management_profile_order = management_profile_order,
    vaccine_events = vaccine_events,
    vaccine_parameters = vaccine_parameters,
    vaccine_outcomes = vaccine_outcomes,
    vaccine_summary_matrix = vaccine_summary_matrix
  )
}
