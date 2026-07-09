## Extended Data Figure 1 data preparation ------------------------------------

load_extended_data_figure_1_inputs <- function() {
  list(
    profile_inputs = readr::read_csv(model_path("data", "processed", "country_profile_inputs.csv"), show_col_types = FALSE),
    contacts = readr::read_csv(model_path("data", "processed", "country_contact_matrices_8groups.csv"), show_col_types = FALSE),
    resistance_timeline = readr::read_csv(model_path("data", "raw", "country_resistance_timeline.csv"), show_col_types = FALSE)
  )
}

prepare_extended_data_figure_1_data <- function(inputs = load_extended_data_figure_1_inputs()) {
  profile_inputs <- inputs$profile_inputs %>%
    transmute(
      country = config_key,
      country_label = factor(format_country(config_key), levels = country_label_levels),
      dtp1_coverage,
      dtp3_coverage,
      maternal_coverage,
      maternal_program,
      adolescent_booster,
      routine_dose_count,
      routine_first_shot_months,
      routine_last_shot_months
    )

  contacts <- inputs$contacts %>%
    mutate(
      country = stringr::str_replace_all(country, " ", "_"),
      country_label = factor(format_country(country), levels = country_label_levels),
      source_age_group = factor(source_age_group, levels = names(age_labels), labels = age_labels)
    )

  resistance_plot <- inputs$resistance_timeline %>%
    mutate(
      country = str_replace_all(country, " ", "_"),
      country_label = factor(format_country(country), levels = country_label_levels),
      country_code = factor(country_codes[country], levels = country_codes[country_levels]),
      sample_size_plot = replace_na(as.numeric(sample_size), 0),
      evidence_group = case_when(
        str_detect(evidence_type, "^measured") ~ "Measured isolate fraction",
        TRUE ~ "Conservative anchor"
      )
    )

  coverage_plot <- profile_inputs %>%
    select(country_label, dtp1_coverage, dtp3_coverage, maternal_coverage) %>%
    pivot_longer(-country_label, names_to = "programme", values_to = "coverage") %>%
    mutate(programme = factor(
      recode(programme, dtp1_coverage = "DTP1", dtp3_coverage = "DTP3", maternal_coverage = "Maternal"),
      levels = c("DTP1", "DTP3", "Maternal")
    )) %>%
    filter(is.finite(coverage))

  contacts_summary <- contacts %>%
    group_by(country_label, source_age_group) %>%
    summarise(total_contacts = sum(contacts_per_day, na.rm = TRUE), .groups = "drop")

  list(
    profile_inputs = profile_inputs,
    contacts = contacts,
    resistance_plot = resistance_plot,
    coverage_plot = coverage_plot,
    contacts_summary = contacts_summary
  )
}
