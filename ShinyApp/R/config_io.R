app_root_r <- function() {
  env_root <- Sys.getenv("SHINYAPP_ROOT", unset = "")
  if (nzchar(env_root)) {
    return(normalizePath(env_root, mustWork = FALSE))
  }

  sourced_file <- tryCatch(normalizePath(sys.frame(1)$ofile, mustWork = FALSE), error = function(e) NA_character_)
  if (!is.na(sourced_file) && nzchar(sourced_file)) {
    return(normalizePath(file.path(dirname(sourced_file), ".."), mustWork = FALSE))
  }

  if (file.exists(file.path(getwd(), "data", "model_settings.yaml"))) {
    return(normalizePath(getwd(), mustWork = FALSE))
  }
  if (file.exists(file.path(getwd(), "ShinyApp", "data", "model_settings.yaml"))) {
    return(normalizePath(file.path(getwd(), "ShinyApp"), mustWork = FALSE))
  }
  stop("Could not locate ShinyApp root.", call. = FALSE)
}

app_path_r <- function(...) {
  file.path(app_root_r(), ...)
}

require_namespace_r <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop("Package '", pkg, "' is required. Run ShinyApp/install_packages.R first.", call. = FALSE)
  }
  invisible(TRUE)
}

read_yaml_r <- function(path) {
  require_namespace_r("yaml")
  yaml::read_yaml(path)
}

read_csv_r <- function(path, ...) {
  require_namespace_r("readr")
  readr::read_csv(path, show_col_types = FALSE, ...)
}

write_yaml_r <- function(x, path) {
  require_namespace_r("yaml")
  yaml::write_yaml(x, path)
}

deep_update_r <- function(base, updates) {
  if (is.null(updates)) return(base)
  out <- base
  for (nm in names(updates)) {
    if (is.list(updates[[nm]]) && !is.data.frame(updates[[nm]]) &&
        is.list(out[[nm]]) && !is.data.frame(out[[nm]])) {
      out[[nm]] <- deep_update_r(out[[nm]], updates[[nm]])
    } else {
      out[[nm]] <- updates[[nm]]
    }
  }
  out
}

set_by_path_r <- function(config, path, value) {
  parts <- strsplit(path, "\\.", fixed = FALSE)[[1]]
  assign_in <- function(node, idx) {
    key <- parts[[idx]]
    if (idx == length(parts)) {
      node[[key]] <- value
    } else {
      if (is.null(node[[key]]) || !is.list(node[[key]])) node[[key]] <- list()
      node[[key]] <- assign_in(node[[key]], idx + 1)
    }
    node
  }
  assign_in(config, 1)
}

value_at_path_r <- function(config, path) {
  node <- config
  for (part in strsplit(path, "\\.", fixed = FALSE)[[1]]) {
    if (!is.list(node) || is.null(node[[part]])) return(NULL)
    node <- node[[part]]
  }
  node
}

load_app_configs_r <- function() {
  settings <- read_yaml_r(app_path_r("data", "model_settings.yaml"))
  countries <- read_yaml_r(app_path_r("data", "country_profiles.yaml"))
  runtime <- settings$runtime
  if (is.null(runtime)) {
    stop("model_settings.yaml is missing runtime.", call. = FALSE)
  }
  list(
    settings = settings,
    baseline = runtime$baseline_parameters,
    vaccines = runtime$vaccine_scenarios,
    resistance = runtime$resistance_scenarios,
    interventions = runtime$intervention_scenarios,
    sensitivity = runtime$sensitivity_parameters,
    data_sources = runtime$data_sources,
    countries = countries
  )
}

available_countries_r <- function(configs = load_app_configs_r()) {
  names(configs$countries)
}

profile_to_age_frame_r <- function(config) {
  records <- config$age_groups
  data.frame(
    age_group = vapply(records, `[[`, character(1), "label"),
    population = vapply(records, function(x) as.numeric(x$population), numeric(1)),
    vaccine_coverage = vapply(records, function(x) as.numeric(x$vaccine_coverage %||% 0), numeric(1)),
    reporting_rate = vapply(records, function(x) as.numeric(x$reporting_rate %||% 0), numeric(1)),
    symptom_probability = vapply(records, function(x) as.numeric(x$symptom_probability %||% 0.4), numeric(1)),
    stringsAsFactors = FALSE
  )
}

`%||%` <- function(lhs, rhs) {
  if (is.null(lhs)) rhs else lhs
}

validate_age_profile_r <- function(df) {
  required <- c("age_group", "population", "vaccine_coverage", "reporting_rate", "symptom_probability")
  missing <- setdiff(required, names(df))
  if (length(missing)) {
    stop("Age profile is missing columns: ", paste(missing, collapse = ", "), call. = FALSE)
  }
  if (!setequal(as.character(df$age_group), MODEL_AGE_GROUPS)) {
    stop("Age profile must contain exactly the model age groups.", call. = FALSE)
  }
  df <- df[match(MODEL_AGE_GROUPS, as.character(df$age_group)), required]
  df$population <- as.numeric(df$population)
  df$vaccine_coverage <- as.numeric(df$vaccine_coverage)
  df$reporting_rate <- as.numeric(df$reporting_rate)
  df$symptom_probability <- as.numeric(df$symptom_probability)
  if (any(!is.finite(df$population)) || any(df$population <= 0)) {
    stop("Age-group populations must be finite and > 0.", call. = FALSE)
  }
  assert_probability_r(df$vaccine_coverage, "vaccine_coverage")
  assert_probability_r(df$reporting_rate, "reporting_rate")
  assert_probability_r(df$symptom_probability, "symptom_probability")
  df
}

contact_matrix_from_long_r <- function(df) {
  required <- c("source_age_group", "target_age_group", "contacts_per_day")
  missing <- setdiff(required, names(df))
  if (length(missing)) {
    stop("Contact matrix upload is missing columns: ", paste(missing, collapse = ", "), call. = FALSE)
  }
  if (!setequal(as.character(df$source_age_group), MODEL_AGE_GROUPS) ||
      !setequal(as.character(df$target_age_group), MODEL_AGE_GROUPS)) {
    stop("Contact matrix must cover all 8 source and target age groups.", call. = FALSE)
  }
  mat <- matrix(0, nrow = length(MODEL_AGE_GROUPS), ncol = length(MODEL_AGE_GROUPS),
                dimnames = list(MODEL_AGE_GROUPS, MODEL_AGE_GROUPS))
  for (i in seq_len(nrow(df))) {
    target <- as.character(df$target_age_group[[i]])
    source <- as.character(df$source_age_group[[i]])
    mat[target, source] <- as.numeric(df$contacts_per_day[[i]])
  }
  validate_contact_matrix_r(mat)
}

validate_contact_matrix_r <- function(mat) {
  mat <- as.matrix(mat)
  if (!identical(dim(mat), c(8L, 8L))) {
    stop("Contact matrix must be 8 x 8.", call. = FALSE)
  }
  storage.mode(mat) <- "double"
  if (any(!is.finite(mat)) || any(mat < 0)) {
    stop("Contact matrix entries must be finite and non-negative.", call. = FALSE)
  }
  rownames(mat) <- MODEL_AGE_GROUPS
  colnames(mat) <- MODEL_AGE_GROUPS
  mat
}

read_example_table_r <- function(filename) {
  path <- app_path_r("data", "outputs", "tables", filename)
  if (!file.exists(path)) return(data.frame())
  read_csv_r(path)
}

calibration_artifact_path_r <- function(country) {
  safe_country <- gsub(" ", "_", country, fixed = TRUE)
  app_path_r("data", "outputs", "calibrations", paste0(safe_country, "_calibrated_config.yaml"))
}

load_calibration_overlay_r <- function(country) {
  path <- calibration_artifact_path_r(country)
  if (!file.exists(path)) return(list())
  artifact <- read_yaml_r(path)
  cfg <- artifact$config %||% artifact
  overlay <- list()
  for (path_key in c(
    "transmission.beta_S",
    "transmission.seasonal_amplitude",
    "reporting_multiplier",
    "importation.rate_per_100k_per_year",
    "importation.resistant_fraction",
    "resistance.importation_fraction"
  )) {
    value <- value_at_path_r(cfg, path_key)
    if (!is.null(value)) overlay <- set_by_path_r(overlay, path_key, value)
  }
  overlay
}
