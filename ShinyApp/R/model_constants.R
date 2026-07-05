MODEL_AGE_GROUPS <- c(
  "infant_0_2m",
  "infant_3_11m",
  "child_1_4y",
  "child_5_9y",
  "adolescent_10_17y",
  "young_adult_18_39y",
  "middle_adult_40_64y",
  "elderly_65plus"
)

MODEL_AGE_LABELS <- c(
  infant_0_2m = "0-2 months",
  infant_3_11m = "3-11 months",
  child_1_4y = "1-4 years",
  child_5_9y = "5-9 years",
  adolescent_10_17y = "10-17 years",
  young_adult_18_39y = "18-39 years",
  middle_adult_40_64y = "40-64 years",
  elderly_65plus = "65+ years"
)

MODEL_INFANT_AGE_GROUPS <- c("infant_0_2m", "infant_3_11m")
MODEL_CHILD_1_9_AGE_GROUPS <- c("child_1_4y", "child_5_9y")
MODEL_ADOLESCENT_AGE_GROUPS <- c("adolescent_10_17y")
MODEL_CHILD_ADOLESCENT_AGE_GROUPS <- c(
  MODEL_INFANT_AGE_GROUPS,
  MODEL_CHILD_1_9_AGE_GROUPS,
  MODEL_ADOLESCENT_AGE_GROUPS
)

MODEL_VACCINE_ORIGINS <- c(
  "unvaccinated",
  "maternal",
  "dose1_recent",
  "dose1_waned",
  "dose2_recent",
  "dose2_waned",
  "recent",
  "waned"
)

MODEL_STRAINS <- c("S", "R")

MODEL_SUSCEPTIBLE_BY_ORIGIN <- c(
  unvaccinated = "S",
  maternal = "M_protected",
  dose1_recent = "V_dose1_recent",
  dose1_waned = "V_dose1_waned",
  dose2_recent = "V_dose2_recent",
  dose2_waned = "V_dose2_waned",
  recent = "V_recent",
  waned = "V_waned"
)

MODEL_COMPARTMENT_ALIASES <- c(
  V = "M_protected",
  V_dose3plus_recent = "V_recent",
  V_dose3plus_waned = "V_waned",
  R = "R_natural",
  W = "W_natural",
  E_S = "E_S_unvaccinated",
  E_R = "E_R_unvaccinated",
  I_S_sym = "I_S_sym_unvaccinated",
  I_S_asym = "I_S_asym_unvaccinated",
  I_R_sym = "I_R_sym_unvaccinated",
  I_R_asym = "I_R_asym_unvaccinated",
  T_S = "T_S_unvaccinated",
  T_R = "T_R_unvaccinated"
)

susceptible_name_r <- function(origin) {
  unname(MODEL_SUSCEPTIBLE_BY_ORIGIN[[origin]])
}

exposed_name_r <- function(strain, origin) {
  paste0("E_", strain, "_", origin)
}

infectious_name_r <- function(strain, symptom, origin) {
  paste0("I_", strain, "_", symptom, "_", origin)
}

treated_name_r <- function(strain, origin) {
  paste0("T_", strain, "_", origin)
}

resolve_compartment_name_r <- function(name) {
  if (name %in% names(MODEL_COMPARTMENT_ALIASES)) {
    return(unname(MODEL_COMPARTMENT_ALIASES[[name]]))
  }
  name
}

MODEL_COMPARTMENTS <- c(
  unname(MODEL_SUSCEPTIBLE_BY_ORIGIN),
  unlist(lapply(MODEL_STRAINS, function(strain) {
    vapply(MODEL_VACCINE_ORIGINS, function(origin) exposed_name_r(strain, origin), character(1))
  }), use.names = FALSE),
  unlist(lapply(MODEL_STRAINS, function(strain) {
    unlist(lapply(c("sym", "asym"), function(symptom) {
      vapply(MODEL_VACCINE_ORIGINS, function(origin) infectious_name_r(strain, symptom, origin), character(1))
    }), use.names = FALSE)
  }), use.names = FALSE),
  unlist(lapply(MODEL_STRAINS, function(strain) {
    vapply(MODEL_VACCINE_ORIGINS, function(origin) treated_name_r(strain, origin), character(1))
  }), use.names = FALSE),
  "R_natural",
  "W_natural"
)

if (length(MODEL_COMPARTMENTS) != 74) {
  stop("Internal model error: expected 74 compartments per age group.")
}

make_state_index_r <- function(age_groups = MODEL_AGE_GROUPS) {
  list(
    age_groups = age_groups,
    compartments = MODEL_COMPARTMENTS,
    n_age = length(age_groups),
    n_compartments = length(MODEL_COMPARTMENTS),
    size = length(age_groups) * length(MODEL_COMPARTMENTS)
  )
}

state_matrix_r <- function(y, index) {
  matrix(
    as.numeric(y),
    nrow = index$n_age,
    ncol = index$n_compartments,
    byrow = TRUE,
    dimnames = list(index$age_groups, index$compartments)
  )
}

state_vector_r <- function(state) {
  as.numeric(t(state))
}

compartment_index_r <- function(name, index = make_state_index_r()) {
  resolved <- resolve_compartment_name_r(name)
  out <- match(resolved, index$compartments)
  if (is.na(out)) {
    stop("Unknown compartment: ", name, call. = FALSE)
  }
  out
}

strain_state_names_r <- function(strain) {
  c(
    vapply(MODEL_VACCINE_ORIGINS, function(origin) exposed_name_r(strain, origin), character(1)),
    unlist(lapply(c("sym", "asym"), function(symptom) {
      vapply(MODEL_VACCINE_ORIGINS, function(origin) infectious_name_r(strain, symptom, origin), character(1))
    }), use.names = FALSE),
    vapply(MODEL_VACCINE_ORIGINS, function(origin) treated_name_r(strain, origin), character(1))
  )
}

clip_probability_r <- function(x) {
  pmin(pmax(as.numeric(x), 0), 1)
}

assert_probability_r <- function(x, label = "value") {
  values <- as.numeric(x)
  if (any(!is.finite(values)) || any(values < 0 | values > 1)) {
    stop(label, " must be finite and within [0, 1].", call. = FALSE)
  }
  invisible(TRUE)
}
