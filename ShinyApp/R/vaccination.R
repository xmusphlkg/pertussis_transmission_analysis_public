vaccine_susceptibility_r <- function(ve_sus, relative_effect = 1) {
  max(0, 1 - max(0, as.numeric(relative_effect)) * as.numeric(ve_sus))
}

origin_relative_effect_r <- function(origin,
                                     waned_relative_effect = 0.35,
                                     maternal_relative_effect = 0.75,
                                     dose1_relative_effect = 0.45,
                                     dose2_relative_effect = 0.75) {
  waned <- clip_probability_r(waned_relative_effect)
  if (origin == "maternal") {
    return(clip_probability_r(maternal_relative_effect))
  }
  if (startsWith(origin, "dose1_")) {
    dose_effect <- clip_probability_r(dose1_relative_effect)
    return(dose_effect * ifelse(grepl("_waned$", origin), waned, 1))
  }
  if (startsWith(origin, "dose2_")) {
    dose_effect <- clip_probability_r(dose2_relative_effect)
    return(dose_effect * ifelse(grepl("_waned$", origin), waned, 1))
  }
  if (origin == "recent") {
    return(1)
  }
  if (origin == "waned") {
    return(waned)
  }
  0
}

origin_symptomatic_probability_r <- function(base_probability, ve_sym, origin,
                                             waned_relative_effect = 0.35,
                                             maternal_relative_effect = 0.75,
                                             dose1_relative_effect = 0.45,
                                             dose2_relative_effect = 0.75) {
  effect <- origin_relative_effect_r(
    origin,
    waned_relative_effect = waned_relative_effect,
    maternal_relative_effect = maternal_relative_effect,
    dose1_relative_effect = dose1_relative_effect,
    dose2_relative_effect = dose2_relative_effect
  )
  clip_probability_r(as.numeric(base_probability) * (1 - as.numeric(ve_sym) * effect))
}

origin_infectiousness_multiplier_r <- function(ve_inf, origin,
                                               waned_relative_effect = 0.35,
                                               maternal_relative_effect = 0.75,
                                               dose1_relative_effect = 0.45,
                                               dose2_relative_effect = 0.75) {
  effect <- origin_relative_effect_r(
    origin,
    waned_relative_effect = waned_relative_effect,
    maternal_relative_effect = maternal_relative_effect,
    dose1_relative_effect = dose1_relative_effect,
    dose2_relative_effect = dose2_relative_effect
  )
  clip_probability_r(1 - as.numeric(ve_inf) * effect)
}

origin_recovery_rate_multiplier_r <- function(ve_dur, origin,
                                              waned_relative_effect = 0.35,
                                              maternal_relative_effect = 0.75,
                                              dose1_relative_effect = 0.45,
                                              dose2_relative_effect = 0.75) {
  effect <- origin_relative_effect_r(
    origin,
    waned_relative_effect = waned_relative_effect,
    maternal_relative_effect = maternal_relative_effect,
    dose1_relative_effect = dose1_relative_effect,
    dose2_relative_effect = dose2_relative_effect
  )
  1 / max(0.05, 1 - as.numeric(ve_dur) * effect)
}

origin_is_vaccine_dose_r <- function(origin) {
  startsWith(origin, "dose") || origin %in% c("recent", "waned")
}

origin_is_waned_r <- function(origin) {
  grepl("_waned$", origin) || identical(origin, "waned")
}

origin_dose_category_r <- function(origin) {
  if (startsWith(origin, "dose1_")) return("dose1")
  if (startsWith(origin, "dose2_")) return("dose2")
  if (origin %in% c("recent", "waned")) return("dose3plus")
  origin
}

default_initial_origin_distribution_r <- function(age) {
  switch(
    age,
    infant_0_2m = c(maternal = 1),
    infant_3_11m = c(dose1_recent = 0.25, dose2_recent = 0.35, recent = 0.40),
    child_1_4y = c(dose2_waned = 0.10, recent = 0.55, waned = 0.35),
    child_5_9y = c(dose2_waned = 0.10, recent = 0.40, waned = 0.50),
    adolescent_10_17y = c(dose2_waned = 0.10, recent = 0.25, waned = 0.65),
    young_adult_18_39y = c(dose2_waned = 0.10, recent = 0.10, waned = 0.80),
    middle_adult_40_64y = c(dose2_waned = 0.05, recent = 0.05, waned = 0.90),
    elderly_65plus = c(dose2_waned = 0.05, recent = 0.03, waned = 0.92),
    c(dose2_waned = 0.10, recent = 0.10, waned = 0.80)
  )
}

default_routine_target_origin_distribution_r <- function(age) {
  switch(
    age,
    infant_0_2m = numeric(),
    infant_3_11m = c(dose1_recent = 0.25, dose2_recent = 0.35, recent = 0.40),
    child_1_4y = c(recent = 0.65, waned = 0.35),
    child_5_9y = c(recent = 0.50, waned = 0.50),
    adolescent_10_17y = c(recent = 0.30, waned = 0.70),
    young_adult_18_39y = c(recent = 0.10, waned = 0.90),
    middle_adult_40_64y = c(recent = 0.05, waned = 0.95),
    elderly_65plus = c(recent = 0.03, waned = 0.97),
    c(recent = 0.10, waned = 0.90)
  )
}
