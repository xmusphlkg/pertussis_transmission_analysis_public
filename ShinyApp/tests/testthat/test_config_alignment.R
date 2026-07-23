root <- normalizePath(file.path(getwd(), "ShinyApp"), mustWork = FALSE)
if (!file.exists(file.path(root, "R", "config_io.R"))) {
  root <- normalizePath(file.path(getwd(), "..", ".."), mustWork = FALSE)
}
Sys.setenv(SHINYAPP_ROOT = root)

source(file.path(root, "R", "config_io.R"))

test_that("exploratory app baseline matches submitted core parameter meanings", {
  configs <- load_app_configs_r()
  baseline <- configs$baseline
  submission_path <- normalizePath(
    file.path(root, "data", configs$settings$runtime$app_scope$authoritative_submission_config),
    mustWork = TRUE
  )
  submitted <- read_yaml_r(submission_path)$runtime$baseline_parameters

  expect_equal(baseline$natural_history$latent_duration, 9.0)
  expect_equal(baseline$natural_history$infectious_duration_symptomatic, 24.0)
  expect_equal(baseline$natural_history$infectious_duration_asymptomatic, 17.3)
  expect_equal(baseline$natural_history$recovered_immunity_duration, 12227.5)
  expect_equal(baseline$natural_history$vaccine_protection_duration, 798.0)
  expect_equal(baseline$immunity_model$waned_vaccine_duration, 798.0)
  expect_equal(baseline$natural_history$R_to_W_duration, 6113.75)
  expect_equal(baseline$natural_history$W_to_S_duration, 6113.75)
  expect_equal(baseline$PEP$effectiveness_sensitive, 0.823)

  paths <- c(
    "natural_history.latent_duration",
    "natural_history.infectious_duration_symptomatic",
    "natural_history.infectious_duration_asymptomatic",
    "natural_history.recovered_immunity_duration",
    "natural_history.vaccine_protection_duration",
    "natural_history.R_to_W_duration",
    "natural_history.W_to_S_duration",
    "immunity_model.waned_vaccine_duration",
    "PEP.coverage_household_contacts",
    "PEP.effectiveness_sensitive",
    "PEP.effectiveness_resistant",
    "PEP.activation_prevalence"
  )
  for (path in paths) {
    expect_equal(value_at_path_r(baseline, path), value_at_path_r(submitted, path), info = path)
  }

  expect_identical(configs$settings$runtime$app_scope$reproduces_submitted_analyses, FALSE)
  expect_identical(baseline$calibration$reproduces_submitted_calibration, FALSE)
  expect_identical(baseline$calibration$method, "staged_fast")
})
