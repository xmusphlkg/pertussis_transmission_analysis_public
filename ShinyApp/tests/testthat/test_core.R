root <- normalizePath(file.path(getwd(), "ShinyApp"), mustWork = FALSE)
if (!file.exists(file.path(root, "R", "model_constants.R"))) {
  root <- normalizePath(file.path(getwd(), "..", ".."), mustWork = FALSE)
}
Sys.setenv(SHINYAPP_ROOT = root)

source(file.path(root, "R", "model_constants.R"))
source(file.path(root, "R", "vaccination.R"))
source(file.path(root, "R", "config_io.R"))
source(file.path(root, "R", "prepare_params.R"))
source(file.path(root, "R", "force_of_infection.R"))
source(file.path(root, "R", "ode_rhs.R"))
source(file.path(root, "R", "outputs.R"))
source(file.path(root, "R", "solve_model.R"))
source(file.path(root, "R", "scenarios.R"))
source(file.path(root, "R", "optimizer.R"))

test_that("compartment structure matches the model contract", {
  expect_equal(length(MODEL_COMPARTMENTS), 74)
  index <- make_state_index_r()
  expect_equal(index$size, 8 * 74)
})

test_that("probability validation rejects invalid values", {
  expect_error(assert_probability_r(c(0.1, 1.2), "bad"), "within")
  expect_silent(assert_probability_r(c(0, 0.5, 1), "ok"))
})

test_that("contact matrix validation requires 8x8 non-negative matrix", {
  expect_error(validate_contact_matrix_r(matrix(1, 7, 8)), "8 x 8")
  bad <- matrix(1, 8, 8)
  bad[1, 1] <- -1
  expect_error(validate_contact_matrix_r(bad), "non-negative")
  expect_silent(validate_contact_matrix_r(matrix(1, 8, 8)))
})

test_that("initial state preserves configured population", {
  configs <- load_app_configs_r()
  config <- make_config_r("China", resistance_scenario = "moderate", load_calibration = FALSE, configs = configs)
  config <- set_interactive_runtime_r(config, burn_in_years = 0, analysis_end_date = "2025-12-31")
  params <- prepare_params_r(config, scenario = "test")
  index <- make_state_index_r(params$age_groups)
  state <- state_matrix_r(initial_state_r(params, index), index)
  expect_equal(unname(rowSums(state)), unname(params$population), tolerance = 1e-6)
})

test_that("parallel strategy runner returns all requested strategies", {
  skip_on_os("windows")
  configs <- load_app_configs_r()
  config <- make_config_r("Australia", resistance_scenario = "moderate", load_calibration = FALSE, configs = configs)
  config <- set_interactive_runtime_r(
    config,
    analysis_start_date = "2025-01-01",
    analysis_end_date = "2025-02-01",
    burn_in_years = 0,
    output_time_step = 31,
    solver_method = "rk4",
    rk4_dt_analysis = 31
  )
  strategies <- c("current", "pregnancy_tdap_scaleup", "resistance_guided_treatment")
  result <- run_strategy_set_r(config, strategies = strategies, configs = configs, n_jobs = 2)
  expect_setequal(result$summary$scenario, strategies)
})
