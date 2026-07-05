Sys.setenv(SHINYAPP_ROOT = normalizePath(getwd(), mustWork = FALSE))

source(file.path("R", "model_constants.R"))
source(file.path("R", "vaccination.R"))
source(file.path("R", "config_io.R"))
source(file.path("R", "prepare_params.R"))
source(file.path("R", "force_of_infection.R"))
source(file.path("R", "ode_rhs.R"))
source(file.path("R", "outputs.R"))
source(file.path("R", "solve_model.R"))
source(file.path("R", "scenarios.R"))
source(file.path("R", "optimizer.R"))

required_app_packages <- c("shiny", "bslib", "deSolve", "yaml", "readr", "dplyr", "tidyr", "purrr", "ggplot2", "scales")
missing_required <- required_app_packages[!vapply(required_app_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_required)) {
  stop("Missing required R packages: ", paste(missing_required, collapse = ", "),
       ". Run install_packages.R in the ShinyApp directory.", call. = FALSE)
}

suppressPackageStartupMessages({
  library(shiny)
  library(bslib)
  library(ggplot2)
  library(scales)
})

configs <- load_app_configs_r()
country_choices <- available_countries_r(configs)
has_plotly <- requireNamespace("plotly", quietly = TRUE)
has_dt <- requireNamespace("DT", quietly = TRUE)
hosted_shiny <- nzchar(Sys.getenv("SHINY_PORT")) || nzchar(Sys.getenv("R_CONFIG_ACTIVE"))
default_workers <- as.integer(Sys.getenv("SHINYAPP_N_JOBS", unset = NA_character_))
if (!is.finite(default_workers)) {
  default_workers <- if (.Platform$OS.type == "windows" || hosted_shiny) {
    1
  } else {
    min(4, max(1, parallel::detectCores(logical = FALSE) %||% 1))
  }
}

app_theme <- bs_theme(
  version = 5,
  bg = "#F4F1EA",
  fg = "#232323",
  primary = "#00468B",
  secondary = "#74787D",
  success = "#42B540",
  info = "#0099B4",
  warning = "#FDAF91",
  danger = "#ED0000"
)

# Copied from scripts_R/10_shared.R so ShinyApp remains deployable alone.
MANUSCRIPT_PALETTE_PRIMARY_9 <- c(
  "#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF",
  "#FDAF91FF", "#AD002AFF", "#ADB6B6FF", "#1B1919FF"
)
MANUSCRIPT_PALETTE_SECONDARY_9 <- c(
  "#18B587FF", "#99CCD9FF", "#FF0069FF", "#202B2AFF", "#2AC0E7FF",
  "#24576FFF", "#6E422CFF", "#F0926FFF", "#E59EC0FF"
)
MANUSCRIPT_COLOURS <- c(
  black = "#232323",
  grey = "#52575C",
  mid_grey = "#74787D",
  pale_grey = "#DCD7CE",
  light_grey = "#F4F1EA",
  blue = MANUSCRIPT_PALETTE_PRIMARY_9[[1]],
  green = MANUSCRIPT_PALETTE_PRIMARY_9[[3]],
  orange = MANUSCRIPT_PALETTE_PRIMARY_9[[6]],
  vermillion = MANUSCRIPT_PALETTE_PRIMARY_9[[2]],
  purple = MANUSCRIPT_PALETTE_PRIMARY_9[[5]]
)
MANUSCRIPT_DISCRETE_CORE <- c(
  deep_teal = MANUSCRIPT_PALETTE_PRIMARY_9[[4]],
  orange = MANUSCRIPT_PALETTE_PRIMARY_9[[6]],
  blue = MANUSCRIPT_PALETTE_PRIMARY_9[[1]],
  vermillion = MANUSCRIPT_PALETTE_PRIMARY_9[[2]],
  green = MANUSCRIPT_PALETTE_PRIMARY_9[[3]],
  purple = MANUSCRIPT_PALETTE_PRIMARY_9[[5]],
  grey = MANUSCRIPT_PALETTE_PRIMARY_9[[8]]
)

APP_PALETTE <- c(
  MANUSCRIPT_DISCRETE_CORE,
  sky = MANUSCRIPT_PALETTE_SECONDARY_9[[2]],
  magenta = MANUSCRIPT_PALETTE_SECONDARY_9[[3]],
  blue_dark = MANUSCRIPT_PALETTE_SECONDARY_9[[6]],
  brown = MANUSCRIPT_PALETTE_SECONDARY_9[[7]],
  lavender = MANUSCRIPT_PALETTE_SECONDARY_9[[9]],
  black = MANUSCRIPT_COLOURS[["black"]]
)

APP_STRATEGY_COLOURS <- c(
  current = MANUSCRIPT_COLOURS[["mid_grey"]],
  higher_child_coverage = MANUSCRIPT_COLOURS[["mid_grey"]],
  timeliness_only = MANUSCRIPT_DISCRETE_CORE[["blue"]],
  adolescent_booster = MANUSCRIPT_DISCRETE_CORE[["purple"]],
  pregnancy_tdap_scaleup = MANUSCRIPT_DISCRETE_CORE[["orange"]],
  cocooning_adjunct = MANUSCRIPT_DISCRETE_CORE[["deep_teal"]],
  maternal_immunization = MANUSCRIPT_DISCRETE_CORE[["green"]],
  targeted_pep_high_risk = MANUSCRIPT_DISCRETE_CORE[["vermillion"]],
  resistance_guided_treatment = MANUSCRIPT_DISCRETE_CORE[["vermillion"]],
  transmission_blocking_vaccine = MANUSCRIPT_DISCRETE_CORE[["deep_teal"]],
  next_generation_vaccine = MANUSCRIPT_DISCRETE_CORE[["blue"]],
  combined_strategy = MANUSCRIPT_COLOURS[["black"]]
)

make_plot_output <- function(id, height = "360px") {
  if (has_plotly) plotly::plotlyOutput(id, height = height) else plotOutput(id, height = height)
}

make_table_output <- function(id) {
  if (has_dt) DT::DTOutput(id) else tableOutput(id)
}

format_strategy_labels_r <- function(x) {
  labels <- unname(STRATEGY_LABELS_R[as.character(x)])
  labels[is.na(labels)] <- as.character(x)[is.na(labels)]
  labels
}

palette_for_values_r <- function(values) {
  values <- unique(as.character(values))
  strategy_label_colours <- setNames(
    unname(APP_STRATEGY_COLOURS[names(STRATEGY_LABELS_R)]),
    unname(STRATEGY_LABELS_R[names(STRATEGY_LABELS_R)])
  )
  mapped <- strategy_label_colours[values]
  missing <- is.na(mapped)
  if (any(missing)) mapped[missing] <- rep(APP_PALETTE, length.out = sum(missing))
  setNames(unname(mapped), values)
}

theme_app_plot <- function() {
  theme_minimal(base_size = 12) +
    theme(
      plot.background = element_rect(fill = "#ffffff", colour = NA),
      panel.background = element_rect(fill = "#ffffff", colour = NA),
      panel.grid.major = element_line(colour = "#DCD7CE", linewidth = 0.35),
      panel.grid.minor = element_blank(),
      axis.title = element_text(colour = "#52575C"),
      axis.text = element_text(colour = "#74787D"),
      legend.position = "bottom",
      legend.title = element_blank(),
      legend.key.height = unit(0.45, "cm"),
      plot.margin = margin(10, 16, 10, 10)
    )
}

strategy_choices <- setNames(DEFAULT_STRATEGIES_R, STRATEGY_LABELS_R[DEFAULT_STRATEGIES_R])

custom_age_input_ui <- function(template_config) {
  age_df <- profile_to_age_frame_r(template_config)
  tagList(lapply(seq_len(nrow(age_df)), function(i) {
    age <- age_df$age_group[[i]]
    div(
      class = "age-row",
      tags$strong(MODEL_AGE_LABELS[[age]] %||% age),
      numericInput(paste0("pop_", age), "Population", value = round(age_df$population[[i]]), min = 1, step = 1000),
      sliderInput(paste0("cov_", age), "Coverage", min = 0, max = 1, value = age_df$vaccine_coverage[[i]], step = 0.01),
      sliderInput(paste0("rep_", age), "Reporting", min = 0, max = 1, value = age_df$reporting_rate[[i]], step = 0.01)
    )
  }))
}

build_custom_config <- function(input, configs) {
  template <- input$template_country %||% "China"
  product <- input$custom_product %||% "aP"
  vaccine_scenario <- if (product == "wP") "infection_blocking" else "symptom_protective"
  config <- make_config_r(
    country_profile = template,
    vaccine_scenario = vaccine_scenario,
    resistance_scenario = "moderate",
    load_calibration = TRUE,
    configs = configs
  )
  config$country <- input$custom_country_name %||% "Custom_country"
  config$metadata <- config$metadata %||% list()
  config$metadata$iso3 <- input$custom_iso3 %||% "CUS"
  config$metadata$vaccine_product <- product
  config$metadata$template_country <- template

  age_profile <- NULL
  if (!is.null(input$age_profile_csv$datapath)) {
    age_profile <- validate_age_profile_r(read_csv_r(input$age_profile_csv$datapath))
  } else {
    age_profile <- data.frame(
      age_group = MODEL_AGE_GROUPS,
      population = vapply(MODEL_AGE_GROUPS, function(age) as.numeric(input[[paste0("pop_", age)]]), numeric(1)),
      vaccine_coverage = vapply(MODEL_AGE_GROUPS, function(age) as.numeric(input[[paste0("cov_", age)]]), numeric(1)),
      reporting_rate = vapply(MODEL_AGE_GROUPS, function(age) as.numeric(input[[paste0("rep_", age)]]), numeric(1)),
      symptom_probability = profile_to_age_frame_r(config)$symptom_probability,
      stringsAsFactors = FALSE
    )
    age_profile <- validate_age_profile_r(age_profile)
  }
  for (i in seq_len(nrow(age_profile))) {
    age <- age_profile$age_group[[i]]
    idx <- match(age, vapply(config$age_groups, `[[`, character(1), "label"))
    config$age_groups[[idx]]$population <- age_profile$population[[i]]
    config$age_groups[[idx]]$vaccine_coverage <- age_profile$vaccine_coverage[[i]]
    config$age_groups[[idx]]$reporting_rate <- age_profile$reporting_rate[[i]]
    config$age_groups[[idx]]$pep_detection_rate <- age_profile$reporting_rate[[i]]
    config$age_groups[[idx]]$symptom_probability <- age_profile$symptom_probability[[i]]
  }

  if (!is.null(input$contact_matrix_csv$datapath)) {
    mat <- contact_matrix_from_long_r(read_csv_r(input$contact_matrix_csv$datapath))
    config$contact_matrix$rows <- contact_rows_from_matrix_r(mat)
    config$contact_matrix$reciprocity_correction$enabled <- FALSE
  }

  resistance <- clip_probability_r(input$custom_resistance %||% 0.05)
  config$resistance$target_prevalence_at_analysis_start <- resistance
  config$resistance$importation_fraction <- resistance
  config$importation$resistant_fraction <- resistance
  config$initial_conditions$initial_resistance_prevalence <- resistance
  maternal_cov <- clip_probability_r(input$custom_maternal_coverage %||% 0)
  config$demography$birth_entry <- list(S = 1 - maternal_cov, V = maternal_cov)
  if (isTRUE(input$custom_adolescent_booster)) {
    config <- set_age_group_value_r(config, "adolescent_10_17y", "vaccine_coverage", 0.75, "min")
  }
  config
}

plot_time_series <- function(timeseries) {
  require_namespace_r("dplyr")
  annual <- timeseries |>
    dplyr::group_by(.data$scenario, .data$calendar_year) |>
    dplyr::summarise(
      symptomatic_cases = sum(.data$symptomatic_cases, na.rm = TRUE),
      infant_cases = sum(.data$infant_cases, na.rm = TRUE),
      .groups = "drop"
    )
  annual$strategy_label <- format_strategy_labels_r(annual$scenario)
  ggplot(annual, aes(.data$calendar_year, .data$infant_cases, colour = .data$strategy_label)) +
    geom_line(linewidth = 0.8) +
    geom_point(size = 1.6) +
    scale_colour_manual(values = palette_for_values_r(annual$strategy_label)) +
    scale_y_continuous(labels = label_comma()) +
    labs(x = NULL, y = "Infant symptomatic cases", colour = NULL) +
    theme_app_plot()
}

plot_age_burden <- function(timeseries) {
  require_namespace_r("dplyr")
  age <- timeseries |>
    dplyr::group_by(.data$scenario, .data$age_group) |>
    dplyr::summarise(symptomatic_cases = sum(.data$symptomatic_cases, na.rm = TRUE), .groups = "drop")
  age$age_group <- factor(age$age_group, levels = MODEL_AGE_GROUPS, labels = MODEL_AGE_LABELS[MODEL_AGE_GROUPS])
  age$strategy_label <- format_strategy_labels_r(age$scenario)
  ggplot(age, aes(.data$age_group, .data$symptomatic_cases, fill = .data$strategy_label)) +
    geom_col(position = "dodge", width = 0.72) +
    scale_fill_manual(values = palette_for_values_r(age$strategy_label)) +
    scale_y_continuous(labels = label_comma()) +
    labs(x = NULL, y = "Symptomatic cases", fill = NULL) +
    theme_app_plot() +
    theme(axis.text.x = element_text(angle = 35, hjust = 1))
}

plot_frontier <- function(frontier) {
  frontier$strategy_label <- frontier$strategy_label %||% format_strategy_labels_r(frontier$scenario)
  frontier$frontier_status <- ifelse(frontier$non_dominated, "Pareto frontier", "Dominated")
  ggplot(frontier, aes(.data$annualized_infant_cases_per_100k, .data$annualized_resistant_infections_per_100k)) +
    geom_point(aes(colour = .data$frontier_status, size = .data$implementation_intensity), alpha = 0.88) +
    geom_text(aes(label = .data$strategy_label), vjust = -0.75, size = 3, check_overlap = TRUE) +
    scale_colour_manual(values = c(`Pareto frontier` = MANUSCRIPT_DISCRETE_CORE[["deep_teal"]], Dominated = MANUSCRIPT_COLOURS[["mid_grey"]])) +
    scale_x_continuous(labels = label_comma()) +
    scale_y_continuous(labels = label_comma()) +
    labs(x = "Infant cases per 100k/year", y = "Resistant infections per 100k/year", colour = NULL, size = "Intensity") +
    theme_app_plot()
}

ui <- page_sidebar(
  tags$head(tags$link(rel = "stylesheet", type = "text/css", href = "styles.css")),
  title = div(
    class = "app-title",
    span(class = "app-title-main", "Pertussis Strategy Simulator")
  ),
  theme = app_theme,
  sidebar = sidebar(
    width = 420,
    div(
      class = "control-section",
      div(class = "control-section-title", icon("globe"), span("Country")),
      radioButtons(
        "country_mode",
        NULL,
        choices = c("Built-in profile" = "builtin", "Custom country" = "custom"),
        inline = TRUE
      ),
      conditionalPanel(
        "input.country_mode == 'builtin'",
        selectInput("country", "Country", choices = country_choices, selected = "China")
      ),
      conditionalPanel(
        "input.country_mode == 'custom'",
        div(
          class = "compact-grid two-col",
          textInput("custom_country_name", "Country name", value = "Custom_country"),
          textInput("custom_iso3", "ISO3", value = "CUS")
        ),
        div(
          class = "compact-grid two-col",
          selectInput("template_country", "Template", choices = country_choices, selected = "China"),
          selectInput("custom_product", "Product type", choices = c("aP", "wP", "DTP"), selected = "aP")
        ),
        sliderInput("custom_resistance", "Baseline resistance", min = 0, max = 1, value = 0.05, step = 0.01),
        sliderInput("custom_maternal_coverage", "Maternal coverage", min = 0, max = 1, value = 0.2, step = 0.01),
        checkboxInput("custom_adolescent_booster", "Adolescent booster programme", value = FALSE),
        accordion(
          class = "app-accordion",
          accordion_panel("Age profile", uiOutput("custom_age_inputs")),
          accordion_panel(
            "Uploads",
            fileInput("age_profile_csv", "Age profile CSV", accept = ".csv"),
            fileInput("contact_matrix_csv", "Contact matrix CSV", accept = ".csv")
          )
        )
      )
    ),
    div(
      class = "control-section",
      div(class = "control-section-title", icon("sliders"), span("Runtime")),
      div(
        class = "compact-grid two-col",
        selectInput("horizon", "Horizon", choices = c("2025-2035 fast" = "fast", "2025-2050 publication" = "publication"), selected = "fast"),
        selectInput(
          "solver_method",
          "Solver",
          choices = c("Fast RK4" = "rk4", "deSolve LSODA" = "lsoda"),
          selected = "rk4"
        )
      ),
      div(
        class = "compact-grid two-col",
        numericInput("n_jobs", "Parallel workers", value = default_workers, min = 1, max = 16, step = 1),
        numericInput("burn_in_years", "Burn-in years", value = 3, min = 0, max = 15, step = 1)
      )
    ),
    div(
      class = "control-section",
      div(class = "control-section-title", icon("list-check"), span("Strategies")),
      div(
        class = "strategy-grid",
        checkboxGroupInput("strategies", NULL, choices = strategy_choices, selected = DEFAULT_STRATEGIES_R)
      )
    ),
    div(
      class = "control-section",
      div(class = "control-section-title", icon("gauge-high"), span("Optimization")),
      accordion(
        class = "app-accordion",
        accordion_panel(
          "Weights",
          sliderInput("w_infant", "Infant cases", min = 0, max = 1, value = 0.55, step = 0.05),
          sliderInput("w_severe", "Deaths/hospitalizations", min = 0, max = 1, value = 0.20, step = 0.05),
          sliderInput("w_resistance", "Resistant infections", min = 0, max = 1, value = 0.15, step = 0.05),
          sliderInput("w_implementation", "Implementation intensity", min = 0, max = 1, value = 0.10, step = 0.05)
        ),
        accordion_panel(
          "Fast calibration",
          numericInput("target_incidence", "Target reported incidence per 100k/year", value = NA, min = 0, step = 1),
          checkboxInput("run_quick_calibration", "Run one-dimensional beta calibration", value = FALSE)
        )
      )
    ),
    actionButton("run", tagList(icon("play"), span("Run simulation")), class = "btn-primary run-button")
  ),
  div(
    class = "results-shell",
    div(
      class = "metric-strip",
      layout_columns(
        col_widths = c(3, 3, 3, 3),
        value_box(
          title = "Best strategy",
          value = textOutput("best_strategy", inline = TRUE),
          showcase = icon("trophy"),
          class = "metric-box"
        ),
        value_box(
          title = "Infant reduction",
          value = textOutput("best_infant", inline = TRUE),
          showcase = icon("baby"),
          class = "metric-box"
        ),
        value_box(
          title = "Resistance reduction",
          value = textOutput("best_resistance", inline = TRUE),
          showcase = icon("shield-halved"),
          class = "metric-box"
        ),
        value_box(
          title = "Score",
          value = textOutput("best_score", inline = TRUE),
          showcase = icon("chart-line"),
          class = "metric-box"
        )
      )
    ),
    div(
      class = "results-tabs",
      navset_card_tab(
        nav_panel(tagList(icon("table"), "Ranking"), make_table_output("ranking_table")),
        nav_panel(tagList(icon("chart-line"), "Time series"), make_plot_output("timeseries_plot")),
        nav_panel(tagList(icon("chart-column"), "Age burden"), make_plot_output("age_plot")),
        nav_panel(tagList(icon("bullseye"), "Frontier"), make_plot_output("frontier_plot")),
        nav_panel(tagList(icon("clipboard-list"), "Summary"), make_table_output("summary_table")),
        nav_panel(
          tagList(icon("download"), "Downloads"),
          div(
            class = "download-grid",
            downloadButton("download_summary", "scenario_summary.csv", icon = icon("download")),
            downloadButton("download_timeseries", "timeseries.csv", icon = icon("download")),
            downloadButton("download_ranking", "strategy_ranking.csv", icon = icon("download")),
            downloadButton("download_frontier", "pareto_frontier.csv", icon = icon("download")),
            downloadButton("download_config", "country_config.yaml", icon = icon("download"))
          )
        )
      )
    )
  )
)

server <- function(input, output, session) {
  result_store <- reactiveVal(NULL)
  config_store <- reactiveVal(NULL)

  output$custom_age_inputs <- renderUI({
    template <- input$template_country %||% "China"
    template_config <- make_config_r(template, load_calibration = TRUE, configs = configs)
    custom_age_input_ui(template_config)
  })

  observeEvent(input$run, {
    req(input$strategies)
    withProgress(message = "Preparing model", value = 0, {
      base_config <- if (identical(input$country_mode, "custom")) {
        build_custom_config(input, configs)
      } else {
        make_config_r(input$country, load_calibration = TRUE, configs = configs)
      }
      end_date <- if (identical(input$horizon, "publication")) "2050-12-31" else "2035-12-31"
      base_config <- set_interactive_runtime_r(
        base_config,
        analysis_start_date = "2025-01-01",
        analysis_end_date = end_date,
        burn_in_years = input$burn_in_years,
        output_time_step = if (identical(input$solver_method, "rk4")) 90 else 30,
        solver_method = input$solver_method,
        rk4_dt_analysis = if (identical(input$horizon, "publication")) 30 else 30,
        rk4_dt_burnin = 30
      )
      if (isTRUE(input$run_quick_calibration) && is.finite(input$target_incidence) && input$target_incidence > 0) {
        incProgress(0.05, detail = "Fast beta calibration")
        base_config <- quick_calibrate_beta_r(base_config, input$target_incidence, configs)
      }
      progress_fun <- function(detail, fraction) {
        if ((input$n_jobs %||% 1) <= 1) {
          incProgress(0.9 / max(length(input$strategies), 1), detail = detail)
        } else {
          incProgress(0.05, detail = "Running strategies in parallel")
        }
      }
      res <- run_strategy_set_r(base_config, input$strategies, configs, progress_fun, n_jobs = input$n_jobs)
      weights <- c(
        infant = input$w_infant,
        severe = input$w_severe,
        resistance = input$w_resistance,
        implementation = input$w_implementation
      )
      ranked <- rank_strategy_results_r(res$summary, weights)
      result_store(list(timeseries = res$timeseries, summary = res$summary, ranked = ranked))
      config_store(base_config)
      incProgress(1, detail = "Done")
    })
  }, ignoreInit = TRUE)

  best <- reactive({
    res <- result_store()
    if (is.null(res)) return(NULL)
    res$ranked$best
  })

  output$best_strategy <- renderText({
    b <- best()
    if (is.null(b)) return("Not run")
    b$strategy_label[[1]] %||% b$scenario[[1]]
  })
  output$best_infant <- renderText({
    b <- best()
    if (is.null(b)) return("--")
    percent(b$relative_reduction_infant_cases[[1]], accuracy = 0.1)
  })
  output$best_resistance <- renderText({
    b <- best()
    if (is.null(b)) return("--")
    percent(b$relative_reduction_resistant_infections[[1]], accuracy = 0.1)
  })
  output$best_score <- renderText({
    b <- best()
    if (is.null(b)) return("--")
    number(b$strategy_score[[1]], accuracy = 0.001)
  })

  render_table <- function(id, expr) {
    if (has_dt) {
      output[[id]] <- DT::renderDT({
        DT::datatable(
          expr(),
          class = "compact stripe hover",
          options = list(pageLength = 12, scrollX = TRUE, autoWidth = TRUE),
          rownames = FALSE
        )
      })
    } else {
      output[[id]] <- renderTable(expr(), striped = TRUE, bordered = TRUE)
    }
  }

  render_table("ranking_table", function() {
    res <- result_store()
    req(res)
    cols <- c("scenario", "strategy_label", "strategy_score", "relative_reduction_infant_cases",
              "relative_reduction_resistant_infections", "annualized_infant_cases_per_100k",
              "annualized_resistant_infections_per_100k", "implementation_intensity")
    res$ranked$scored[, intersect(cols, names(res$ranked$scored)), drop = FALSE]
  })
  render_table("summary_table", function() {
    res <- result_store()
    req(res)
    res$summary
  })

  if (has_plotly) {
    output$timeseries_plot <- plotly::renderPlotly({
      res <- result_store()
      req(res)
      plotly::ggplotly(plot_time_series(res$timeseries))
    })
    output$age_plot <- plotly::renderPlotly({
      res <- result_store()
      req(res)
      plotly::ggplotly(plot_age_burden(res$timeseries))
    })
    output$frontier_plot <- plotly::renderPlotly({
      res <- result_store()
      req(res)
      plotly::ggplotly(plot_frontier(res$ranked$frontier))
    })
  } else {
    output$timeseries_plot <- renderPlot({
      res <- result_store()
      req(res)
      plot_time_series(res$timeseries)
    })
    output$age_plot <- renderPlot({
      res <- result_store()
      req(res)
      plot_age_burden(res$timeseries)
    })
    output$frontier_plot <- renderPlot({
      res <- result_store()
      req(res)
      plot_frontier(res$ranked$frontier)
    })
  }

  output$download_summary <- downloadHandler(
    filename = function() "scenario_summary.csv",
    content = function(file) readr::write_csv(result_store()$summary, file)
  )
  output$download_timeseries <- downloadHandler(
    filename = function() "timeseries.csv",
    content = function(file) readr::write_csv(result_store()$timeseries, file)
  )
  output$download_ranking <- downloadHandler(
    filename = function() "strategy_ranking.csv",
    content = function(file) readr::write_csv(result_store()$ranked$scored, file)
  )
  output$download_frontier <- downloadHandler(
    filename = function() "pareto_frontier.csv",
    content = function(file) readr::write_csv(result_store()$ranked$frontier, file)
  )
  output$download_config <- downloadHandler(
    filename = function() "custom_country_config.yaml",
    content = function(file) write_yaml_r(config_store(), file)
  )
}

shinyApp(ui, server)
