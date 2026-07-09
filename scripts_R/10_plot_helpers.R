## Shared plotting helpers -----------------------------------------------------
## Keep small, reusable utilities here so figure scripts can focus on their
## panel-specific data transformations and ggplot layers.

read_output_table <- function(name) {
  readr::read_csv(model_path("outputs", "tables", name), show_col_types = FALSE)
}

read_output_summary <- function(name) {
  readr::read_csv(model_path("outputs", "summaries", name), show_col_types = FALSE)
}

read_csv_local <- function(...) {
  readr::read_csv(model_path(...), show_col_types = FALSE)
}

read_table <- read_output_table
read_summary <- read_output_summary

require_columns <- function(df, cols, dataset_name) {
  missing_cols <- setdiff(cols, names(df))
  if (length(missing_cols) > 0) {
    stop(
      dataset_name, " is missing required column(s): ",
      paste(missing_cols, collapse = ", "),
      call. = FALSE
    )
  }
  invisible(df)
}

required_columns <- require_columns

positive_rate <- function(x) {
  is.finite(x) & !is.na(x) & x > 0
}

pct_label <- function(x) {
  lancet_percent(x, accuracy = 1)
}

parse_range_bounds <- function(x) {
  numeric_token <- "([+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+))"
  pattern <- paste0(
    "^\\s*", numeric_token,
    "\\s*(?:-|to)\\s*",
    numeric_token, "\\s*$"
  )
  match <- stringr::str_match(x, pattern)
  tibble::tibble(
    lower = as.numeric(match[, 2]),
    upper = as.numeric(match[, 3])
  )
}

bounded_proportional_response <- function(current, scenario) {
  denominator <- pmax(current, scenario, na.rm = FALSE)
  dplyr::case_when(
    !is.finite(current) | !is.finite(scenario) ~ NA_real_,
    denominator > 0 ~ (current - scenario) / denominator,
    TRUE ~ 0
  )
}

vaccine_profile_spec <- function(style = c("short", "mechanism")) {
  style <- match.arg(style)
  target_levels <- c("infection_blocking", "transmission_blocking", "next_generation")
  display_levels <- c("symptom_protective", target_levels)

  if (style == "mechanism") {
    labels <- c(
      no_vaccine = "No vaccine",
      symptom_protective = "Current aP-like mechanism scenario",
      infection_blocking = "Infection-blocking mechanism",
      transmission_blocking = "Transmission-blocking mechanism",
      next_generation = "High transmission-blocking mechanism"
    )
    axis_labels <- c(
      "No vaccine" = "No vaccine",
      "Current aP-like mechanism scenario" = "Current aP-like\nmechanism scenario",
      "Infection-blocking mechanism" = "Infection-blocking\nmechanism",
      "Transmission-blocking mechanism" = "Transmission-blocking\nmechanism",
      "High transmission-blocking mechanism" = "High transmission-\nblocking mechanism",
      "Combined future upper-bound scenario" = "Combined future\nupper-bound scenario"
    )
    residual_label <- "Combined future upper-bound scenario"
    colours <- c(
      "Current aP-like mechanism scenario" = manuscript_vaccine_profile_colours[["Current aP"]],
      "Infection-blocking mechanism" = manuscript_vaccine_profile_colours[["Inf.-blocking"]],
      "Transmission-blocking mechanism" = manuscript_vaccine_profile_colours[["Trans.-blocking"]],
      "High transmission-blocking mechanism" = manuscript_vaccine_profile_colours[["Upper-bound"]],
      "Combined future upper-bound scenario" = manuscript_colour("black")
    )
  } else {
    labels <- c(
      no_vaccine = "No vaccine",
      symptom_protective = "Current aP",
      infection_blocking = "Inf.-blocking",
      transmission_blocking = "Trans.-blocking",
      next_generation = "High-blocking"
    )
    axis_labels <- c(
      "No vaccine" = "No vaccine",
      "Current aP" = "Current acellular-\npertussis-like",
      "Inf.-blocking" = "Infection-\nblocking",
      "Trans.-blocking" = "Transmission-\nblocking",
      "High-blocking" = "High-blocking",
      "Combined stress test" = "Combined future\nstress test"
    )
    residual_label <- "Combined stress test"
    colours <- c(
      "Current aP" = manuscript_vaccine_profile_colours[["Current aP"]],
      "Inf.-blocking" = manuscript_vaccine_profile_colours[["Inf.-blocking"]],
      "Trans.-blocking" = manuscript_vaccine_profile_colours[["Trans.-blocking"]],
      "High-blocking" = manuscript_vaccine_profile_colours[["Upper-bound"]],
      "Combined stress test" = manuscript_colour("black")
    )
  }

  display_labels <- unname(labels[display_levels])
  legend_levels <- c(display_labels, residual_label)

  list(
    labels = labels,
    axis_labels = axis_labels,
    target_levels = target_levels,
    display_levels = display_levels,
    display_labels = display_labels,
    residual_display_levels = legend_levels,
    legend_levels = legend_levels,
    colours = colours
  )
}

scale_colour_vaccine_profile <- function(values, levels, guide = "none") {
  scale_colour_manual(
    values = values,
    limits = unname(levels),
    breaks = unname(levels),
    drop = FALSE,
    name = "Vaccine profile",
    guide = guide
  )
}

scale_shape_profile_median <- function(name = "Estimate",
                                       profile_label = "Profiles",
                                       median_label = "Median",
                                       profile_shape = 16,
                                       median_shape = 23,
                                       profile_colour = manuscript_colour("mid_grey"),
                                       median_colour = manuscript_colour("black"),
                                       profile_fill = profile_colour,
                                       median_fill = median_colour,
                                       profile_alpha = 0.65,
                                       median_alpha = 1,
                                       profile_size = 1.5,
                                       median_size = 2.2,
                                       profile_stroke = 0,
                                       median_stroke = 0.2,
                                       keywidth = unit(0.36, "cm"),
                                       keyheight = unit(0.46, "cm"),
                                       ncol = 1,
                                       title.position = "top",
                                       title.hjust = 0,
                                       ...) {
  labels <- c(profile_label, median_label)
  scale_shape_manual(
    values = setNames(c(profile_shape, median_shape), labels),
    breaks = labels,
    name = name,
    guide = guide_legend(
      ncol = ncol,
      byrow = TRUE,
      title.position = title.position,
      title.hjust = title.hjust,
      keywidth = keywidth,
      keyheight = keyheight,
      override.aes = list(
        colour = c(profile_colour, median_colour),
        fill = c(profile_fill, median_fill),
        alpha = c(profile_alpha, median_alpha),
        size = c(profile_size, median_size),
        stroke = c(profile_stroke, median_stroke)
      ),
      ...
    )
  )
}

scale_linetype_iqr <- function(label = "Cross-profile\nIQR",
                               name = "Interval",
                               value = "solid",
                               colour = manuscript_colour("grey"),
                               linewidth = 1.05,
                               alpha = 0.74,
                               order = 2,
                               keywidth = unit(0.52, "cm"),
                               keyheight = unit(0.28, "cm"),
                               title.position = "top",
                               title.hjust = 0,
                               ...) {
  scale_linetype_manual(
    values = c("IQR" = value),
    breaks = "IQR",
    labels = c("IQR" = label),
    name = name,
    guide = guide_legend(
      order = order,
      title.position = title.position,
      title.hjust = title.hjust,
      keywidth = keywidth,
      keyheight = keyheight,
      override.aes = list(
        colour = colour,
        linewidth = linewidth,
        alpha = alpha
      ),
      ...
    )
  )
}
