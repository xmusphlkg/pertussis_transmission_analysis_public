score_strategies_r <- function(summary,
                               weights = c(infant = 0.55, severe = 0.20, resistance = 0.15, implementation = 0.10)) {
  out <- summary
  weights <- weights / sum(weights)
  for (col in c(
    "relative_reduction_infant_cases",
    "relative_reduction_deaths",
    "relative_reduction_hospitalizations",
    "relative_reduction_resistant_infections"
  )) {
    if (!col %in% names(out)) out[[col]] <- 0
    out[[col]][is.na(out[[col]])] <- 0
  }
  out$implementation_intensity <- as.numeric(out$implementation_intensity %||% STRATEGY_INTENSITY_R[out$scenario] %||% 1)
  max_intensity <- max(out$implementation_intensity, na.rm = TRUE)
  if (!is.finite(max_intensity) || max_intensity <= 0) max_intensity <- 1
  severe_reduction <- rowMeans(cbind(out$relative_reduction_deaths, out$relative_reduction_hospitalizations), na.rm = TRUE)
  out$strategy_score <- weights[["infant"]] * out$relative_reduction_infant_cases +
    weights[["severe"]] * severe_reduction +
    weights[["resistance"]] * out$relative_reduction_resistant_infections -
    weights[["implementation"]] * (out$implementation_intensity / max_intensity)
  out$strategy_label <- out$strategy_label %||% unname(STRATEGY_LABELS_R[out$scenario])
  out[order(out$strategy_score, decreasing = TRUE), , drop = FALSE]
}

pareto_frontier_r <- function(summary) {
  data <- summary
  data$annualized_resistant_infections_per_100k <- as.numeric(data$annualized_resistant_infections_per_100k)
  data$annualized_infant_cases_per_100k <- as.numeric(data$annualized_infant_cases_per_100k)
  data$total_deaths <- as.numeric(data$total_deaths)
  dominated <- rep(FALSE, nrow(data))
  for (i in seq_len(nrow(data))) {
    for (j in seq_len(nrow(data))) {
      if (i == j) next
      no_worse <- data$annualized_infant_cases_per_100k[[j]] <= data$annualized_infant_cases_per_100k[[i]] &&
        data$annualized_resistant_infections_per_100k[[j]] <= data$annualized_resistant_infections_per_100k[[i]] &&
        data$total_deaths[[j]] <= data$total_deaths[[i]]
      strictly_better <- data$annualized_infant_cases_per_100k[[j]] < data$annualized_infant_cases_per_100k[[i]] ||
        data$annualized_resistant_infections_per_100k[[j]] < data$annualized_resistant_infections_per_100k[[i]] ||
        data$total_deaths[[j]] < data$total_deaths[[i]]
      if (isTRUE(no_worse && strictly_better)) {
        dominated[[i]] <- TRUE
        break
      }
    }
  }
  data$non_dominated <- !dominated
  data
}

rank_strategy_results_r <- function(summary, weights = c(infant = 0.55, severe = 0.20, resistance = 0.15, implementation = 0.10)) {
  scored <- score_strategies_r(summary, weights)
  frontier <- pareto_frontier_r(scored)
  list(scored = scored, frontier = frontier, best = scored[1, , drop = FALSE])
}
