source("scripts/bmjgh_figures/theme.R")

input_files <- "outputs/core2026/current_model_accuracy_20261103/period_predictions.csv.gz"
d <- subset(read.csv(input_files), role == "training")
d$country <- factor(sub("_", " ", d$country), country_levels)
stopifnot(all(table(d$country) > 0), all(d$expected_cases >= 0), all(d$observed_cases >= 0),
          !anyDuplicated(d[c("country", "period_start")]))
frequency <- c(Australia = "monthly", England = "annual / quarterly", Sweden = "monthly",
               Japan = "weekly", "New Zealand" = "monthly", China = "monthly",
               Brazil = "monthly", Thailand = "monthly")
d$setting <- factor(paste0(d$country, " (", frequency[as.character(d$country)], ")"),
                    paste0(country_levels, " (", frequency[country_levels], ")"))
p <- ggplot(d, aes(observed_cases, expected_cases)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", linewidth = .4, colour = "#747B80") +
  geom_point(size = 1.35, alpha = .62, colour = bmj_outcome[["Cases"]]) +
  facet_wrap(~setting, ncol = 4) + coord_equal() +
  scale_x_continuous(trans = "log1p", breaks = c(0, 10, 1000, 100000),
                     labels = c("0", "10", "1k", "100k")) +
  scale_y_continuous(trans = "log1p", breaks = c(0, 10, 1000, 100000),
                     labels = c("0", "10", "1k", "100k")) +
  labs(title = "A: Fitted and observed cases", x = "Observed cases per reporting interval", y = "Fitted cases per reporting interval")
caption <- paste("**Figure S20. Agreement between fitted and observed surveillance counts.**",
                 "Each point represents one surveillance interval included in fitting. The diagonal marks agreement on identical log(1 + cases) axes, which retain zero counts. Figure S18 shows the timing of these observations.")
save_figure("S20", p, list(A = p), list(A = d), caption, height = 110)
