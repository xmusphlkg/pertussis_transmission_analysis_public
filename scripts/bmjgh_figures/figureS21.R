source("scripts/bmjgh_figures/theme.R")

input_files <- c("outputs/core2026/02_sensitivity/review_round3/validation_windows.csv",
                 "outputs/core2026/japan_2025_followup_20261104/window_metrics.csv")
v <- read.csv(input_files[1], na.strings = c("", "Not available"))
v <- subset(v, !is.na(later_predicted_observed))
rapid <- subset(read.csv(input_files[2]), window == "2025_rapid")
stopifnot(nrow(v) == 6, nrow(rapid) == 1)
d <- data.frame(country = v$country, year = as.character(v$later_window),
                ratio = v$later_predicted_observed, delay = v$peak_delay_days)
d <- rbind(d, data.frame(country = "Japan", year = "2025 (rapid reports)",
                         ratio = rapid$expected_observed_ratio, delay = rapid$peak_midpoint_delay_days))
d$country <- factor(sub("_", " ", d$country), country_levels)
d <- d[order(d$country, d$year), ]
d$setting <- factor(paste(d$country, d$year, sep = ", "), rev(paste(d$country, d$year, sep = ", ")))
a <- ggplot(d, aes(ratio, setting)) +
  geom_vline(xintercept = 1, linewidth = .4, linetype = "dashed", colour = "#777D82") +
  geom_segment(aes(x = 1, xend = ratio, yend = setting), linewidth = .75, colour = "#B6C7D0") +
  geom_point(colour = bmj_outcome[["Cases"]], size = 2.3) +
  scale_x_log10(breaks = c(.2, .5, 1, 2, 4), labels = c("0.2", "0.5", "1", "2", "4")) +
  labs(title = "A: Predicted and observed case totals", x = "Predicted / observed", y = NULL)
b <- ggplot(d, aes(delay, setting)) +
  geom_vline(xintercept = 0, linewidth = .4, linetype = "dashed", colour = "#777D82") +
  geom_segment(aes(x = 0, xend = delay, yend = setting), linewidth = .75, colour = "#B6C7D0") +
  geom_point(colour = bmj_outcome[["Cases"]], size = 2.3) +
  scale_x_continuous(breaks = c(-180, -90, 0, 90, 180)) +
  labs(title = "B: Difference in peak timing", x = "Predicted minus observed peak (days)", y = NULL)
figure <- a | b
caption <- paste("**Figure S21. Case totals and peak timing in later surveillance.**",
  "**A**, Ratio of predicted to observed cases in each surveillance year excluded from fitting.",
  "**B**, Difference between the midpoints of the predicted and observed peak reporting intervals. Positive values denote later predicted peaks.",
  "Japan’s 2025 rapid-report series is shown separately from its 2024 surveillance comparison. At least one peak occurred at the window edge in each comparison except Japan in 2025. Australia and England had no later surveillance year in this analysis.")
save_figure("S21", figure, list(A = a, B = b), list(A = d, B = d), caption, height = 91)
