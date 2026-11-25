source("scripts/bmjgh_figures/theme.R")
folder <- "outputs/core2026/02_sensitivity/mucosal_vaccines/bpze1_policy_20261107/transmission_followup"
draw_file <- "source_data/figures/S13_draws.csv"
draws <- read.csv(draw_file)
published <- subset(read.csv(file.path(folder, "mechanism_contrasts.csv")), fit_group == "local_pooled" &
                      burden == "reported_uniform" & mortality == "included_base" & contrast == "onward_minus_acquisition")
stopifnot(nrow(draws) == 6000, nrow(published) == 4, all(table(draws$country, draws$horizon) == 1500),
          !anyDuplicated(draws[c("country", "particle", "clinical_draw", "horizon")]))
for (i in seq_len(nrow(published))) {
  r <- published[i, ]
  observed <- subset(draws, country == r$country & horizon == r$horizon)$additional_dalys
  stopifnot(max(abs(c(mean(observed), quantile(observed, c(.025, .975))) -
                      c(r$dalys_difference_mean, r$dalys_difference_q025, r$dalys_difference_q975))) < 1e-7)
}
particles <- aggregate(additional_dalys ~ country + horizon + particle, draws, mean)
draws$country <- factor(draws$country, levels = c("England", "Brazil"))
particles$country <- factor(particles$country, levels = levels(draws$country))
published$country <- factor(published$country, levels = levels(draws$country))
period <- function(x) factor(paste(x, "years"), levels = c("20 years", "30 years"))
draws$window <- period(draws$horizon)
particles$window <- period(particles$horizon)
published$window <- period(published$horizon)
pA <- ggplot(draws, aes(additional_dalys, colour = country)) +
  geom_vline(xintercept = 0, colour = "#8A939B", linewidth = .35) +
  geom_step(aes(y = after_stat(y)), stat = "ecdf", linewidth = .68) +
  facet_grid(window ~ country) +
  scale_colour_manual(values = bmj_country, guide = "none") +
  scale_y_continuous(breaks = c(0, .5, 1), labels = scales::label_percent()) +
  labs(title = "A: Distribution of paired health gains",
       x = "Additional DALYs averted (onward minus acquisition)", y = "Cumulative share of paired draws")
pB <- ggplot(particles, aes(additional_dalys, window, colour = country)) +
  geom_vline(xintercept = 0, colour = "#8A939B", linewidth = .35) +
  geom_point(position = position_jitter(width = 0, height = .11, seed = 25), size = 1.4, alpha = .70) +
  geom_linerange(data = published, aes(x = dalys_difference_mean, xmin = dalys_difference_q025,
                                       xmax = dalys_difference_q975, y = as.numeric(window) - .22),
                 linewidth = .56, colour = "#41474C") +
  geom_point(data = published, aes(x = dalys_difference_mean, y = as.numeric(window) - .22),
             shape = 18, size = 2.4, colour = "#41474C") +
  facet_grid(~country) + scale_colour_manual(values = bmj_country, guide = "none") +
  labs(title = "B: Health gains across transmission parameter sets",
       x = "Additional DALYs averted (onward minus acquisition)", y = NULL)
caption <- paste("**Figure S13. Local transmission uncertainty at fixed adolescent uptake in England and Brazil.**",
  "(A) Empirical cumulative distributions of the difference in disability-adjusted life-years (DALYs) averted under onward versus acquisition protection. Each setting contributes 30 compatible local transmission parameter sets, each crossed with 50 paired clinical and cost draws.",
  "(B) Coloured points are the 30 transmission-specific means. The diamond and line show the mean and empirical 2.5th to 97.5th percentiles across all 1,500 paired draws per setting and horizon. Mortality covers infants in England and all ages in Brazil; morbidity covers all ages in both.",
  "Both mechanisms use the same reference strength, three-year duration, 70% standalone uptake and baseline maternal and adolescent Tdap programmes. The adolescent case reductions are unconstrained.")
plot <- (pA / pB) + plot_layout(heights = c(1.8, 1))
panel_b_data <- merge(particles, published[, c("country", "window", "dalys_difference_mean",
                             "dalys_difference_q025", "dalys_difference_q975")], by = c("country", "window"))
input_files <- c(draw_file, file.path(folder, "mechanism_contrasts.csv"))
save_figure("S13", plot, list(A = pA, B = pB), list(A = draws, B = panel_b_data), caption, height = 137)
