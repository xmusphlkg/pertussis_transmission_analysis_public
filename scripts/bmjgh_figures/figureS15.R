source("scripts/bmjgh_figures/theme.R")
input_files <- c("outputs/core2026/07_manuscript/model_study_revision/results_revision/maternal_coverage_infant_outcomes.csv",
                 "outputs/core2026/02_sensitivity/appendix_age_mortality/mortality.csv")
data_a <- subset(read.csv(input_files[1]), route == "standalone" & maternal_coverage == .9 & horizon == 20,
                 select = c(country, horizon, maternal_baseline, maternal_coverage, tdap_coverage,
                            comparator, maternal_only_deaths_averted, deaths_q025, deaths_q975))
names(data_a)[names(data_a) == "comparator"] <- "maternal_expansion_strategy"
data_a$comparator <- "current"
data_b <- subset(read.csv(input_files[2]), contrast == "maternal_to_90")
stopifnot(nrow(data_a) == 8, !anyDuplicated(data_a$country), nrow(data_b) == 24,
          !anyDuplicated(data_b[c("country","mild_to_typical_risk")]), all(data_b$q025 > 0))
data_a$setting <- factor(gsub("_", " ", data_a$country), levels = rev(country_levels))
a <- ggplot(data_a, aes(maternal_only_deaths_averted, setting)) +
  geom_vline(xintercept = 0, linewidth = .35, colour = "#8C989F") +
  geom_linerange(aes(xmin = deaths_q025, xmax = deaths_q975), colour = horizon_colours["20"]) +
  geom_point(colour = horizon_colours["20"], size = 2.1) +
  scale_x_continuous(transform = scales::transform_pseudo_log(sigma = 1),
                     breaks = c(0, 1, 10, 100, 1000), labels = scales::label_comma()) +
  labs(x = "Infant deaths averted over 20 years (pseudo-log scale)", y = NULL,
       title = "A: Infant deaths averted by maternal expansion") +
  theme(axis.text = element_text(size = 8))
data_b$setting <- factor(gsub("_", " ", data_b$country), levels = country_levels)
b <- ggplot(data_b, aes(100 * mild_to_typical_risk, mean_deaths_averted)) +
  geom_line(linewidth = .42, colour = "#618399") +
  geom_linerange(aes(ymin = q025, ymax = q975), colour = horizon_colours["20"]) +
  geom_point(size = 1.8, colour = horizon_colours["20"]) +
  facet_wrap(~setting, ncol = 3, scales = "free_y") +
  scale_x_continuous(breaks = c(0,10,25,50)) +
  labs(x = "Mortality risk after mild disease (% of risk after typical disease)",
       y = "Infant deaths averted over 20 years", title = "B: Mortality risk after mild disease") +
  theme(axis.text = element_text(size = 8), panel.grid.major.x = element_blank())
caption <- paste("**Figure S15. Infant deaths averted by expanding maternal vaccination.**",
  "(A) Mean cumulative infant deaths averted by expanding maternal coverage from baseline to 90% over 20 years in eight settings, with 95% conditional intervals from 1,000 paired clinical-risk draws.",
  "(B) In the six settings with onset-based mortality estimates, points show means under four assumed mortality-risk ratios for mild versus typical disease. Intervals give the 2.5th and 97.5th percentiles of 4,096 mortality-risk draws on the same transmission trajectories. Vertical scales vary by setting. Both panels retain baseline adolescent Tdap coverage and exclude intranasal vaccination.")
plot <- (a / b) + plot_layout(heights = c(1, 1.4))
save_figure("S15", plot, list(A = a, B = b), list(A = data_a, B = data_b), caption, height = 186, width = 190)
