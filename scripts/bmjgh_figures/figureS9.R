source("scripts/bmjgh_figures/theme.R")
d <- read.csv("outputs/core2026/07_manuscript/model_study_revision/results_revision/infant_outcomes_by_profile.csv")
d <- subset(d, route == "standalone" & nasal_coverage == .9 & comparator == "current")
ids <- jsonlite::fromJSON("source_data/figures/profile_order.json")$profile_ids
d$profile_number <- match(d$profile_id, ids)
d$setting <- factor(gsub("_"," ",d$country, fixed = TRUE), levels = country_levels)
d$window <- factor(d$horizon, levels = c(20,30), labels = c("20 years","30 years"))
stopifnot(!anyNA(d$profile_number), all(table(d$setting,d$window) == 17), nrow(d) == 272)
index <- interaction(d$setting, d$window)
d$profile_minimum <- ave(d$deaths_averted, index, FUN = min)
d$profile_median <- ave(d$deaths_averted, index, FUN = median)
d$profile_maximum <- ave(d$deaths_averted, index, FUN = max)
a <- ggplot(d, aes(profile_number, deaths_averted)) +
  geom_vline(xintercept = 5, colour = "#E1E6E9", linewidth = 2.8) +
  geom_hline(yintercept = 0, linetype = 2, linewidth = .3, colour = "#77838A") +
  geom_point(aes(shape = window, x = profile_number + ifelse(horizon == 20,-.14,.14)),
             colour = bmj_outcome[["Deaths"]], stroke = .48, size = 1.6) +
  facet_wrap(~setting, scales = "free_y", ncol = 2) +
  scale_x_continuous(breaks = c(1,5,9,13,17), labels = c("01","05","09","13","17")) +
  scale_shape_manual(NULL, values = c("20 years"=16,"30 years"=2)) +
  labs(title = "A: Infant deaths averted across vaccine profiles", x = "Vaccine profile", y = "Mean cumulative infant deaths averted") +
  bmj_theme() + theme(panel.grid.major.x = element_blank())
caption <- paste("**Figure S9. Infant deaths averted by each adolescent intranasal vaccine profile.**",
  "Setting-specific panels give the mean number of deaths averted among infants younger than one year at 20 and 30 years from 2027.",
  "All comparisons use 90% standalone uptake with baseline maternal and Tdap coverage. Profile numbers follow Table S19 and the shaded column marks reference profile 05.",
  "Positive values indicate fewer infant deaths. Vertical scales differ between settings.")
input_files <- c("outputs/core2026/07_manuscript/model_study_revision/results_revision/infant_outcomes_by_profile.csv",
                 "source_data/figures/profile_order.json")
save_figure("S9", a, list(A = a), list(A = d), caption, height = 178)
