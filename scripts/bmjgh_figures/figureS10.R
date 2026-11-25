source("scripts/bmjgh_figures/theme.R")
d <- read.csv("outputs/core2026/07_manuscript/model_study_revision/results_revision/infant_outcomes_by_profile.csv")
d <- subset(d, route == "standalone" & nasal_coverage == .9 & comparator == "current")
ids <- jsonlite::fromJSON("source_data/figures/profile_order.json")$profile_ids
d$number <- match(d$profile_id, ids)
d$setting <- factor(gsub("_"," ",d$country, fixed = TRUE), levels = rev(country_levels))
d$window <- factor(d$horizon, levels = c(20,30), labels = c("20 years","30 years"))
stopifnot(!anyNA(d$number), all(table(d$setting,d$window) == 17), nrow(d) == 272)
d$all_three_lower <- with(d, reported_cases_averted > 0 & deaths_averted > 0 & yll_averted > 0)
key <- interaction(d$setting, d$window)
d$minimum <- ave(d$yll_averted, key, FUN = min)
d$median <- ave(d$yll_averted, key, FUN = median)
d$maximum <- ave(d$yll_averted, key, FUN = max)
d$n_lower_yll <- ave(as.integer(d$yll_averted > 0), key, FUN = sum)
d$n_lower_all_three <- ave(as.integer(d$all_three_lower), key, FUN = sum)
s <- unique(d[c("country","setting","horizon","window","minimum","median","maximum","n_lower_yll","n_lower_all_three")])
stopifnot(nrow(s) == 16, all(s$n_lower_all_three <= s$n_lower_yll))
a <- ggplot(d, aes(y = setting)) +
  geom_vline(xintercept = 0, colour = "#626F78", linetype = 2, linewidth = .35) +
  geom_segment(data = s, aes(x = minimum, xend = maximum, yend = setting), colour = "#A7A1B0", linewidth = .55) +
  geom_point(aes(x = yll_averted, y = as.integer(setting) + (number-9)/75),
             size = .85, alpha = .75, colour = "#8D819C") +
  geom_point(data = s, aes(x = median), size = 2, shape = 18, colour = "#442760") +
  facet_wrap(~window, nrow = 1) +
  scale_x_continuous(transform = scales::transform_pseudo_log(sigma = 1, base = 10),
                     breaks = c(-1000,-10,0,10,1000), labels = c("-1k","-10","0","10","1k")) +
  scale_y_discrete(drop = FALSE) +
  labs(title = "A: Infant YLL averted across vaccine profiles", x = "Discounted infant YLL averted", y = NULL) + bmj_theme()
b <- ggplot(s, aes(y = setting)) +
  geom_segment(aes(x = n_lower_all_three, xend = n_lower_yll, yend = setting), colour = "#6D7480",
               linewidth = .5) +
  geom_point(aes(x = n_lower_yll, shape = "YLL"), size = 2.6, stroke = .58,
             colour = bmj_outcome[["YLL"]]) +
  geom_point(aes(x = n_lower_all_three, shape = "Cases, deaths and YLL"), size = 1.45,
             colour = "#333C42") +
  scale_shape_manual(NULL, values = c("YLL" = 1, "Cases, deaths and YLL" = 16)) +
  scale_x_continuous(limits = c(-.6,17.6), breaks = c(0,4,8,12,17)) + facet_wrap(~window, nrow = 1) +
  labs(title = "B: Profiles with concordant infant benefits", x = "Profiles reducing the indicated outcomes (out of 17)", y = NULL) + bmj_theme()
fig <- a / b
caption <- paste("**Figure S10. Discounted infant years of life lost and concordance of infant outcomes across vaccine profiles.**",
  "**A**, Each small point shows mean infant years of life lost (YLL) averted under one of 17 profiles. Diamonds show the median across profiles and lines span their minimum and maximum on a symmetric pseudo-logarithmic scale.",
  "**B**, Numbers of profiles reducing discounted YLL (open circles) and numbers reducing reported cases, deaths and discounted YLL together (filled points). Overlapping symbols indicate equal counts.",
  "The profiles are unweighted. Comparisons use 90% standalone adolescent uptake relative to current provision, at baseline maternal and Tdap coverage.")
input_files <- c("outputs/core2026/07_manuscript/model_study_revision/results_revision/infant_outcomes_by_profile.csv",
                 "source_data/figures/profile_order.json")
save_figure("S10", fig, list(A = a, B = b), list(A = d, B = s), caption, height = 148)
