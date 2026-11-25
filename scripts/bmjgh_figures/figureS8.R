source("scripts/bmjgh_figures/theme.R")
base <- "outputs/core2026/07_manuscript/model_study_revision/results_revision/infant_outcomes_reference.csv"
long <- "outputs/core2026/02_sensitivity/review_revision"
sites <- c("Brazil","Japan")
annual_paths <- file.path(long, paste0("annual_",sites,"_adopted_unchanged.csv"))
total_paths <- file.path(long, paste0("long_totals_",sites,"_adopted_unchanged.csv"))
input_files <- c(base, annual_paths, total_paths)
d <- subset(read.csv(base), route == "standalone" & comparator == "current" & nasal_coverage == .9)
keys <- c("country","profile_id","comparator","nasal_coverage","tdap_coverage","maternal_coverage")
pair <- merge(subset(d, horizon == 20), subset(d, horizon == 30), by = keys, suffixes = c("_20","_30"))
fields <- c(Cases = "reported_cases_averted", Deaths = "deaths_averted", YLL = "yll_averted")
diff <- do.call(rbind, lapply(names(fields), function(metric) {
  f <- fields[[metric]]
  ans <- pair[keys]
  ans$measure <- metric
  ans$averted_20y <- pair[[paste0(f,"_20")]]
  ans$averted_30y <- pair[[paste0(f,"_30")]]
  ans$averted_years21_30 <- ans$averted_30y - ans$averted_20y
  ans
}))
diff$measure <- factor(diff$measure, levels = names(fields))
diff$setting <- factor(gsub("_", " ", diff$country, fixed = TRUE), levels = rev(country_levels))
stopifnot(nrow(pair) == 8, nrow(diff) == 24)
annual <- subset(do.call(rbind, lapply(annual_paths, read.csv)), metric == "deaths" & age_group == "infant_0")
annual <- annual[order(annual$country,annual$year), ]
annual$horizon <- annual$year - 2026
annual$cumulative_deaths <- ave(annual$averted, annual$country, FUN = cumsum)
totals <- subset(do.call(rbind, lapply(total_paths, read.csv)), age_group == "infant_0")
ck <- merge(annual, totals, by = c("country","horizon"))
stopifnot(all(table(annual$country) == 50), nrow(ck) == 10,
          max(abs(ck$deaths_averted - ck$cumulative_deaths)) < 1e-7)

a <- ggplot(diff, aes(averted_years21_30, setting, colour = measure)) +
  geom_vline(xintercept = 0, colour = "#626F78", linetype = 2, linewidth = .35) +
  geom_point(size = 1.8, show.legend = FALSE) + facet_wrap(~measure, nrow = 1, scales = "free_x") +
  scale_x_continuous(transform = scales::transform_pseudo_log(sigma = 1), breaks = c(-1000,-10,0,10,1000),
                     labels = c("-1k","-10","0","10","1k")) +
  scale_colour_manual(values = bmj_outcome) +
  labs(title = "A: Additional infant benefits in years 21–30", x = "Additional outcome averted in years 21–30", y = NULL) + bmj_theme()
b <- ggplot(annual, aes(year, averted)) +
  geom_hline(yintercept = 0, colour = "#697782", linewidth = .32) +
  geom_segment(aes(xend = year, y = 0, yend = averted), colour = bmj_outcome[["Deaths"]], linewidth = .5) +
  geom_point(size = .65, colour = bmj_outcome[["Deaths"]]) +
  geom_vline(xintercept = c(2046.5,2056.5), linetype = 3, colour = "#697782", linewidth = .3) +
  facet_wrap(~country, nrow = 1) +
  labs(title = "B: Annual infant deaths averted", x = "Calendar year", y = "Infant deaths averted per year") + bmj_theme()
cplot <- ggplot(annual, aes(year, cumulative_deaths)) +
  geom_hline(yintercept = 0, colour = "#697782", linewidth = .32) +
  geom_step(colour = bmj_outcome[["Deaths"]], linewidth = .45) +
  geom_linerange(data = ck, aes(ymin = deaths_averted_q025, ymax = deaths_averted_q975),
                 colour = "#3A454C", linewidth = .4) +
  geom_point(data = ck, size = 1.3, colour = "#3A454C") +
  geom_vline(xintercept = c(2046.5,2056.5), linetype = 3, colour = "#697782", linewidth = .3) +
  facet_wrap(~country, nrow = 1) +
  labs(title = "C: Cumulative infant deaths averted", x = "End of calendar year", y = "Cumulative infant deaths averted") + bmj_theme()
fig <- wrap_plots(a, b, cplot,
                  ncol = 1, heights = c(1.04,1,1))
caption <- paste("**Figure S8. Additional burden averted in the third decade and annual infant mortality in Brazil and Japan.**",
  "**A**, Difference between 30-year and 20-year cumulative infant reported cases, deaths and discounted YLL averted, equivalent to the gain in years 21–30. Each horizontal axis uses a symmetric pseudo-logarithmic scale.",
  "**B**, Annual differences in projected infant deaths at the end of each model year in the adopted Brazilian and Japanese histories. Dotted lines mark the ends of the 20-year and 30-year windows.",
  "**C**, Running sums of annual mean deaths averted; points and lines at 10-year intervals give the means and 95% conditional intervals of cumulative deaths from 1,000 paired clinical draws.",
  "All panels compare reference profile 05 at 90% standalone uptake with current provision at baseline maternal and Tdap coverage. Deaths are undiscounted. Positive values denote fewer events or less health loss.")
data_c <- merge(annual, totals[c("country","horizon","deaths_averted","deaths_averted_q025","deaths_averted_q975")],
                by = c("country","horizon"), all.x = TRUE)
save_figure("S8", fig, list(A = a, B = b, C = cplot), list(A = diff, B = annual, C = data_c),
            caption, height = 170)
