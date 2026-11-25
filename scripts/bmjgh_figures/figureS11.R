source("scripts/bmjgh_figures/theme.R")
input_files <- "outputs/core2026/02_sensitivity/mucosal_vaccines/matched_refit_followup/matched_refit_summary.csv"
d <- read.csv(input_files)
d$country <- factor(gsub("_", " ", d$country), levels = rev(country_levels))
d$recovery <- factor(d$history, levels = c("baseline", "recovery_365"), labels = c("Baseline recovery", "365-day recovery"))
d$window <- factor(paste(d$horizon, "years"), levels = c("20 years", "30 years"))
stopifnot(nrow(d) == 32, !anyDuplicated(d[c("country", "horizon", "history")]), sum(d$status == "matched") == 25)
yes <- subset(d, status == "matched")
no <- subset(d, status != "matched")
p <- ggplot(yes, aes(y = country, x = onward_minus_acquisition_dalys)) +
  geom_vline(xintercept = 0, colour = "#909A9F", linewidth = .3) +
  geom_linerange(aes(xmin = onward_minus_acquisition_dalys_q025,
                     xmax = onward_minus_acquisition_dalys_q975), linewidth = .58, colour = "#687985") +
  geom_point(colour = bmj_mechanism[["Onward"]], size = 2.1) +
  geom_text(data = no, aes(x = 0, label = "No match"), hjust = -.12, colour = "#52606B",
            size = 6.7, size.unit = "pt") +
  facet_grid(window ~ recovery) +
  scale_x_continuous(transform = scales::pseudo_log_trans(sigma = .8, base = 10),
                     breaks = c(0, 10, 100, 1000), labels = scales::label_comma()) +
  labs(title = "A: Matched health gains under alternative recovery histories", x = "Additional DALYs averted under onward protection at the matched adolescent outcome", y = NULL)
caption <- paste("**Figure S11. Matching adolescent case reductions under two fitted recovery histories.**",
  "Points show the mean difference in disability-adjusted life-years (DALYs) averted by onward versus acquisition protection after matching on adolescent reported cases within each fitted history and time horizon.",
  "The intervals give the 2.5th and 97.5th percentiles across 1,000 paired clinical draws within each fit; mortality covers all ages in Brazil and Thailand and infants elsewhere.",
  "Twenty-five of 32 conditions supported a positive common target. The other seven had no positive full-strength endpoint overlap. The horizontal axis is approximately linear near zero and logarithmic at larger magnitudes. The two fitted histories are unweighted structural scenarios.")
save_figure("S11", p, list(A = p), list(A = d), caption, height = 125)
