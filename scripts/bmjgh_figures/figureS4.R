source("scripts/bmjgh_figures/theme.R")

input_files <- "outputs/core2026/07_manuscript/model_study_revision/results_revision/standard_expansions.csv"
d <- read.csv(input_files)
stopifnot(nrow(d) == 10, all(table(d$programme) == 5),
          max(abs(rowSums(d[c("infant_0","child_1_9","adolescent_10_19","adult_20plus")]) - d$all_ages)) < 1e-6)
ages <- c(infant_0 = "<1", child_1_9 = "1–9", adolescent_10_19 = "10–19", adult_20plus = "20+")
long <- tidyr::pivot_longer(d, cols = names(ages), names_to = "age", values_to = "cases_averted")
long$country <- factor(sub("_", " ", long$country), rev(country_levels))
long$age <- factor(ages[long$age], unname(ages))
limit <- max(abs(long$cases_averted))
p <- ggplot(long, aes(age, country, fill = cases_averted)) +
  geom_tile(colour = "white", linewidth = .8) +
  geom_text(aes(label = scales::comma(round(cases_averted)),
                colour = abs(asinh(cases_averted/20)) > asinh(limit/20)*.78),
            size = 7 / .pt, show.legend = FALSE) +
  scale_colour_manual(values = c("FALSE" = "#24333C", "TRUE" = "white")) +
  facet_wrap(~programme, scales = "free_y", nrow = 1) +
  scale_fill_gradient2(low = "#9A5045", mid = "white", high = "#26677C", midpoint = 0,
                       trans = scales::pseudo_log_trans(sigma = 10), limits = c(-limit, limit),
                       breaks = c(-1000, 0, 1000, 100000),
                       labels = c("−1k","0","1k","100k"),
                       guide = guide_colourbar(barwidth = grid::unit(40, "mm"))) +
  labs(title = "A: Age-specific case reductions after programme expansion", x = "Age (years)", y = NULL, fill = "Cases averted") +
  theme(panel.grid = element_blank(), axis.line = element_blank(), axis.ticks = element_blank())
caption <- paste("**Figure S4. Changes in reported cases at different ages after expansion of a standard programme.**",
  "Each intervention raises either maternal or adolescent Tdap coverage to 70% and leaves the other programme at baseline. Each facet includes the five settings eligible for the respective expansion.",
  "Cells give the 20-year cumulative reduction in reported cases. Negative values indicate additional cases. The common fill scale is centred on zero and compresses large absolute changes. Table S38 gives all-age totals and infant health outcomes.")
save_figure("S4", p, list(A = p), list(A = as.data.frame(long)), caption, height = 99)
