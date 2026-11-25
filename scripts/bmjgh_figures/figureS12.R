source("scripts/bmjgh_figures/theme.R")
root <- "outputs/core2026"
input_files <- c(file.path(root, "07_manuscript/model_study_revision/final_methodological_audit/matching/Japan/scan.csv"),
                 file.path(root, "02_sensitivity/mucosal_vaccines/matching_geometry", c("branch_roots.csv", "branch_comparisons.csv")))
curve <- subset(read.csv(input_files[1]), country == "Japan" & horizon == 30)
roots <- subset(read.csv(input_files[2]), country == "Japan" & horizon == 30)
comparison <- subset(read.csv(input_files[3]), country == "Japan" & horizon == 30)
stopifnot(nrow(roots) == 3, nrow(comparison) == 2, length(unique(roots$target_cases)) == 1,
          all(abs(roots$achieved_cases - roots$target_cases) < .001))
mechanism <- function(x) factor(x, levels = c("acquisition", "onward"), labels = c("Acquisition", "Onward"))
curve$mechanism <- mechanism(curve$mechanism)
roots$mechanism <- mechanism(roots$mechanism)
pA <- ggplot(curve, aes(strength, adolescent_cases_averted, colour = mechanism)) +
  geom_hline(yintercept = 0, linewidth = .3, colour = "#A0A9AE") +
  geom_hline(yintercept = roots$target_cases[1], linewidth = .4, colour = "#50585E", linetype = 2) +
  geom_line(linewidth = .65) +
  geom_point(data = roots, aes(y = achieved_cases), size = 2.2) +
  scale_colour_manual(NULL, values = bmj_mechanism) +
  labs(title = "A: Alternative matching solutions in Japan", x = "Protected-state log-effect strength", y = "Adolescent cases averted over 30 years")
metrics <- c("Infant cases" = "infant_cases", "Infant deaths" = "infant_deaths", "Infant YLL" = "infant_yll")
outcomes <- do.call(rbind, lapply(names(metrics), function(metric) {
  key <- paste0("difference_", metrics[[metric]])
  data.frame(target_cases = comparison$target_cases, onward_strength = comparison$onward_strength,
             acquisition_strength = comparison$acquisition_strength, outcome = metric,
             difference = comparison[[key]], q025 = if (metric == "Infant cases") NA_real_ else comparison[[paste0(key, "_q025")]],
             q975 = if (metric == "Infant cases") NA_real_ else comparison[[paste0(key, "_q975")]])
}))
outcomes$outcome <- factor(outcomes$outcome, levels = names(metrics))
outcomes$branch <- factor(ifelse(outcomes$onward_strength < .1, "Lower", "Higher"), levels = c("Higher", "Lower"))
pB <- ggplot(outcomes, aes(difference, branch)) +
  geom_vline(xintercept = 0, linewidth = .3, colour = "#90999F") +
  geom_linerange(data = subset(outcomes, outcome != "Infant cases"), aes(xmin = q025, xmax = q975),
                 linewidth = .6, colour = "#69757C") +
  geom_point(size = 2.3, colour = bmj_mechanism[["Onward"]]) +
  facet_wrap(~outcome, nrow = 1, scales = "free_x") +
  labs(title = "B: Infant outcomes at each matching solution", x = "Additional outcome averted (onward minus acquisition)", y = "Onward strength")
caption <- paste("**Figure S12. Alternative matches for adolescent cases in Japan at 30 years.**",
  "(A) The dashed line marks an alternative target of 289.58 reported adolescent cases averted. Two onward strengths and one acquisition strength meet this target. Lines connect the evaluated strengths and dots mark the solved roots. Full-strength onward protection increases adolescent cases.",
  "(B) Infant cases, deaths and years of life lost (YLL) under the lower and higher onward solutions relative to the common acquisition solution. Means and paired 95% conditional intervals use the adopted clinical-risk draws; infant reported cases are deterministic. Death counts are undiscounted and YLL are discounted. The two branches give opposite rankings on infant deaths.")
plot <- (pA / pB) + plot_layout(heights = c(1.6, 1))
panel_a_data <- rbind(data.frame(curve[, c("mechanism", "strength", "adolescent_cases_averted")], mark = "scan"),
                      data.frame(mechanism = roots$mechanism, strength = roots$strength,
                                 adolescent_cases_averted = roots$achieved_cases, mark = "root"))
panel_a_data$target_cases <- roots$target_cases[1]
save_figure("S12", plot, list(A = pA, B = pB), list(A = panel_a_data, B = outcomes), caption, height = 124)
