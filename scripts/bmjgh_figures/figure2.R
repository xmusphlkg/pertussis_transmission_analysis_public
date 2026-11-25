source("scripts/bmjgh_figures/theme.R")
core <- "outputs/core2026"
clinical <- file.path(core, "07_manuscript/model_study_revision/results_revision")
audit <- file.path(core, "07_manuscript/model_study_revision/final_methodological_audit/matching")
mucosal <- file.path(core, "02_sensitivity/mucosal_vaccines")
input_files <- c(
  response = file.path(mucosal, "bpze1_policy_20261107/followup/mechanism_response.csv"),
  equal = file.path(clinical, "equal_strength_infant_outcomes.csv"),
  primary = file.path(clinical, "matched_infant_outcomes.csv"),
  sweden = file.path(audit, "Sweden/roots.csv"),
  japan_curve = file.path(audit, "Japan/scan.csv"),
  japan_roots = file.path(audit, "Japan/roots.csv"),
  refit = file.path(mucosal, "matched_refit_followup/matched_refit_summary.csv"),
  refit_yll = file.path(mucosal, "matching_geometry/matched_refit_clinical_magnitude.csv")
)
setting <- function(x) factor(gsub("_", " ", x), levels = rev(country_levels))
window <- function(x) factor(x, levels = c(20, 30), labels = c("20 years", "30 years"))
arm <- function(x) factor(x, levels = c("acquisition", "onward"),
                          labels = c("Acquisition", "Onward"))
mech_colours <- c(Acquisition = "#177563", Onward = "#8B4B82")
mech_shapes <- c(Acquisition = 18, Onward = 15)
horizon_colours <- c("20 years" = "#236B9B", "30 years" = "#C05B19")
horizon_shapes <- c("20 years" = 16, "30 years" = 17)
figure_theme <- theme(axis.text = element_text(size = 8), axis.title = element_text(size = 8.5),
                      strip.text = element_text(size = 8), legend.text = element_text(size = 8),
                      panel.grid.major.x = element_line(colour = "#E5EAED", linewidth = .2),
                      plot.title = element_text(size = 9, face = "bold", hjust = 0),
                      plot.title.position = "plot", plot.margin = margin(4, 6, 4, 4))
matrix_theme <- theme(panel.grid = element_blank(), axis.ticks = element_blank(),
                      axis.line = element_blank(), axis.title.x = element_blank())
signed_scale <- function(breaks) {
  labels <- scales::label_number(scale_cut = c("k" = 1000))(breaks)
  scale_x_continuous(transform = scales::transform_pseudo_log(base = 10, sigma = 1),
                     breaks = breaks, labels = labels,
                     expand = expansion(mult = .06))
}

# The full-strength and matched experiments use the same country-specific coverage.
response <- subset(read.csv(input_files["response"]), route == "standalone" & horizon == 30 &
                   mean_years == 3 & bridge_strength == .5 & clinical_anchor_label == "point" &
                   allocation %in% c(0, 1) & comparison == "versus_current")
a <- response[c("country", "horizon", "profile_id", "strategy_id", "matched_standard_strategy",
                "nasal_coverage", "maternal_coverage", "tdap_coverage",
                "mean_years", "bridge_strength", "allocation", "adolescent_10_19_reported_cases_averted")]
a$mechanism <- arm(ifelse(a$allocation == 1, "acquisition", "onward"))
a$setting <- setting(a$country)
primary <- read.csv(input_files["primary"])
equal <- subset(read.csv(input_files["equal"]), horizon == 30)
b <- do.call(rbind, lapply(c("acquisition", "onward"), function(mechanism) {
  key <- paste0(mechanism, "_infant_yll")
  data.frame(country = equal$country, horizon = equal$horizon, nasal_coverage = equal$nasal_coverage,
             mechanism = arm(mechanism), infant_yll = equal[[key]],
             q025 = equal[[paste0(key, "_q025")]], q975 = equal[[paste0(key, "_q975")]])
}))
coverage_keys <- c("country", "horizon", "nasal_coverage", "mechanism")
b <- merge(b, a[c(coverage_keys, "profile_id", "strategy_id", "matched_standard_strategy",
                  "maternal_coverage", "tdap_coverage")], by = coverage_keys)
b$setting <- setting(b$country)
stopifnot(nrow(a) == 16, all(table(a$setting, a$mechanism) == 1),
          nrow(b) == 16, all(table(b$setting, b$mechanism) == 1),
          all(a$nasal_coverage == pmax(.7, a$tdap_coverage)),
          all(b$q025 <= b$infant_yll & b$infant_yll <= b$q975))
equal_plot <- function(dat, metric, title, x_title, breaks, intervals = FALSE) {
  # Small vertical offsets preserve both intervals when the two arms nearly coincide.
  dat$row <- as.integer(dat$setting) + ifelse(dat$mechanism == "Acquisition", .12, -.12)
  p <- ggplot(dat, aes(.data[[metric]], row)) +
    geom_vline(xintercept = 0, colour = "#71808A", linewidth = .3) +
    geom_path(aes(group = setting), colour = "#8D989D", linewidth = .35)
  if (intervals) {
    p <- p + geom_linerange(aes(xmin = q025, xmax = q975, colour = mechanism),
                            linewidth = .46, show.legend = FALSE)
  }
  p + geom_point(aes(colour = mechanism, shape = mechanism), size = 1.9) +
    scale_colour_manual(NULL, values = mech_colours) + scale_shape_manual(NULL, values = mech_shapes) +
    scale_y_continuous(breaks = 1:8, labels = rev(country_levels), limits = c(.5, 8.5),
                       expand = expansion(0)) +
    signed_scale(breaks) + labs(title = title, x = x_title, y = NULL) + figure_theme
}
pA <- equal_plot(a, "adolescent_10_19_reported_cases_averted", "A: Adolescent cases at equal strength",
                 "Cases averted at 30 years\n(pseudo-log scale)", c(-100, 0, 100, 10000))
pB <- equal_plot(b, "infant_yll", "B: Infant YLL at equal strength",
                 "YLL averted at 30 years\n(pseudo-log scale)",
                 c(-100, 0, 10, 100, 1000), TRUE)

matched <- primary$status == "matched"
stopifnot(nrow(primary) == 16, !anyDuplicated(primary[c("country", "horizon")]), sum(matched) == 12,
          all(matched == (pmin(primary$acquisition_endpoint, primary$onward_endpoint) > 0)),
          max(abs(primary$target_adolescent_cases[matched] -
                  .5 * pmin(primary$acquisition_endpoint, primary$onward_endpoint)[matched])) < 1e-7)

# Sweden's independently refined roots retain the achieved adolescent count at solver precision.
swe <- subset(read.csv(input_files["sweden"]), country == "Sweden" & horizon == 30 &
              target_label == "primary_half_endpoint")
sweden_primary <- subset(primary, country == "Sweden" & horizon == 30)
stopifnot(nrow(swe) == 2, all(table(swe$mechanism) == 1), max(abs(swe$matching_residual)) < .001,
          all(abs(swe$target_cases - sweden_primary$target_adolescent_cases) < 1e-7),
          max(abs(swe$infant_cases_averted - ifelse(swe$mechanism == "acquisition",
              sweden_primary$acquisition_infant_cases, sweden_primary$onward_infant_cases))) < .001)
swe$achieved_cases <- swe$target_cases + swe$matching_residual
sweden <- do.call(rbind, lapply(c("Adolescent", "Infant"), function(group) {
  data.frame(country = swe$country, horizon = swe$horizon, target = swe$target_cases,
             mechanism = arm(swe$mechanism), strength = swe$root_strength,
             age = group, cases = if (group == "Adolescent") swe$achieved_cases else swe$infant_cases_averted)
}))
sweden$age <- factor(sweden$age, levels = c("Adolescent", "Infant"))
sweden$label <- ifelse(sweden$age == "Adolescent", sprintf("%.0f", sweden$cases), sprintf("%.2f", sweden$cases))
pC <- ggplot(sweden, aes(cases, mechanism)) +
  geom_vline(xintercept = 0, colour = "#71808A", linewidth = .3) +
  geom_segment(aes(x = 0, xend = cases, yend = mechanism), colour = "#B8C2C7", linewidth = .48) +
  geom_point(aes(colour = mechanism, shape = mechanism), size = 2.1) +
  geom_text(aes(label = label), hjust = -.28, size = 8, size.unit = "pt") +
  scale_colour_manual(values = mech_colours, guide = "none") +
  scale_shape_manual(values = mech_shapes, guide = "none") +
  scale_y_discrete(limits = rev(levels(sweden$mechanism))) +
  scale_x_continuous(breaks = scales::breaks_pretty(3), expand = expansion(mult = c(.08, .43))) +
  facet_wrap(~age, ncol = 1, scales = "free_x") +
  labs(title = "C: Matched outcomes in Sweden", x = "Reported cases averted", y = NULL) + figure_theme +
  theme(strip.text = element_text(hjust = 0))

primary_yll <- primary[c("country", "horizon", "nasal_coverage", "status", "target_adolescent_cases",
               "acquisition_endpoint", "onward_endpoint", "onward_strength", "acquisition_strength",
               "difference_infant_yll",
               "difference_infant_yll_q025", "difference_infant_yll_q975")]
primary_yll$window <- window(primary_yll$horizon)
primary_yll$setting <- setting(primary_yll$country)
ok <- subset(primary_yll, status == "matched")
nm <- subset(primary_yll, status != "matched")
stopifnot(all(ok$difference_infant_yll_q025 <= ok$difference_infant_yll &
              ok$difference_infant_yll <= ok$difference_infant_yll_q975),
          all(is.na(nm$difference_infant_yll)))
pD <- ggplot(ok, aes(difference_infant_yll, setting)) +
  geom_vline(xintercept = 0, linewidth = .32, linetype = 2, colour = "#71808A") +
  geom_linerange(aes(xmin = difference_infant_yll_q025, xmax = difference_infant_yll_q975,
                     colour = window), linewidth = .55) +
  geom_point(aes(colour = window, shape = window), size = 2.15) +
  geom_text(data = nm, aes(x = 2, label = "NM"), colour = "#626C72", size = 8, size.unit = "pt") +
  scale_colour_manual(values = horizon_colours, guide = "none") +
  scale_shape_manual(values = horizon_shapes, guide = "none") +
  scale_y_discrete(drop = FALSE) +
  signed_scale(c(0, 10, 1000)) +
  facet_wrap(~window, nrow = 1) +
  labs(title = "D: Matched infant YLL differences",
       x = "YLL averted (pseudo-log)\nOnward minus acquisition", y = NULL) + figure_theme

# All evaluated strengths and all detected roots at the independently specified Japanese target.
scan <- subset(read.csv(input_files["japan_curve"]), country == "Japan" & horizon == 30)
roots <- subset(read.csv(input_files["japan_roots"]), country == "Japan" & horizon == 30 &
                target_label == "supplementary_half_sampled_peak")
stopifnot(all(table(scan$mechanism) == 81), !anyDuplicated(scan[c("mechanism", "strength")]),
          all(range(scan$strength) == c(0, 1)), all(scan$endpoint_rule_status == "no_positive_endpoint_overlap"),
          nrow(roots) == 3, sum(roots$mechanism == "onward") == 2, length(unique(roots$target_cases)) == 1,
          all(abs(roots$matching_residual) < .001))
japan <- rbind(
  data.frame(arm = arm(scan$mechanism), strength = scan$strength, record = "scan",
             adolescent_cases_averted = scan$adolescent_cases_averted, infant_deaths_averted = NA_real_),
  data.frame(arm = arm(roots$mechanism), strength = roots$root_strength, record = "root",
             adolescent_cases_averted = roots$target_cases + roots$matching_residual,
             infant_deaths_averted = roots$infant_deaths_averted))
japan$country <- "Japan"
japan$horizon <- 30
japan$alternative_target <- unique(roots$target_cases)
japan$target_rule <- unique(roots$target_label)
root_marks <- subset(japan, record == "root")
root_marks$solution <- rank(root_marks$strength)
japan$solution <- NA_integer_
japan$solution[japan$record == "root"] <- root_marks$solution
strength_scale <- scale_x_continuous(transform = scales::transform_pseudo_log(sigma = .01, base = 10),
                                     breaks = c(0, .03, .1, .3, 1), labels = c("0", ".03", ".1", ".3", "1"),
                                     limits = c(0, 1), expand = expansion(mult = .035))
pE_curve <- ggplot(subset(japan, record == "scan"), aes(strength, adolescent_cases_averted, colour = arm)) +
  geom_hline(yintercept = 0, colour = "#A8B1B6", linewidth = .25) +
  geom_hline(yintercept = unique(japan$alternative_target), colour = "#626C72", linetype = 2, linewidth = .35) +
  geom_line(linewidth = .52) +
  geom_point(data = root_marks, aes(shape = arm), size = 1.9) +
  geom_text(data = root_marks, aes(label = solution), nudge_y = ifelse(root_marks$solution == 2, -300, 280),
            size = 8, size.unit = "pt", show.legend = FALSE) +
  scale_colour_manual(values = mech_colours, guide = "none") +
  scale_shape_manual(values = mech_shapes, guide = "none") + strength_scale +
  scale_y_continuous(breaks = c(-300, 290, 1000, 2000), labels = c("−300", "290", "1k", "2k")) +
  labs(title = "E: Alternative target in Japan", x = NULL, y = "Adolescent cases\naverted") +
  figure_theme + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(),
                       plot.margin = margin(4, 6, 0, 4))
pE_deaths <- ggplot(root_marks, aes(strength, infant_deaths_averted, colour = arm, shape = arm)) +
  geom_point(size = 2.2) +
  geom_text(aes(label = solution), nudge_y = -.15, size = 8, size.unit = "pt", show.legend = FALSE) +
  scale_colour_manual(values = mech_colours, guide = "none") +
  scale_shape_manual(values = mech_shapes, guide = "none") + strength_scale +
  scale_y_continuous(breaks = c(2.4, 3.1, 3.8), limits = c(2.15, 3.95)) +
  labs(x = "Mechanism strength\n(pseudo-log scale)", y = "Infant deaths\naverted") +
  figure_theme + theme(plot.margin = margin(1, 6, 4, 4))
pE <- wrap_plots(pE_curve, pE_deaths, ncol = 1, heights = c(1.4, 1))

# Retain the failures as explicit cells. The 25 available refit magnitudes use infant-only YLL.
refit <- read.csv(input_files["refit"])
mag <- read.csv(input_files["refit_yll"])
hist_keys <- c("country", "horizon", "history")
stopifnot(nrow(refit) == 32, !anyDuplicated(refit[hist_keys]), nrow(mag) == 25,
          !anyDuplicated(mag[hist_keys]))
refit <- merge(refit, mag[c(hist_keys, "target_cases", "difference_infant_yll",
                            "difference_infant_yll_q025", "difference_infant_yll_q975")],
               by = hist_keys, all.x = TRUE)
stopifnot(all((refit$status == "matched") == is.finite(refit$difference_infant_yll)),
          all(abs(refit$target_adolescent_cases_averted - refit$target_cases) < 1e-7, na.rm = TRUE))
refit$target_adolescent_cases <- refit$target_adolescent_cases_averted
adopted <- primary_yll
adopted$history <- "adopted"
history_fields <- c(hist_keys, "status", "nasal_coverage", "target_adolescent_cases", "onward_strength",
              "acquisition_strength", "difference_infant_yll", "difference_infant_yll_q025",
              "difference_infant_yll_q975")
histories <- rbind(adopted[history_fields], refit[history_fields])
histories$setting <- setting(histories$country)
histories$window <- window(histories$horizon)
histories$fit <- factor(histories$history, levels = c("adopted", "baseline", "recovery_365"),
                labels = c("Adopted", "Base", "365 d"))
stopifnot(nrow(histories) == 48, all(table(histories$setting, histories$window, histories$fit) == 1),
          sum(histories$status == "matched") == 37)
histories$label <- sprintf("%.0f", histories$difference_infant_yll)
for (digits in 1:2) {
  more_precision <- is.finite(histories$difference_infant_yll) & abs(histories$difference_infant_yll) < 10^(3 - digits)
  histories$label[more_precision] <- formatC(histories$difference_infant_yll[more_precision], digits = digits, format = "f")
}
histories$label[is.na(histories$difference_infant_yll)] <- "NM"
histories$displayed <- histories$history != "baseline"
comparison <- merge(subset(histories, history == "adopted"), subset(histories, history == "baseline"),
                    by = c("country", "horizon"))
stopifnot(all(comparison$label.x == comparison$label.y))
pF <- ggplot(subset(histories, displayed), aes(fit, setting)) +
  geom_tile(fill = "#F0F2F2", colour = "white", linewidth = .3) +
  geom_text(aes(label = label), size = 7.7, size.unit = "pt") +
  scale_x_discrete(expand = expansion(add = .5), guide = guide_axis(angle = 40)) +
  scale_y_discrete(expand = expansion(add = .5)) +
  facet_wrap(~window, nrow = 1) +
  labs(title = "F: Infant YLL differences by fit", x = NULL, y = NULL) +
  figure_theme + matrix_theme

panels <- list(A = pA, B = pB, C = pC, D = pD, E = pE, F = pF)
hide_settings <- theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())
row1 <- wrap_plots(pA, pB + hide_settings, pC, nrow = 1,
                   widths = c(1.1, 1.05, .95), guides = "collect") &
  theme(legend.position = "bottom")
row2 <- wrap_plots(patchwork::free(pD, type = "label", side = "b"), patchwork::free(pE), pF,
                   nrow = 1, widths = c(1.20, 1.40, 1.00))
plot <- wrap_plots(row1, row2, ncol = 1, heights = c(1, 1))

caption <- paste(
  "**Figure 2. Transmission mechanisms and infant protection.**",
  "All comparisons use standalone intranasal vaccination, the adult bacterial-load area-under-the-curve (AUC) point estimate, translation exponent 0.5, three-year mean protection and baseline maternal and Tdap programmes. Uptake among eligible adolescents is 70% or baseline adolescent Tdap coverage, whichever is higher.",
  "**A, B** show reported cases averted in adolescents aged 10–19 years and discounted infant years of life lost (YLL) averted over 30 years at full strength. Lines pair acquisition-only and onward-contribution-only protection. Axes are approximately linear near zero and logarithmic at larger magnitudes.",
  "**C** shows a matched reduction of 164 adolescent cases over 30 years in Sweden. Acquisition protection adds 0.32 infant cases and onward protection averts 15.29.",
  "**D** shows the difference in infant YLL averted (onward minus acquisition) for the 12 primary matches on a signed pseudo-log scale. Targets equal half the smaller full-strength adolescent case reduction when both mechanisms avert cases. NM indicates no positive target. Intervals in B and D span the central 95% of 1,000 clinical-risk draws conditional on fitted transmission history. D uses paired differences.",
  "**E** shows an alternative 30-year target in Japan of 289.58 adolescent cases averted (dashed line). Lines connect 81 strengths per mechanism, scaling the protected-state log effect. The same three solutions, numbered by increasing strength, appear in both plots. The lower plot gives infant deaths averted relative to current provision.",
  "**F** shows mean onward-minus-acquisition differences in infant YLL averted at targets specific to each setting, horizon and fit. Adopted and 365 d denote the adopted fit and the full refit with 365-day contact recovery. The baseline-recovery refit gave the same displayed values and unmatched comparisons as the adopted fit. Panel data include targets and unrounded results for all three fits."
)
save_figure(2, plot, panels, list(A = a, B = b, C = sweden, D = primary_yll, E = japan, F = histories),
            caption, width = 190, height = 152)
