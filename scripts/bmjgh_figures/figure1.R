# Infant outcomes under the standalone adolescent programme.
source("scripts/bmjgh_figures/theme.R")
theme_set(bmj_theme() + theme(axis.text = element_text(size = 8),
                              strip.text = element_text(size = 8),
                              legend.text = element_text(size = 8),
                              legend.title = element_text(size = 8)))

result_dir <- "outputs/core2026/07_manuscript/model_study_revision/results_revision"
long_dir <- "outputs/core2026/02_sensitivity/review_revision"
reference_file <- file.path(result_dir, "infant_outcomes_reference.csv")
profiles_file <- file.path(result_dir, "infant_outcomes_by_profile.csv")
baseline_file <- file.path(result_dir, "baseline_infant_deaths_by_horizon.csv")
check_file <- file.path(result_dir, "baseline_infant_deaths.csv")
programmes_file <- file.path(result_dir, "baseline_programmes.csv")
design_file <- "source_data/figures/profile_order.json"
countries_long <- c("Brazil", "Japan")
annual_files <- file.path(long_dir, paste0("annual_", countries_long, "_adopted_unchanged.csv"))
endpoint_files <- file.path(long_dir, paste0("long_totals_", countries_long, "_adopted_unchanged.csv"))
input_files <- c(reference_file, profiles_file, baseline_file, check_file, programmes_file,
                 design_file, annual_files, endpoint_files)

design <- jsonlite::fromJSON(design_file)
reference_id <- "b050_a050_d03_point"
ref <- subset(read.csv(reference_file), route == "standalone")
profiles <- subset(read.csv(profiles_file), route == "standalone")
programmes <- read.csv(programmes_file)
setting <- function(x) factor(gsub("_", " ", x, fixed = TRUE), levels = rev(country_levels))
for (rows in list(ref, profiles)) {
  baseline <- programmes[match(rows$country, programmes$country), ]
  stopifnot(all(rows$nasal_coverage == .9), all(rows$comparator == "current"),
            all(rows$maternal_coverage == baseline$maternal_coverage),
            all(rows$tdap_coverage == baseline$adolescent_coverage),
            all(is.finite(rows$deaths_averted)), all(is.finite(rows$yll_averted)))
}
stopifnot(all(ref$profile_id == reference_id), design$profile_ids[5] == reference_id,
          nrow(ref) == 16, all(table(ref$country, ref$horizon) == 1),
          setequal(ref$horizon, c(20, 30)), setequal(ref$country, design$countries),
          setequal(profiles$country, design$countries), setequal(profiles$horizon, c(20, 30)),
          setequal(profiles$profile_id, design$profile_ids), length(design$profile_ids) == 17,
          all(table(profiles$country, profiles$horizon, profiles$profile_id) == 1))
baseline <- read.csv(baseline_file)
key <- c("country", "horizon")
stopifnot(nrow(baseline) == 16, !anyDuplicated(baseline[key]), all(baseline$baseline_infant_deaths > 0))
ref <- merge(ref, baseline, by = key)
ref$percent_baseline_deaths <- 100 * ref$deaths_averted / ref$baseline_infant_deaths
published <- merge(ref, read.csv(check_file), by = key)
stopifnot(nrow(ref) == 16, nrow(published) == 8, all(published$horizon == 20),
          max(abs(published$baseline_infant_deaths - published$infant_deaths)) < 1e-7,
          max(abs(published$percent_baseline_deaths - published$standalone_percent_baseline_deaths)) < 1e-7)
ref$position <- as.integer(setting(ref$country)) + ifelse(ref$horizon == 20, .16, -.16)
ref$window <- factor(ref$horizon, levels = c(20, 30))
profiles$position <- as.integer(setting(profiles$country))
profiles$profile_number <- match(profiles$profile_id, design$profile_ids)

window_scales <- list(
  scale_colour_manual("Evaluation horizon", values = horizon_colours,
                      labels = function(x) paste(x, "years")),
  scale_shape_manual("Evaluation horizon", values = horizon_shapes,
                     labels = function(x) paste(x, "years"))
)
zero_vertical <- geom_vline(xintercept = 0, linewidth = .32, linetype = 2, colour = "#707A80")
site_axis <- scale_y_continuous(breaks = 1:8, labels = rev(country_levels),
                                limits = c(.5, 8.5), expand = c(0, 0))
signed_axis <- function(breaks, sigma) {
  scale_x_continuous(transform = scales::transform_pseudo_log(sigma = sigma),
                     breaks = breaks, labels = ifelse(abs(breaks) >= 1000, paste0(breaks / 1000, "k"), breaks),
                     expand = expansion(mult = .07))
}
forest <- function(field, title, xlabel, axis, intervals = TRUE) {
  p <- ggplot(ref, aes(.data[[field]], position, colour = window, shape = window)) +
    zero_vertical
  if (intervals) {
    p <- p + geom_segment(aes(x = .data[[paste0(field, "_q025")]],
                              xend = .data[[paste0(field, "_q975")]], yend = position),
                          linewidth = .55, show.legend = FALSE)
  }
  p + geom_point(size = 1.9) + window_scales + site_axis + axis +
    labs(title = title, x = xlabel, y = NULL) + theme(axis.ticks.y = element_blank())
}
a <- forest("yll_averted", "A: Discounted infant YLL", "YLL averted (pseudo-log)",
            signed_axis(c(-10, 0, 10, 100, 1000), 1))
b <- forest("deaths_averted", "B: Infant deaths averted", "Deaths averted (pseudo-log)",
            signed_axis(c(-10, -1, 0, 1, 10, 100), .1))
cplot <- forest("percent_baseline_deaths", "C: Relative mortality reduction", "Baseline infant deaths averted (%)",
                scale_x_continuous(breaks = c(-5, 0, 5, 10, 15, 20)), intervals = FALSE)

attributes <- unique(profiles[c("profile_number", "duration_years", "translation_exponent",
                                "acquisition_allocation", "clinical_anchor")])
stopifnot(nrow(attributes) == 17)
fraction_labels <- c("0" = "0", "0.25" = "¼", "0.5" = "½", "1" = "1")
profile_strips <- rbind(
  data.frame(x = attributes$profile_number, y = 11.15, label = as.character(attributes$duration_years)),
  data.frame(x = attributes$profile_number, y = 10.25,
             label = unname(fraction_labels[as.character(attributes$translation_exponent)])),
  data.frame(x = attributes$profile_number, y = 9.35,
             label = unname(fraction_labels[as.character(attributes$acquisition_allocation)]))
)
stopifnot(!anyNA(profile_strips))
extent <- max(abs(profiles$yll_averted))
heatmap <- function(rows, title) {
  ggplot(rows, aes(profile_number, position)) +
    geom_tile(aes(fill = yll_averted), width = .97, height = .96) +
    geom_vline(xintercept = c(9.5, 15.5), colour = "white", linewidth = .8) +
    annotate("rect", xmin = 4.51, xmax = 5.49, ymin = .52, ymax = 8.48,
             fill = NA, colour = "#242E33", linewidth = .45) +
    geom_text(data = profile_strips, aes(x, y, label = label), inherit.aes = FALSE,
              size = 7.6, size.unit = "pt", family = "Arial", colour = "#334149") +
    scale_x_continuous(breaks = 1:17, labels = sprintf("%02d", 1:17),
                       expand = c(0, 0), limits = c(.49, 17.51)) +
    scale_y_continuous(breaks = c(1:8, 9.35, 10.25, 11.15),
                       labels = c(rev(country_levels), "Acq. share", "Exponent", "Years"),
                       limits = c(.48, 11.73), expand = c(0, 0)) +
    scale_fill_gradient2("Discounted infant YLL averted", low = "#953E71", mid = "#FBFAF7",
                         high = "#00736C", midpoint = 0, limits = c(-extent, extent),
                         transform = scales::transform_pseudo_log(sigma = 2),
                         breaks = c(-1000, -100, -10, 0, 10, 100, 1000),
                         labels = c("−1k", "−100", "−10", "0", "10", "100", "1k")) +
    guides(fill = guide_colourbar(direction = "horizontal",
                                  barwidth = grid::unit(50, "mm"), barheight = grid::unit(2.8, "mm"))) +
    labs(title = title, x = NULL, y = NULL) +
    theme(panel.grid.major.x = element_blank(), axis.ticks = element_blank(), axis.line = element_blank(),
          legend.position = "bottom")
}
profiles_20 <- subset(profiles, horizon == 20)
profiles_30 <- subset(profiles, horizon == 30)
d <- heatmap(profiles_20, "D: Vaccine profiles, 20 years")
e <- heatmap(profiles_30, "E: Vaccine profiles, 30 years")

# The only ten-year profile has a matched three-year counterpart in all eight settings.
duration <- subset(profiles, clinical_anchor == "point" & translation_exponent == .5 &
                   acquisition_allocation == .5 & duration_years %in% c(3, 10))
pair_key <- c("country", "horizon", "clinical_anchor", "translation_exponent",
              "acquisition_allocation", "comparator", "maternal_coverage", "tdap_coverage",
              "nasal_coverage", "route")
pair <- merge(subset(duration, duration_years == 3), subset(duration, duration_years == 10),
              by = pair_key, suffixes = c("_3", "_10"))
stopifnot(nrow(pair) == 16, !anyDuplicated(pair[key]), all(pair$profile_id_3 == reference_id),
          all(table(duration$country, duration$horizon, duration$duration_years) == 1))
duration$window <- factor(duration$horizon, levels = c(20, 30), labels = c("20 years", "30 years"))
pair$window <- factor(pair$horizon, levels = c(20, 30), labels = levels(duration$window))
f <- ggplot(pair, aes(y = position_3)) + zero_vertical +
  geom_segment(aes(x = deaths_averted_3, xend = deaths_averted_10, yend = position_10),
               colour = "#989FA1", linewidth = .55) +
  geom_point(data = duration, aes(deaths_averted, position, fill = factor(duration_years),
                                  shape = factor(horizon), colour = factor(horizon)),
             size = 1.95, stroke = .6) +
  scale_colour_manual(values = horizon_colours, guide = "none") +
  scale_shape_manual(values = c("20" = 21, "30" = 24), guide = "none") +
  scale_fill_manual("Mean protection", values = c("3" = "white", "10" = "#323D43"),
                    labels = c("3 years", "10 years"),
                    guide = guide_legend(override.aes = list(shape = 21, colour = "#323D43"))) +
  facet_wrap(~window, nrow = 1) + site_axis + signed_axis(c(-10, 0, 10, 100), .3) +
  labs(title = "F: Protection duration and infant deaths", x = "Infant deaths averted (pseudo-log)", y = NULL) +
  theme(axis.ticks.y = element_blank())

annual <- subset(do.call(rbind, lapply(annual_files, read.csv)),
                 scenario == "adopted_unchanged" & age_group == "infant_0" & metric == "deaths")
annual <- annual[order(annual$country, annual$year), ]
annual$horizon <- annual$year - 2026
stopifnot(all(table(annual$country, annual$horizon) == 1), setequal(annual$horizon, 1:50))
annual$cumulative_deaths <- ave(annual$averted, annual$country, FUN = cumsum)
totals <- subset(do.call(rbind, lapply(endpoint_files, read.csv)),
                 scenario == "adopted_unchanged" & age_group == "infant_0" & horizon %in% c(20, 30, 50))
points <- merge(annual, totals[c(key, "deaths_averted", "deaths_averted_q025", "deaths_averted_q975")],
                by = key)
stopifnot(nrow(points) == 6, max(abs(points$cumulative_deaths - points$deaths_averted)) < 1e-7)
checks <- merge(points, ref, by = key)
stopifnot(nrow(checks) == 4, max(abs(checks$deaths_averted.x - checks$deaths_averted.y)) < .001)
y_breaks <- pretty(c(0, annual$cumulative_deaths, points$deaths_averted_q025,
                     points$deaths_averted_q975), n = 4)
g <- ggplot(annual, aes(horizon, cumulative_deaths)) +
  geom_hline(yintercept = 0, colour = "#707A80", linetype = 2, linewidth = .33) +
  geom_step(colour = "#54696C", linewidth = .44) +
  geom_linerange(data = points, aes(ymin = deaths_averted_q025, ymax = deaths_averted_q975,
                                    colour = factor(horizon)), linewidth = .54, show.legend = FALSE) +
  geom_point(data = points, aes(colour = factor(horizon), shape = factor(horizon)), size = 2) +
  window_scales + facet_wrap(~country, nrow = 1) +
  scale_x_continuous(breaks = c(1, 20, 30, 50), expand = expansion(mult = .05)) +
  scale_y_continuous(breaks = y_breaks, limits = range(y_breaks), expand = c(0, 0)) +
  labs(title = "G: Infant deaths over 50 years", x = "Year of programme", y = "Cumulative infant deaths averted")

without_sites <- theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())
top <- wrap_plots(a, b + without_sites,
                  cplot + without_sites, nrow = 1, guides = "collect")
middle <- wrap_plots(d, e + without_sites,
                     nrow = 1, guides = "collect")
bottom <- wrap_plots(f, g, nrow = 1, widths = c(1.02, 1))
fig <- wrap_plots(top, middle, bottom, ncol = 1, heights = c(.94, 1.13, 1))

common <- c("country", "horizon", "route", "nasal_coverage", "maternal_coverage", "tdap_coverage",
            "comparator", "strategy_id", "profile_id")
metric_columns <- function(metric) c(metric, paste0(metric, c("_q025", "_q975")))
g_source <- merge(annual[c(key, "scenario", "year", "age_group", "metric", "averted", "cumulative_deaths")],
                  points[c(key, metric_columns("deaths_averted"))], by = key, all.x = TRUE)
g_source <- merge(g_source, unique(subset(ref, country %in% countries_long)[setdiff(common, "horizon")]),
                  by = "country")
source <- list(A = ref[c(common, metric_columns("yll_averted"))],
               B = ref[c(common, metric_columns("deaths_averted"))],
               C = ref[c(common, "deaths_averted", "baseline_infant_deaths", "percent_baseline_deaths")],
               D = profiles_20, E = profiles_30, F = duration, G = g_source)
panels <- list(A = a, B = b, C = cplot, D = d, E = e, F = f, G = g)
caption <- paste(
  "**Figure 1. Projected infant outcomes across settings, vaccine profiles and evaluation horizons.**",
  "Standalone intranasal vaccination at 90% uptake among eligible adolescents is compared with the same baseline maternal and adolescent Tdap programme without intranasal vaccination. Positive values indicate outcomes averted.",
  "**A–C** show infant outcomes averted by reference profile 05 over 20 and 30 years from 2027. A gives discounted infant years of life lost (YLL), B cumulative infant deaths and C deaths averted as a percentage of mean baseline infant deaths. Death counts are undiscounted. YLL are discounted at 3% annually for time to death and remaining life-years.",
  "**D, E** show mean discounted infant YLL averted over 20 and 30 years on a shared pseudo-log colour scale (sigma 2, natural logarithm). Profiles 01–09 vary translation and mechanism allocation at three-year mean protection, 10–15 vary duration at equal allocation, and 16–17 use the lower and upper adult bacterial-load area-under-the-curve (AUC) ratio anchors. Rows above the matrices give duration, translation exponent and acquisition share. The outline marks profile 05. Profile definitions and counts with fewer deaths or lower YLL appear in tables S19 and S46.",
  "**F** compares cumulative infant deaths averted with three and ten years of protection at the point AUC estimate, translation exponent 0.5 and equal allocation between acquisition and onward transmission. Lines join durations within each setting and horizon. A, B and F use signed pseudo-log horizontal axes.",
  "**G** shows cumulative infant deaths averted in Brazil and Japan under profile 05. Circles, triangles and squares mark 20, 30 and 50 years. Intervals in A, B and G span the central 95% of 1,000 paired clinical-risk draws conditional on fitted transmission parameters. Intervals in G use cumulative draws."
)
save_figure(1, fig, panels, source, caption, width = 190, height = 225)
