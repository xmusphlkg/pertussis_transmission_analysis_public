source("scripts/bmjgh_figures/theme.R")
theme_update(axis.text = element_text(size = 8), legend.text = element_text(size = 8),
             strip.text = element_text(size = 8))

base <- "outputs/core2026/02_sensitivity"
round2 <- file.path(base, "review_round2")
review <- file.path(base, "review_revision")
settings <- gsub(" ", "_", country_levels)
component_files <- file.path(review, paste0("hospital_components_", settings, ".csv"))
policy <- file.path(base, "mucosal_vaccines/bpze1_policy_20261107")
input_files <- c(file.path(round2, c("price_boundaries.csv", "health_components.csv")),
                 file.path(policy, "product_catalog.csv"), component_files,
                 file.path(review, "local_hospital_boundaries.csv"))

caps <- subset(read.csv(input_files[1]), burden == "reported_uniform")
catalog <- subset(read.csv(input_files[3]), profile_id == "b050_a050_d03_point")
key <- c("country", "strategy_id", "route")
stopifnot(!anyDuplicated(catalog[key]), all(caps$comparator == "current"))
caps <- merge(caps, catalog[c(key, "profile_id", "maternal_coverage", "nasal_coverage", "tdap_coverage")],
              by = key, sort = FALSE)
baseline <- unique(subset(catalog, route == "standalone", c(country, tdap_coverage)))
names(baseline)[2] <- "baseline_tdap_coverage"
caps <- merge(caps, baseline, by = "country", sort = FALSE)
caps$setting <- factor(gsub("_", " ", caps$country), levels = rev(country_levels))
caps$period <- as.character(caps$horizon)
caps$highest <- caps$threshold == ave(caps$threshold, caps$country, FUN = max)
stopifnot(nrow(caps) == 416L, nrow(baseline) == 8L,
          !anyDuplicated(caps[c("country", "horizon", "route", "mortality_scope", "threshold_id")]),
          all(caps$price_ceiling_q025 <= caps$price_ceiling_q975))
# Tied source endpoints have the same ceiling and selected strategy.
top <- subset(caps, highest)
id <- c("country", "horizon", "route", "mortality_scope")
selection <- c("strategy_id", "comparator", "threshold", "price_ceiling", "price_ceiling_q025", "price_ceiling_q975")
stopifnot(nrow(unique(top[c(id, selection)])) == 64L)
top <- top[order(top$threshold_id), ]
top <- top[!duplicated(top[id]), ]
harmonised <- subset(top, mortality_scope == "infant_only")
da <- subset(harmonised, route == "standalone")
db <- subset(harmonised, route == "coadmin")
added_tdap <- subset(db, tdap_coverage > baseline_tdap_coverage)
stopifnot(nrow(added_tdap) == 10L,
          setequal(added_tdap$country, c("England", "Japan", "China", "Brazil", "Thailand")))

money <- "Ceiling (2024 Int$/dose)"
signed_axis <- scales::pseudo_log_trans(sigma = .035, base = 10)
zero_x <- geom_vline(xintercept = 0, linetype = 2, colour = "#69757E", linewidth = .32)
zero_y <- geom_hline(yintercept = 0, linetype = 2, colour = "#69757E", linewidth = .32)
sites <- function(labels = rev(country_levels)) {
  scale_y_continuous(breaks = seq_along(labels), labels = labels,
                     limits = c(.4, length(labels) + .6), expand = c(0, 0))
}
year_labels <- c("20" = "20 years", "30" = "30 years")
forest <- function(d, title, breaks) {
  d$y <- as.integer(d$setting) + ifelse(d$horizon == 20, .19, -.19)
  ggplot(d, aes(price_ceiling, y, colour = period)) + zero_x +
    geom_segment(aes(x = price_ceiling_q025, xend = price_ceiling_q975, yend = y),
                 linewidth = .48, show.legend = FALSE) +
    geom_point(aes(shape = period), size = 1.8) +
    scale_colour_manual(NULL, values = horizon_colours, labels = year_labels) +
    scale_shape_manual(NULL, values = horizon_shapes, labels = year_labels) +
    scale_x_continuous(transform = signed_axis, breaks = breaks,
                       labels = scales::label_number(trim = TRUE, accuracy = .1, drop0trailing = TRUE)) +
    sites() + labs(x = "Ceiling (2024 Int$/dose)\nPseudo-log scale", y = NULL, title = title)
}
a <- forest(da, "A: Standalone delivery", c(0, .1, 1, 10))
brazil <- subset(da, country == "Brazil" & horizon == 30)
a <- a + annotate("text", x = .045, y = as.integer(brazil$setting) - .24, hjust = 0,
                  label = sprintf("%.5f", brazil$price_ceiling), size = 7.5, size.unit = "pt")
b <- forest(db, "B: Coadministration", c(-100, -10, -1, 0, 1, 10))

dc <- subset(caps, route == "standalone" & horizon == 20 & mortality_scope == "infant_only")
stopifnot(nrow(dc) == 52L, all(dc$price_ceiling > 0))
dc <- dc[order(dc$country, dc$price_ceiling, dc$threshold_id), ]
dc$n_records <- ave(dc$threshold, dc$country, FUN = length)
dc$record <- ifelse(dc$highest, "Highest valuation", "Other valuation")
# Pack neighbouring records vertically, keeping every monetary value at its observed x.
pack_records <- function(x) {
  x <- log10(x)
  lanes <- integer(length(x))
  for (i in seq_along(x)) {
    neighbours <- seq_len(i - 1)
    used <- lanes[neighbours[abs(x[neighbours] - x[i]) < .050]]
    lanes[i] <- setdiff(seq_along(x), used)[1]
  }
  (lanes - mean(range(lanes))) * .15
}
dc$y <- as.integer(dc$setting) + ave(dc$price_ceiling, dc$country, FUN = pack_records)
counts <- dc$n_records[match(rev(country_levels), dc$setting)]
stopifnot(all(abs(dc$y - as.integer(dc$setting)) <= .4))
cplot <- ggplot(dc, aes(price_ceiling, y)) +
  geom_point(aes(fill = record), shape = 21, colour = "#354B5B", stroke = .16, size = .8) +
  scale_fill_manual(NULL, values = c("Highest valuation" = horizon_colours[["20"]],
                                     "Other valuation" = "white")) +
  scale_x_log10(breaks = c(.005, .05, .5, 4), labels = c("0.005", "0.05", "0.5", "4")) +
  sites(paste0(rev(country_levels), " (", counts, ")")) +
  guides(fill = guide_legend(override.aes = list(size = 1.4))) +
  theme(legend.location = "plot") +
  labs(title = "C: Ceilings across valuations", x = "Standalone ceiling (2024 Int$/dose)\nLog scale", y = NULL)

health <- subset(read.csv(input_files[2]), horizon == 20 & burden == "reported_uniform" &
                   mortality_scope == "infant_only", c(country, horizon, strategy_id, infant_yll, yld, dalys))
cost <- do.call(rbind, lapply(component_files, read.csv))
cost_fields <- c("hospital_episode_savings", "outpatient_savings", "standard_programme_savings")
join <- c("country", "horizon", "strategy_id")
totals <- merge(subset(da, horizon == 20), health, by = join)
totals <- merge(totals, cost[c(join, "discounted_nasal_receipts", cost_fields)], by = join)
stopifnot(nrow(totals) == 8L, setequal(totals$country, settings),
          all(totals$discounted_nasal_receipts > 0),
          max(abs(totals$infant_yll + totals$yld - totals$dalys)) < 1e-8)
totals$health_value <- totals$threshold * (totals$infant_yll + totals$yld)
parts <- c("health_value", cost_fields)
totals$total <- rowSums(totals[parts]) / totals$discounted_nasal_receipts
stopifnot(max(abs(totals$total - totals$price_ceiling)) < 1e-9)
dd <- tidyr::pivot_longer(totals, all_of(parts), names_to = "component", values_to = "amount")
dd$value <- dd$amount / dd$discounted_nasal_receipts
dd$component <- factor(dd$component, levels = parts,
                       labels = c("Health value", "Hospital savings", "Outpatient savings", "Programme savings"))
dd$row <- as.integer(dd$setting) + (.5 - (as.integer(dd$component) - 1) / 3) * .55
d <- ggplot(dd, aes(value, row)) + zero_x +
  geom_point(aes(colour = component, shape = component), size = 1.6) +
  geom_point(data = totals, aes(total, as.integer(setting), colour = "Net ceiling", shape = "Net ceiling"), size = 2) +
  scale_colour_manual(NULL, values = c("Health value" = "#236876", "Hospital savings" = "#A85B20",
                      "Outpatient savings" = "#5C7237", "Programme savings" = "#886892", "Net ceiling" = "#222222")) +
  scale_shape_manual(NULL, values = c("Health value" = 16, "Hospital savings" = 17,
                     "Outpatient savings" = 15, "Programme savings" = 1, "Net ceiling" = 18)) +
  guides(colour = guide_legend(ncol = 2, byrow = TRUE), shape = guide_legend(ncol = 2, byrow = TRUE)) +
  sites() +
  theme(legend.box = "vertical", legend.location = "plot", legend.margin = margin(0, 0, 0, 0),
        legend.spacing.y = grid::unit(.2, "mm")) +
  labs(title = "D: Ceiling components", x = "Standalone contribution\n(2024 Int$/dose)", y = NULL)

de <- subset(top, route == "standalone" & ((country == "Thailand" & horizon == 20) |
                                           (country == "Brazil" & horizon == 30)))
pair_keys <- c("country", "horizon", "threshold_id", "threshold", "strategy_id", "comparator")
stopifnot(nrow(de) == 4L, nrow(unique(de[pair_keys])) == 2L)
de$scope <- factor(de$mortality_scope, levels = c("infant_only", "included"), labels = c("Infant", "All ages"))
de$condition <- factor(paste(de$country, de$horizon, "years"),
                       levels = c("Thailand 20 years", "Brazil 30 years"),
                       labels = c("Thailand, 20 years", "Brazil, 30 years"))
e <- ggplot(de, aes(scope, price_ceiling, group = country)) + zero_y +
  geom_line(linewidth = .4, colour = "#80909A") +
  geom_errorbar(aes(ymin = price_ceiling_q025, ymax = price_ceiling_q975, colour = period),
                width = .14, linewidth = .42) +
  geom_point(aes(colour = period, shape = period), size = 1.9) +
  scale_shape_manual(values = horizon_shapes, guide = "none") +
  scale_colour_manual(values = horizon_colours, guide = "none") +
  facet_wrap(~condition, scales = "free_y", nrow = 1, labeller = label_wrap_gen(width = 10)) +
  scale_y_continuous(n.breaks = 4, labels = scales::label_number()) +
  theme(panel.grid.major.x = element_blank(), panel.spacing.x = grid::unit(3, "mm")) +
  labs(title = "E: Mortality age scope", x = "Ages included in mortality", y = money)

primary <- subset(da, country == "Australia" & horizon == 20)
local <- subset(read.csv(file.path(review, "local_hospital_boundaries.csv")),
                country == "Australia" & horizon == 20 & burden == "reported_uniform" &
                route == "standalone" & unit_cost_multiplier == 1 & threshold_id == primary$threshold_id)
stopifnot(nrow(local) == 1L, local$threshold == primary$threshold,
          local$strategy_id == primary$strategy_id,
          local$same_coverage_standard_strategy == primary$comparator,
          local$best_standard_strategies == primary$comparator,
          abs(local$ceiling_vs_same_coverage_standard - local$ceiling_vs_best_standard) < 1e-9)
df <- primary[rep(1, 2), ]
df$assumption <- factor(c("Primary risks", "Local ratios"), levels = c("Primary risks", "Local ratios"))
df[2, c("price_ceiling", "price_ceiling_q025", "price_ceiling_q975")] <-
  local[c("ceiling_vs_best_standard", "same_coverage_ceiling_q025", "same_coverage_ceiling_q975")]
f <- ggplot(df, aes(assumption, price_ceiling, group = 1)) + zero_y +
  geom_line(colour = "#80909A", linewidth = .45) +
  geom_errorbar(aes(ymin = price_ceiling_q025, ymax = price_ceiling_q975),
                width = .12, linewidth = .43, colour = horizon_colours[["20"]]) +
  geom_point(size = 2.0, shape = horizon_shapes[["20"]], colour = horizon_colours[["20"]]) +
  geom_text(aes(label = sprintf("%.2f", price_ceiling)), nudge_x = .1, hjust = 0, size = 2.85) +
  scale_y_continuous(breaks = c(0, .5, 1, 1.5, 2, 2.5), expand = expansion(mult = c(.05, .07))) +
  theme(panel.grid.major.x = element_blank()) +
  labs(title = "F: Australian admission risks", x = "Admission assumptions", y = money)

panels <- list(A = a, B = b, C = cplot, D = d, E = e, F = f)
# A and B share setting labels and a horizon key. C retains its valuation counts.
routes <- wrap_plots(a, b, ncol = 2, axes = "collect_y", guides = "collect") &
  theme(legend.position = "bottom")
first_row <- wrap_plots(routes, cplot, ncol = 2, widths = c(1.8, 1))
second_row <- wrap_plots(d, patchwork::free(e, side = "lr"), f, ncol = 3,
                         widths = c(1.10, 1.86, .94))
figure <- wrap_plots(first_row, second_row, ncol = 1)
caption <- c("**Figure 4. Delivered-cost ceilings and their determinants.**", "",
  paste(
    "Ceilings are in 2024 international dollars per administered intranasal dose. A–D use the highest mean ceiling within each route, setting, horizon and valuation, relative to the standard programme with the highest expected net health benefit (table S28). A negative ceiling indicates lower expected net monetary benefit than the comparator at zero intranasal procurement and administration cost. All panels use the three-year reference profile and reported acute morbidity at all ages.",
    "**A, B** show standalone and coadministered ceilings at each setting’s highest sourced valuation with infant mortality included. Selected coadministration introduces adolescent Tdap in England, Japan, China, Brazil and Thailand and retains baseline Tdap coverage elsewhere. The horizontal pseudo-log axes have different ranges.",
    "**C** shows the 52 published valuation records for 20-year standalone delivery, with counts by setting. Filled points mark all records tied for the highest valuation. Records have equal visual weight.",
    "**D** breaks down the 20-year standalone ceilings in A into the health value of infant years of life lost and all-age acute disability averted, and savings on hospital, outpatient and other programme costs. Signed contributions are divided by discounted intranasal doses. Diamonds give their algebraic sum. Thailand’s outpatient contribution is −0.000041 Int$/dose.",
    "**E** compares standalone ceilings at the highest valuation under infant-only and all-age mortality. Selecting the strategy and standard comparator separately under each mortality scope yielded the same pair within each setting. Vertical scales differ by setting.",
    "**F** shows Australia’s 20-year standalone ceiling at its highest valuation, holding strategy and comparator fixed while replacing primary hospitalisation risks with local age-specific admission-to-notification ratios. Points in A, B, E and F show mean ceilings. Intervals span the central 95% of 1,000 paired clinical-risk and cost draws conditional on the fitted transmission parameters and reference vaccine profile."
  )
)
save_figure(4, figure, panels, list(A = da, B = db, C = dc, D = dd, E = de, F = df),
            caption, width = 190, height = 155)
mapping_fields <- c("country", "horizon", "route", "mortality_scope", "burden", "threshold_id",
                    "threshold", "profile_id", "strategy_id", "comparator", "nasal_coverage",
                    "maternal_coverage", "tdap_coverage", "price_ceiling",
                    "price_ceiling_q025", "price_ceiling_q975")
write.csv(caps[order(caps$country, caps$horizon, caps$route, caps$mortality_scope, caps$threshold_id),
               mapping_fields],
          "build/figures/source_data/economic_strategy_mapping.csv",
          row.names = FALSE)
