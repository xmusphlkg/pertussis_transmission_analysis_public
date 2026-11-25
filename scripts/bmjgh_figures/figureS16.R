source("scripts/bmjgh_figures/theme.R")
input_files <- "outputs/core2026/02_sensitivity/review_round2/price_boundaries.csv"
z <- subset(read.csv(input_files),
            burden == "reported_uniform" & mortality_scope == "infant_only")
z$setting <- factor(gsub("_", " ", z$country), levels = country_levels)
z$period <- factor(z$horizon, levels = c(20, 30), labels = c("20 years", "30 years"))
stopifnot(nrow(z) == 208)
show_route <- function(d, title) {
  ggplot(d, aes(threshold / 1000, price_ceiling, colour = period, shape = period)) +
    geom_hline(yintercept = 0, linewidth = .28, linetype = 2) +
    geom_linerange(aes(ymin = price_ceiling_q025, ymax = price_ceiling_q975), linewidth = .38) +
    geom_point(size = 1.25) +
    facet_wrap(~setting, ncol = 4, scales = "free") +
    scale_colour_manual(NULL, values = c("#466F85", "#B87934")) +
    scale_shape_manual(NULL, values = c(16, 17)) +
    labs(x = "Valuation per DALY averted (thousand 2024 Int$)", y = "Ceiling (2024 Int$/dose)",
         title = title)
}
data <- list(A = subset(z, route == "standalone"), B = subset(z, route == "coadmin"))
panels <- list(A = show_route(data$A, "A: Standalone cost ceilings"), B = show_route(data$B, "B: Coadministration cost ceilings"))
plot <- wrap_plots(panels, ncol = 1, guides = "collect") &
  theme(legend.position = "bottom")
caption <- c("**Figure S16. Delivered-cost ceilings and conditional intervals at every sourced valuation.**", "",
  paste("Points show means for standalone (A) or coadministered (B) intranasal vaccination. Lines give 2.5th–97.5th percentiles from 1,000 paired clinical and cost draws conditional on fixed transmission parameters and the programmes defining the mean ceiling.",
        "The 52 valuation records are setting-specific. Records with identical numerical valuations overlap. Each setting has its own scales. All comparisons include infant deaths and all-age reported-case morbidity. The dashed line marks zero delivered cost."))
save_figure("S16", plot, panels, data, caption, height = 210)
