source("scripts/bmjgh_figures/theme.R")
input_files <- "outputs/core2026/02_sensitivity/review_round2/price_boundaries.csv"
z <- subset(read.csv(input_files),
            burden == "reported_uniform" & country %in% c("Brazil", "Thailand"))
z$scope <- factor(z$mortality_scope, levels = c("infant_only", "included"), labels = c("Infant deaths", "All-age deaths"))
z$condition <- factor(paste(z$country, z$horizon, "years"), levels = c("Brazil 20 years", "Brazil 30 years",
                                                                       "Thailand 20 years", "Thailand 30 years"))
stopifnot(nrow(z) == 128)
show_scope <- function(d, title) {
  ggplot(d, aes(threshold / 1000, price_ceiling, colour = scope, shape = scope)) +
    geom_point(size = 1.8, stroke = .5) +
    facet_wrap(~condition, ncol = 2, scales = "free") +
    scale_colour_manual(NULL, values = c("#466F85", "#B87934")) +
    scale_shape_manual(NULL, values = c(16, 2)) +
    labs(x = "Valuation per DALY averted (thousand 2024 Int$)", y = "Ceiling (2024 Int$/dose)",
         title = title)
}
data <- list(A = subset(z, route == "standalone"), B = subset(z, route == "coadmin"))
panels <- list(A = show_scope(data$A, "A: Standalone cost ceilings by mortality scope") + geom_hline(yintercept = 0, linewidth = .28, linetype = 2),
               B = show_scope(data$B, "B: Coadministration cost ceilings by mortality scope"))
plot <- wrap_plots(panels, ncol = 1, guides = "collect") &
  theme(legend.position = "bottom")
caption <- c("**Figure S17. Sensitivity of delivered-cost ceilings to the ages included in mortality estimates.**", "",
  paste("Points show the expected ceiling for standalone (A) and coadministered (B) delivery when infant deaths or all-age deaths contribute to valued health gains.",
        "Brazil and Thailand had mortality estimates for older ages. All comparisons include morbidity from reported cases at all ages.",
        "Each facet has its own scales. Records with identical numerical valuations overlap. The dashed line marks zero for standalone delivery. Coadministration ceilings were negative throughout."))
save_figure("S17", plot, panels, data, caption, height = 196)
