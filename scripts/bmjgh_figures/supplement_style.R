source("scripts/bmjgh_figures/theme.R")
theme_set(theme_classic(base_size = 8) +
  theme(legend.position = "bottom", strip.background = element_blank(),
        strip.text = element_text(size = 7), axis.text = element_text(size = 6),
        plot.tag = element_text(size = 9, face = "bold")))
figure_countries <- gsub(" ", "_", country_levels, fixed = TRUE)

export_supplement <- function(plot, number, width, height) {
  out <- file.path("build", "figures")
  dir.create(out, recursive = TRUE, showWarnings = FALSE)
  ggsave(file.path(out, paste0("figure", number, ".pdf")), plot,
         device = grDevices::cairo_pdf, units = "mm", width = width, height = height, bg = "white")
  ggsave(file.path(out, paste0("figure", number, ".png")), plot,
         device = ragg::agg_png, units = "mm", width = width, height = height, dpi = 300, bg = "white")
}
