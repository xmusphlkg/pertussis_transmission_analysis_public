# Run figure scripts from the project root.
suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
})

country_levels <- c("Australia", "England", "Sweden", "Japan", "New Zealand",
                    "China", "Brazil", "Thailand")
bmj_country <- setNames(c("#466F85", "#77568B", "#437569", "#9D6842",
                           "#686781", "#835347", "#326C78", "#69734E"), country_levels)
bmj_mechanism <- c(Acquisition = "#2878A5", Onward = "#C47827", Combined = "#27886F")
bmj_outcome <- c(Cases = "#2878A5", Deaths = "#27886F", YLL = "#8061A0")
bmj_route <- c(Standalone = "#8061A0", Coadministration = "#27886F")
bmj_programme <- c(Maternal = "#27886F", Adolescent = "#BD782B")
horizon_colours <- c("20" = "#236B9B", "30" = "#C05B19", "50" = "#7956A1")
horizon_shapes <- c("20" = 16, "30" = 17, "50" = 15)

bmj_theme <- function(base_size = 8.5) {
  theme_classic(base_size = base_size, base_family = "Arial") +
    theme(text = element_text(colour = "#29353D"),
          axis.text = element_text(size = 7, colour = "#354149"),
          axis.title = element_text(size = 8),
          axis.line = element_line(colour = "#71808A", linewidth = .25),
          axis.ticks = element_line(colour = "#71808A", linewidth = .25),
          axis.ticks.length = grid::unit(1, "mm"),
          panel.grid.major.x = element_line(colour = "#E4E8EB", linewidth = .23),
          strip.background = element_blank(),
          strip.text = element_text(size = 7, face = "bold", margin = margin(2, 2, 2, 2)),
          panel.spacing = grid::unit(2.5, "mm"),
          plot.title = element_text(size = 9, face = "bold", hjust = 0),
          plot.title.position = "plot",
          plot.tag = element_text(size = 10, face = "bold"),
          legend.position = "bottom", legend.title = element_text(size = 7),
          legend.text = element_text(size = 7), legend.key.height = grid::unit(3, "mm"),
          legend.key.width = grid::unit(4, "mm"), legend.box.spacing = grid::unit(1, "mm"),
          plot.background = element_rect(fill = "white", colour = NA),
          plot.margin = margin(4, 5, 4, 4))
}
apply_figure_style <- function() theme_set(bmj_theme())
apply_figure_style()

save_figure <- function(number, plot, panels, data, caption, height = 165, width = 180) {
  stem <- paste0("figure", number)
  stopifnot(length(panels) > 0L, identical(names(panels), LETTERS[seq_along(panels)]),
            identical(names(panels), names(data)),
            all(vapply(data, function(d) is.data.frame(d) && nrow(d) > 0, logical(1))))
  out <- "build/figures"
  for (subdir in c("panels", "source_data", "qa")) {
    dir.create(file.path(out, subdir), recursive = TRUE, showWarnings = FALSE)
  }
  base <- file.path(out, stem)
  save_plot <- function(filename, object, device, w, h, ...) {
    ggsave(filename, plot = object, device = device, units = "mm", width = w, height = h,
           bg = "white", ..., limitsize = FALSE)
  }
  save_plot(paste0(base, ".pdf"), plot, grDevices::cairo_pdf, width, height, family = "Arial")
  save_plot(paste0(base, ".svg"), plot, svglite::svglite, width, height)
  save_plot(paste0(base, ".png"), plot, ragg::agg_png, width, height, dpi = 600)
  save_plot(paste0(base, ".tiff"), plot, ragg::agg_tiff, width, height, dpi = 600, compression = "lzw")
  save_plot(file.path(out, "qa", paste0(stem, "_preview.png")), plot, ragg::agg_png, width, height, dpi = 160)
  for (letter in names(panels)) {
    key <- paste(stem, letter, sep = "_")
    file <- file.path(out, "panels", key)
    save_plot(paste0(file, ".pdf"), panels[[letter]], grDevices::cairo_pdf, width, 100, family = "Arial")
    save_plot(paste0(file, ".svg"), panels[[letter]], svglite::svglite, width, 100)
    save_plot(paste0(file, ".png"), panels[[letter]], ragg::agg_png, width, 100, dpi = 160)
    write.csv(data[[letter]], file.path(out, "source_data", paste0(key, ".csv")),
              row.names = FALSE, na = "")
  }
  writeLines(caption, paste0(base, ".legend.md"), useBytes = TRUE)
  inputs <- get0("input_files", envir = parent.frame(), inherits = FALSE, ifnotfound = character())
  stopifnot(all(file.exists(inputs)))
  jsonlite::write_json(list(figure = stem, panels = length(panels), width_mm = width,
                            height_mm = height, raster_dpi = 600,
                            input_files = as.list(unique(inputs)),
                            source_rows = lapply(data, nrow)),
                       paste0(base, ".metadata.json"), auto_unbox = TRUE, pretty = TRUE)
  invisible(base)
}
