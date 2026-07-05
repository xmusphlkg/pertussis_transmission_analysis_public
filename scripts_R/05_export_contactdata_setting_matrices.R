args <- commandArgs(FALSE)
file_arg <- sub("^--file=", "", args[grepl("^--file=", args)])
script_dir <- if (length(file_arg) > 0) dirname(normalizePath(file_arg[[1]])) else file.path(getwd(), "scripts_R")
source(file.path(script_dir, "10_shared.R"))

suppressPackageStartupMessages({
  library(contactdata)
  library(readr)
  library(yaml)
})

settings <- yaml::read_yaml(model_path("config", "model_settings.yaml"))
countries <- settings$runtime$data_sources$countries
locations <- c("all", "home", "school", "work", "other")
contactdata_aliases <- c(
  "United Kingdom" = "UK",
  "United States" = "US"
)

age_to_bin <- function(x) {
  lower <- as.integer(sub("_.*$", "", x))
  sprintf("[%02d,%02d)", lower, lower + 5L)
}

rows <- list()
row_idx <- 1L
for (code in names(countries)) {
  meta <- countries[[code]]
  country_key <- meta$config_key
  contactdata_country <- meta$contactdata_country
  if (contactdata_country %in% names(contactdata_aliases)) {
    contactdata_country <- unname(contactdata_aliases[[contactdata_country]])
  }
  data_source <- as.character(meta$contactdata_source)
  for (location in locations) {
    matrix <- contactdata::contact_matrix(
      contactdata_country,
      location = location,
      data_source = data_source
    )
    for (source_age in rownames(matrix)) {
      for (target_age in colnames(matrix)) {
        rows[[row_idx]] <- data.frame(
          country = country_key,
          contactdata_country = contactdata_country,
          data_source = data_source,
          location = location,
          source_age_bin = age_to_bin(source_age),
          target_age_bin = age_to_bin(target_age),
          contacts_per_day = as.numeric(matrix[source_age, target_age]),
          stringsAsFactors = FALSE
        )
        row_idx <- row_idx + 1L
      }
    }
  }
}

out <- dplyr::bind_rows(rows)
raw_dir <- model_path("data", "raw", "external")
dir.create(raw_dir, recursive = TRUE, showWarnings = FALSE)
readr::write_csv(out, file.path(raw_dir, "contactdata_prem_contact_matrices_by_location_16.csv"))
readr::write_csv(
  out |>
    dplyr::filter(location == "all") |>
    dplyr::select(-location),
  file.path(raw_dir, "contactdata_prem_contact_matrices_16.csv")
)

message("Wrote contactdata all/home/school/work/other matrices for ", length(countries), " countries.")
