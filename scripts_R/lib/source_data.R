## Source-data helpers ---------------------------------------------------------

source_data_table_path <- function(filename) {
  model_path("outputs", "tables", filename)
}

write_source_data_table <- function(data, filename) {
  readr::write_csv(data, source_data_table_path(filename))
  invisible(source_data_table_path(filename))
}

write_source_data_list <- function(source_data, filename_map) {
  missing_names <- setdiff(names(source_data), names(filename_map))
  if (length(missing_names) > 0) {
    stop(
      "Missing source-data filename(s) for: ",
      paste(missing_names, collapse = ", "),
      call. = FALSE
    )
  }

  purrr::iwalk(source_data, function(data, name) {
    write_source_data_table(data, filename_map[[name]])
  })

  invisible(source_data)
}
