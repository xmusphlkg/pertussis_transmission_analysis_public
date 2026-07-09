## Lightweight plotting bootstrap ---------------------------------------------
## Sources manuscript styling and helper functions without eagerly loading every
## publication summary table. Figure modules should use explicit load_*()
## functions for the data they actually need.

if (exists("script_dir", inherits = TRUE) && file.exists(file.path(get("script_dir"), "10_shared.R"))) {
  script_dir <- normalizePath(get("script_dir"), mustWork = FALSE)
} else if (file.exists(file.path(getwd(), "scripts_R", "10_shared.R"))) {
  script_dir <- normalizePath(file.path(getwd(), "scripts_R"), mustWork = FALSE)
} else {
  bootstrap_file <- tryCatch(normalizePath(sys.frame(1)$ofile), error = function(e) NA_character_)
  if (is.na(bootstrap_file)) {
    bootstrap_file <- file.path(getwd(), "scripts_R", "lib", "bootstrap.R")
  }
  script_dir <- normalizePath(file.path(dirname(bootstrap_file), ".."), mustWork = FALSE)
}

old_skip_shared_data <- getOption("pertussis.skip_shared_data", default = NULL)
options(pertussis.skip_shared_data = TRUE)
source(file.path(script_dir, "10_shared.R"))
if (is.null(old_skip_shared_data)) {
  options(pertussis.skip_shared_data = NULL)
} else {
  options(pertussis.skip_shared_data = old_skip_shared_data)
}

source(file.path(script_dir, "lib", "source_data.R"))
