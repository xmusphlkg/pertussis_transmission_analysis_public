library(testthat)

Sys.setenv(SHINYAPP_ROOT = normalizePath(file.path(getwd(), ".."), mustWork = FALSE))
test_check("ShinyApp")
