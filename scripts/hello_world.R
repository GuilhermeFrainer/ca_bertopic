# An advanced Hello World script in R using libraries
library(jsonlite)
library(stm)

cat("==========================================\n")
cat("Hello from R + renv inside Docker!\n")
cat("==========================================\n")

# Test jsonlite
test_list <- list(
  status = "success",
  message = "If you can see this, jsonlite is working!",
  timestamp = Sys.time()
)
cat("JSON test output:\n")
cat(toJSON(test_list, auto_unbox = TRUE, pretty = TRUE), "\n")

# Test stm (just print version)
cat("\nSTM package version:", as.character(packageVersion("stm")), "\n")

cat("\nFull Session Info:\n")
print(sessionInfo())
