## Everything the notebook needs on the R side. Run once.
##   Rscript install_R_packages.R
pkgs <- c("jsonlite", "jpeg", "png")
missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) {
  install.packages(missing, repos = "https://cloud.r-project.org")
} else {
  message("All three already installed.")
}

## To open the notebook itself you also need an R kernel for Jupyter:
##   install.packages("IRkernel")
##   IRkernel::installspec(name = "ir", displayname = "R")
