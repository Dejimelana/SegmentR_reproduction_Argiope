## Argiope x SegmentR — one-time setup, run from R.
##
##   source("setup.R")        # installs the R packages, then reports what is still missing
##   argiope_status()         # check at any time
##   argiope_install_python() # only if you have no suitable Python yet (downloads ~500 MB)
##
## Sourcing this file installs the three small R packages and then only *reports* on the
## Python side. Nothing large is downloaded unless you call argiope_install_python().

.BUNDLE <- local({
  for (i in seq_len(sys.nframe())) {
    ofile <- sys.frame(i)$ofile
    if (!is.null(ofile)) return(dirname(normalizePath(ofile, winslash = "/", mustWork = FALSE)))
  }
  normalizePath(".", winslash = "/")
})

R_PACKAGES <- c("jsonlite", "jpeg", "png")
PY_PACKAGES <- c("numpy", "opencv-python", "scikit-learn", "scikit-image",
                 "matplotlib", "pillow", "scipy", "segmentation-models-pytorch")

## ---------------------------------------------------------------- R packages

install_r_packages <- function() {
  missing <- R_PACKAGES[!vapply(R_PACKAGES, requireNamespace, logical(1), quietly = TRUE)]
  if (!length(missing)) {
    message("R packages: all present (", paste(R_PACKAGES, collapse = ", "), ")")
    return(invisible(character()))
  }
  message("Installing R packages: ", paste(missing, collapse = ", "))
  install.packages(missing, repos = "https://cloud.r-project.org")
  invisible(missing)
}

## ---------------------------------------------------------------- Python side

#' Where this bundle looks for Python, and whether that interpreter can actually run it.
argiope_python_status <- function() {
  cfg <- file.path(.BUNDLE, "python_path.txt")
  py <- ""
  if (file.exists(cfg)) py <- trimws(readLines(cfg, warn = FALSE))[1]
  if (!nzchar(py)) py <- Sys.getenv("ARGIOPE_PYTHON", unset = "")
  if (!nzchar(py)) py <- unname(Sys.which("python"))
  if (!nzchar(py) || !file.exists(py)) {
    return(list(python = NA_character_, ok = FALSE, missing = PY_PACKAGES))
  }
  probe <- c(
    "import importlib.util as u",
    "mods = {'numpy':'numpy','cv2':'opencv-python','sklearn':'scikit-learn',",
    "        'skimage':'scikit-image','matplotlib':'matplotlib','PIL':'pillow',",
    "        'scipy':'scipy','torch':'torch',",
    "        'segmentation_models_pytorch':'segmentation-models-pytorch'}",
    "print(','.join(p for m, p in mods.items() if u.find_spec(m) is None))")
  tmp <- tempfile(fileext = ".py")
  writeLines(probe, tmp)
  out <- suppressWarnings(system2(py, shQuote(tmp), stdout = TRUE, stderr = TRUE))
  unlink(tmp)
  missing <- if (length(out)) trimws(out[length(out)]) else ""
  missing <- if (nzchar(missing)) strsplit(missing, ",")[[1]] else character()
  list(python = py, ok = length(missing) == 0, missing = missing)
}

#' Record the interpreter this bundle should use, so setup happens only once.
argiope_use_python <- function(path) {
  if (!file.exists(path)) stop("no such interpreter: ", path, call. = FALSE)
  writeLines(normalizePath(path, winslash = "/"), file.path(.BUNDLE, "python_path.txt"))
  message("Recorded in python_path.txt: ", normalizePath(path, winslash = "/"))
  invisible(path)
}

#' Create a Python environment for this bundle, from R, and record it.
#'
#' Uses reticulate's miniconda. CPU-only torch by default: roughly 500 MB instead of 2.5 GB,
#' and plenty for inference over a folder of photographs. Pass gpu = TRUE for the CUDA build.
argiope_install_python <- function(envname = "argiope-segmentr", gpu = FALSE) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    install.packages("reticulate", repos = "https://cloud.r-project.org")
  }
  have_conda <- tryCatch({ reticulate::conda_binary(); TRUE }, error = function(e) FALSE)
  if (!have_conda) {
    message("Installing miniconda (via reticulate) ...")
    reticulate::install_miniconda()
  }
  envs <- tryCatch(reticulate::conda_list()$name, error = function(e) character())
  if (!(envname %in% envs)) {
    message("Creating conda environment: ", envname)
    reticulate::conda_create(envname, python_version = "3.11")
  }
  message("Installing torch (", if (gpu) "CUDA" else "CPU", ") ...")
  reticulate::conda_install(
    envname, packages = "torch", pip = TRUE,
    pip_options = if (gpu) character() else "--index-url https://download.pytorch.org/whl/cpu")
  message("Installing ", paste(PY_PACKAGES, collapse = ", "), " ...")
  reticulate::conda_install(envname, packages = PY_PACKAGES, pip = TRUE)

  py <- reticulate::conda_python(envname)
  argiope_use_python(py)

  # Verify before declaring success. Installing torch and the rest happens in two pip
  # passes, and a run started in between sees a half-built environment: every image comes
  # back "No module named 'segmentation_models_pytorch'", which looks like a segmentation
  # failure and is not one.
  st <- argiope_python_status()
  if (!isTRUE(st$ok)) {
    stop("The environment was created but is incomplete - missing: ",
         paste(st$missing, collapse = ", "),
         "
  Re-run argiope_install_python() to finish it, then argiope_status().",
         call. = FALSE)
  }
  message("Python environment ready: ", py)
  invisible(py)
}

## ---------------------------------------------------------------- report

argiope_status <- function() {
  cat("Bundle:", .BUNDLE, "\n\n")
  r_missing <- R_PACKAGES[!vapply(R_PACKAGES, requireNamespace, logical(1), quietly = TRUE)]
  cat("R packages   :",
      if (length(r_missing)) paste("MISSING:", paste(r_missing, collapse = ", ")) else "ok", "\n")

  ck <- file.path(.BUNDLE, "checkpoints", "opistho_unet.pt")
  cat("Checkpoint   :",
      if (file.exists(ck)) sprintf("ok (%.0f MB)", file.size(ck) / 1e6)
      else "MISSING checkpoints/opistho_unet.pt", "\n")

  st <- argiope_python_status()
  if (is.na(st$python)) {
    cat("Python       : none found\n")
  } else {
    cat("Python       :", st$python, "\n")
    cat("Python pkgs  :",
        if (st$ok) "ok" else paste("MISSING:", paste(st$missing, collapse = ", ")), "\n")
  }

  ready <- !length(r_missing) && file.exists(ck) && isTRUE(st$ok)
  cat("\n", if (ready) "READY - run: source(\"run_demo.R\")"
      else "NOT READY - see above; argiope_install_python() builds the Python side for you",
      "\n", sep = "")
  invisible(ready)
}

install_r_packages()
argiope_status()
