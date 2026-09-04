## =============================================================================
##  argiope_pipeline_segmentR.R
##  Opisthosoma masks and CIELAB palettes from a folder of spider photographs.
##  Everything, in one file, in reading order.
## =============================================================================
##
##  WHAT THIS DOES
##  --------------
##  Give it a folder of photographs. For each one it produces a mask of the
##  opisthosoma (the spider's abdomen), the dominant colours of that region in
##  CIELAB, a paginated gallery and a four-panel dashboard per specimen.
##
##  WHERE THE WORK HAPPENS
##  ----------------------
##  The segmenter is a U-Net trained in PyTorch. R does not reimplement it. R
##  calls a Python script ONCE for the whole folder (one model load, not one per
##  image), and then reads that run's own artefacts and draws everything itself.
##
##      R  ──►  adapt_unet.py  ──►  U-Net (torch)          ──►  mask
##                             └──►  CIELAB palette, JSON, cut-outs
##      R  ◄──  runs/<id>/ (masks, colors.csv, per-image JSON)
##      R  ──►  grid · dashboard · PDF
##
##  HOW TO RUN
##  ----------
##  Sections 1-3 only DEFINE functions and settings. Section 4 is the walkthrough
##  and it is the only part that runs anything.
##
##      step by step   open this file, put the cursor on a line or select a block,
##                     Ctrl+Enter. Every step prints something worth looking at
##                     before you move on. This is the intended way.
##
##      all at once    source("argiope_pipeline_segmentR.R")
##
##  Settings live in one place, section 2 (CFG). Change them there.
##
##  LAYOUT THIS FILE EXPECTS
##  ------------------------
##      <folder>/argiope_pipeline_segmentR.R    <- this file
##      <folder>/adapt_unet.py                  <- what R calls
##      <folder>/repro_segmentr.py              <- imported by adapt_unet.py
##      <folder>/argiope_unet.py                <- the segmenter class
##      <folder>/checkpoints/opistho_unet.pt    <- the trained weights
##      <folder>/sample_images/                 <- or any folder of images
##
##  RELATION TO THE OTHER FILES
##  ---------------------------
##  This file is self-contained and is the only R entry point in this folder.
##  setup.R is a subset of section 1.2, kept separately so you can check the machine
##  WITHOUT running the pipeline -- sourcing this file runs the walkthrough.
##
##  ORDER NOTE: helpers come first and refer to constants defined in section 2.
##  That is fine in R — a function body resolves its names when it is CALLED,
##  and nothing is called until section 3.
##
##  Requires: jsonlite, jpeg, png (R) — see argiope_status().
## =============================================================================


## #############################################################################
##  SECTION 1 — AUXILIARY FUNCTIONS
##  Nothing here runs anything: these are the pieces the pipeline uses.
## #############################################################################

## -----------------------------------------------------------------------------
## 1.1  Where am I?
##      Resolved when this file is LOADED, not when a function is later called:
##      the source() frame carrying $ofile is gone by then, and the --file=
##      fallback would point at whichever outer script did the sourcing.
## -----------------------------------------------------------------------------

.ARGIOPE_SCRIPT <- local({
  for (i in seq_len(sys.nframe())) {                 # source()d
    ofile <- sys.frame(i)$ofile
    if (!is.null(ofile)) return(normalizePath(ofile, winslash = "/", mustWork = FALSE))
  }
  ca <- commandArgs(trailingOnly = FALSE)            # Rscript this_file.R
  m <- grep("^--file=", ca, value = TRUE)
  if (length(m)) {
    return(normalizePath(sub("^--file=", "", m[1]), winslash = "/", mustWork = FALSE))
  }
  NA_character_
})

#' The folder this script lives in: where adapt_unet.py and checkpoints/ are looked for.
.here <- function() {
  if (is.na(.ARGIOPE_SCRIPT) || !nzchar(.ARGIOPE_SCRIPT)) {
    return(normalizePath(".", winslash = "/"))
  }
  dirname(.ARGIOPE_SCRIPT)
}

.need <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(sprintf('package "%s" is required: install.packages("%s")', pkg, pkg), call. = FALSE)
  }
}


## -----------------------------------------------------------------------------
## 1.2  Dependencies — the R side, and checking the Python side
## -----------------------------------------------------------------------------

#' Install the R packages this file needs, skipping any already present.
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

#' Which interpreter will be used, and can it actually run the pipeline?
#'
#' Returns the path, whether every Python package is importable, and the names of
#' any that are not. Checking this BEFORE running matters: a half-installed
#' environment makes every image fail with an import error, which reads like a
#' segmentation failure and is not one.
argiope_python_status <- function() {
  cfg <- file.path(.here(), "python_path.txt")
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

#' Record which interpreter this folder should use, so setup happens only once.
argiope_use_python <- function(path) {
  if (!file.exists(path)) stop("no such interpreter: ", path, call. = FALSE)
  writeLines(normalizePath(path, winslash = "/"), file.path(.here(), "python_path.txt"))
  message("Recorded in python_path.txt: ", normalizePath(path, winslash = "/"))
  invisible(path)
}

#' Build a Python environment for this folder, from R.
#'
#' CPU-only torch by default: roughly 500 MB instead of 2.5 GB, and plenty for
#' inference. It verifies before declaring success — torch and the rest install in
#' two pip passes, and anything started in between sees a half-built environment.
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

  st <- argiope_python_status()
  if (!isTRUE(st$ok)) {
    stop("The environment was created but is incomplete - missing: ",
         paste(st$missing, collapse = ", "),
         "\n  Re-run argiope_install_python() to finish it, then argiope_status().",
         call. = FALSE)
  }
  message("Python environment ready: ", py)
  invisible(py)
}

#' One report: R packages, weights, interpreter, Python packages. TRUE if ready.
argiope_status <- function() {
  cat("Folder:", .here(), "\n\n")
  r_missing <- R_PACKAGES[!vapply(R_PACKAGES, requireNamespace, logical(1), quietly = TRUE)]
  cat("R packages   :",
      if (length(r_missing)) paste("MISSING:", paste(r_missing, collapse = ", ")) else "ok", "\n")

  ck <- .default_weights()
  cat("Checkpoint   :",
      if (!is.null(ck)) sprintf("ok (%.0f MB)", file.size(ck) / 1e6)
      else "MISSING checkpoints/opistho_unet.pt", "\n")

  st <- argiope_python_status()
  if (is.na(st$python)) {
    cat("Python       : none found\n")
  } else {
    cat("Python       :", st$python, "\n")
    cat("Python pkgs  :",
        if (st$ok) "ok" else paste("MISSING:", paste(st$missing, collapse = ", ")), "\n")
  }

  ready <- !length(r_missing) && !is.null(ck) && isTRUE(st$ok)
  cat("\n", if (ready) "READY" else
      "NOT READY - argiope_install_python() builds the Python side for you", "\n", sep = "")
  invisible(ready)
}


## -----------------------------------------------------------------------------
## 1.3  Finding Python and the weights
## -----------------------------------------------------------------------------

#' Resolution order: argument, options(argiope.python=), $ARGIOPE_PYTHON,
#' python_path.txt beside this file, the interpreter next to an `argiope`
#' executable, then PATH. Fails with a message that says what to set.
argiope_python <- function(python = NULL) {
  if (!is.null(python) && nzchar(python)) return(python)
  opt <- getOption("argiope.python")
  if (!is.null(opt) && nzchar(opt)) return(opt)
  env <- Sys.getenv("ARGIOPE_PYTHON", unset = "")
  if (nzchar(env)) return(env)
  cfg <- file.path(.here(), "python_path.txt")
  if (file.exists(cfg)) {
    recorded <- trimws(readLines(cfg, warn = FALSE))[1]
    if (length(recorded) && nzchar(recorded) && file.exists(recorded)) return(recorded)
  }
  exe <- getOption("argiope.executable", default = unname(Sys.which("argiope")))
  if (!is.null(exe) && nzchar(exe)) {
    cand <- file.path(dirname(dirname(exe)), "python.exe")
    if (file.exists(cand)) return(cand)
  }
  found <- unname(Sys.which("python"))
  if (nzchar(found)) return(found)
  stop("Could not find a Python interpreter.\n",
       '  options(argiope.python = ".../python.exe")  or  argiope_install_python()',
       call. = FALSE)
}

#' The weights shipped beside this file, or NULL if there are none.
.default_weights <- function() {
  p <- file.path(.here(), "checkpoints", "opistho_unet.pt")
  if (file.exists(p)) p else NULL
}


## -----------------------------------------------------------------------------
## 1.4  Reading images, cheaply
##      Full-resolution photographs are large; displays are not. Decimation keeps
##      drawing fast. Statistics are always computed at full resolution.
## -----------------------------------------------------------------------------

.read_image <- function(path) {
  ext <- tolower(tools::file_ext(path))
  if (ext %in% c("jpg", "jpeg")) { .need("jpeg"); jpeg::readJPEG(path) }
  else { .need("png"); png::readPNG(path) }
}

.as_gray <- function(a) if (length(dim(a)) == 3L) a[, , 1] else a

#' How much to decimate so the longest side is at most `maxdim`.
.decimate_k <- function(dims, maxdim) max(1L, as.integer(ceiling(max(dims) / maxdim)))

#' Take every k-th row and column. The image and its mask must use the SAME k or
#' they stop lining up.
.decimate <- function(a, k) {
  if (k <= 1L) return(a)
  d <- dim(a)
  ri <- seq(1L, d[1], by = k); ci <- seq(1L, d[2], by = k)
  if (length(d) == 3L) a[ri, ci, , drop = FALSE] else a[ri, ci, drop = FALSE]
}


## -----------------------------------------------------------------------------
## 1.5  Masks: outlining and compositing
##      No morphology package needed — a dilation is four shifts and an OR.
## -----------------------------------------------------------------------------

.dilate <- function(m, r = 1L) {
  for (i in seq_len(r)) {
    up <- rbind(m[-1, , drop = FALSE], FALSE)
    dn <- rbind(FALSE, m[-nrow(m), , drop = FALSE])
    lf <- cbind(m[, -1, drop = FALSE], FALSE)
    rt <- cbind(FALSE, m[, -ncol(m), drop = FALSE])
    m <- m | up | dn | lf | rt
  }
  m
}

#' The photograph with the mask outlined and everything outside it pushed back.
#' The ring is drawn OUTSIDE the mask, so no masked pixel is hidden by it.
.compose <- function(rgb, mask, dim_mul = 0.55, dim_add = 0.10, ring = 2L) {
  if (length(dim(rgb)) == 2L) rgb <- array(rgb, c(dim(rgb), 3))
  out <- rgb[, , 1:3, drop = FALSE]
  outside <- !mask
  for (ch in 1:3) {
    v <- out[, , ch]; v[outside] <- v[outside] * dim_mul + dim_add; out[, , ch] <- v
  }
  edge <- .dilate(mask, ring) & !mask
  for (ch in 1:3) { v <- out[, , ch]; v[edge] <- ACCENT_RGB[ch]; out[, , ch] <- v }
  out
}

#' Load one photograph and its mask, decimated together, ready to draw.
.load_composite <- function(item, maxdim = 520) {
  rgb <- .read_image(item$path)
  k <- .decimate_k(dim(rgb)[1:2], maxdim)
  frame_px <- prod(dim(rgb)[1:2])
  rgb <- .decimate(rgb, k)
  if (!isTRUE(item$has_mask)) return(list(img = rgb, frac = NA_real_, empty = TRUE))
  mask <- .as_gray(.read_image(item$mask)) > 0.5
  if (!all(dim(mask) == dim(rgb)[1:2])) mask <- .decimate(mask, k)
  if (!all(dim(mask) == dim(rgb)[1:2])) {
    return(list(img = rgb, frac = NA_real_, empty = TRUE, note = "mask/image size mismatch"))
  }
  list(img = .compose(rgb, mask), frac = sum(mask) * (k^2) / frame_px, empty = FALSE)
}


## -----------------------------------------------------------------------------
## 1.6  Colour: assigning pixels to palette centroids
##      Done in CIELAB with grDevices::convertColor — base R, and the same call
##      the original SegmentR used. This reproduces the pipeline's own rule
##      rather than approximating it.
## -----------------------------------------------------------------------------

.assign_clusters <- function(rgb_px, centers_lab) {
  lab <- grDevices::convertColor(rgb_px, from = "sRGB", to = "Lab")
  d2 <- vapply(seq_len(nrow(centers_lab)),
               function(i) colSums((t(lab) - centers_lab[i, ])^2),
               numeric(nrow(lab)))
  if (is.null(dim(d2))) d2 <- matrix(d2, nrow = nrow(lab))
  max.col(-d2, ties.method = "first")
}


## -----------------------------------------------------------------------------
## 1.7  Reading a finished run
## -----------------------------------------------------------------------------

#' Is a cached run still the answer to the question being asked?
#'
#' Reusing blindly is how a run made against a half-installed environment, or
#' against a different folder, keeps being served long after the cause is fixed.
.reusable <- function(run_dir, dir, n = NULL) {
  cfg <- tryCatch(jsonlite::fromJSON(file.path(run_dir, "run_config.json")),
                  error = function(e) NULL)
  smry <- tryCatch(jsonlite::fromJSON(file.path(run_dir, "summary.json")),
                   error = function(e) NULL)
  if (is.null(cfg) || is.null(smry)) return(list(ok = FALSE, why = "it is unreadable"))

  want <- normalizePath(dir, winslash = "/", mustWork = FALSE)
  got <- normalizePath(as.character(cfg$images)[1], winslash = "/", mustWork = FALSE)
  if (!identical(tolower(want), tolower(got))) {
    return(list(ok = FALSE, why = paste0("it was made from a different folder (", got, ")")))
  }
  if (isTRUE(as.integer(smry$processed)[1] == 0L)) {
    return(list(ok = FALSE, why = "it produced no masks at all, so it is not worth keeping"))
  }
  if (is.null(n)) {
    n_now <- length(list.files(dir, pattern = "[.](jpe?g|png)$", recursive = TRUE,
                               ignore.case = TRUE))
    if (!is.null(smry$n_images) && n_now != as.integer(smry$n_images)[1]) {
      return(list(ok = FALSE,
                  why = sprintf("the folder now holds %d images, the run covered %d",
                                n_now, as.integer(smry$n_images)[1])))
    }
  }
  list(ok = TRUE, why = "")
}

#' The detection stored beside a mask: its label, score and box.
.read_detection <- function(g, item) {
  stem <- tools::file_path_sans_ext(item$image)
  jp <- file.path(g$run_dir, "json", paste0(item$group, "__", stem, ".json"))
  if (!file.exists(jp)) return(NULL)
  recs <- jsonlite::fromJSON(jp, simplifyVector = FALSE)
  if (!length(recs)) return(NULL)
  d <- recs[[1]]
  list(label = d$label, score = d$score,
       box = unlist(d$box)[c("xmin", "ymin", "xmax", "ymax")])
}

#' The rows a plot should draw, after filtering and selection.
.selected <- function(g, select, include_empty) {
  it <- g$items
  if (!include_empty) it <- it[it$has_mask, ]
  if (!is.null(select)) {
    it <- if (is.numeric(select)) it[select, , drop = FALSE] else
      it[it$image %in% select, , drop = FALSE]
  }
  it[stats::complete.cases(it$image), , drop = FALSE]
}

#' A bare "nothing to plot" hides the cause; name it.
.explain_empty <- function(g) {
  reasons <- g$items$reason[!g$items$has_mask]
  reasons <- reasons[!is.na(reasons) & nzchar(reasons)]
  if (!length(reasons)) {
    message("Nothing to plot: no image in this gallery has a mask.")
    return(invisible(NULL))
  }
  tb <- sort(table(reasons), decreasing = TRUE)
  message("Nothing to plot: every image was skipped. Most common reason: \"",
          names(tb)[1], "\" (", tb[[1]], " of ", length(reasons), ").")
  if (grepl("No module named", names(tb)[1])) {
    message("  That is a Python environment problem, not a segmentation one. ",
            "Run argiope_status(), fix it, then run again.")
  }
  invisible(NULL)
}


## #############################################################################
##  SECTION 2 — CONFIGURATION
##  Every knob in one place. Change things here, not in the code above.
## #############################################################################

## What to process, and where the results go ----------------------------------
CFG <- list(
  images   = "sample_images",  # folder of .jpg/.jpeg/.png, searched recursively
                               #   also bundled: "crops" (40 annotation crops)
                               #   and "random_ArTaxOr+GBIF" (40 unseen spiders)
  out      = "runs",           # where run directories are written
  run_id   = "demo",           # name of this run's directory
  n        = NULL,             # NULL = the whole folder; a number = seeded sample
  seed     = 42,               # sampling seed
  n_colors = 5,                # colour clusters per mask
  weights  = NULL,             # NULL = checkpoints/opistho_unet.pt beside this file
  device   = NULL,             # NULL = "cuda" if available, else "cpu"
  reuse    = TRUE,             # reuse a matching finished run instead of recomputing

  ## Drawing
  per_page = 6,                # cells per gallery page
  maxdim   = 520,              # longest side of the rasters drawn (bigger = slower)
  pdf_file = "demo_gallery.pdf"
)

## Packages ------------------------------------------------------------------
R_PACKAGES <- c("jsonlite", "jpeg", "png")
PY_PACKAGES <- c("numpy", "opencv-python", "scikit-learn", "scikit-image",
                 "matplotlib", "pillow", "scipy", "segmentation-models-pytorch")

## Colours used by the plots (not the spiders' — the page's) -------------------
ACCENT_RGB <- c(232, 198, 74) / 255   # the mask outline
INK        <- "#191C11"
MUTED      <- "#7C8270"
SPECIES_LABEL <- c(argiope_argentata = "A. argentata",
                   argiope_aurantia  = "A. aurantia",
                   argiope_bruennichi = "A. bruennichi")


## #############################################################################
##  SECTION 3 — THE PIPELINE
##  One function per stage: segment · load · inspect · choose · grid · dashboard
##  · export. Defining them runs nothing; section 4 is where they get used.
## #############################################################################

## -----------------------------------------------------------------------------
## 3.1  STEP 1 — segment a folder
##      Shells out to adapt_unet.py once. Everything after this reads artefacts.
## -----------------------------------------------------------------------------

argiope_gallery <- function(dir, out = NULL, run_id = "r-gallery", n = NULL, seed = 42,
                            n_colors = 5, python = NULL, adapter = NULL, reuse = TRUE,
                            quiet = FALSE, weights = NULL, device = NULL) {
  .need("jsonlite")
  if (!dir.exists(dir) && !file.exists(dir)) stop("no such directory: ", dir, call. = FALSE)
  if (is.null(out)) out <- file.path(tempdir(), "argiope_gallery")
  ## absolutise before handing it over: adapt_unet.py resolves a relative --out
  ## against the project root, not against R's working directory
  if (!dir.exists(out)) dir.create(out, recursive = TRUE, showWarnings = FALSE)
  out <- normalizePath(out, winslash = "/", mustWork = TRUE)
  if (is.null(adapter)) adapter <- file.path(.here(), "adapt_unet.py")
  if (!file.exists(adapter)) stop("adapt_unet.py not found: ", adapter, call. = FALSE)

  run_dir <- file.path(out, run_id)
  done <- file.exists(file.path(run_dir, "summary.json"))
  if (reuse && done) {
    verdict <- .reusable(run_dir, dir, n)
    if (!isTRUE(verdict$ok)) {
      done <- FALSE
      if (!quiet) message("Not reusing the cached run: ", verdict$why)
    }
  }

  if (!(reuse && done)) {
    args <- c(shQuote(normalizePath(adapter, winslash = "/")),
              "--images", shQuote(normalizePath(dir, winslash = "/")),
              "--out", shQuote(out), "--run-id", shQuote(run_id),
              "--n-colors", n_colors, "--seed", seed, "--no-qa")
    if (!is.null(n)) args <- c(args, "--n", n)
    if (is.null(weights)) weights <- .default_weights()
    if (!is.null(weights)) {
      args <- c(args, "--weights", shQuote(normalizePath(weights, winslash = "/")))
    }
    if (!is.null(device)) args <- c(args, "--device", device)

    if (!quiet) message("Running the pipeline over ", dir, " (one model load) ...")
    st <- system2(argiope_python(python), args,
                  stdout = if (quiet) FALSE else "", stderr = if (quiet) FALSE else "")
    if (!identical(as.integer(st)[1], 0L)) {
      stop("adapt_unet.py failed (exit ", st, ")", call. = FALSE)
    }
  } else if (!quiet) {
    message("Reusing the existing run in ", run_dir)
  }
  argiope_load_gallery(run_dir)
}


## -----------------------------------------------------------------------------
## 3.2  STEP 2 — load a finished run as a gallery object
## -----------------------------------------------------------------------------

argiope_load_gallery <- function(run_dir) {
  .need("jsonlite")
  cfg_path <- file.path(run_dir, "run_config.json")
  if (!file.exists(cfg_path)) stop("not a run directory: ", run_dir, call. = FALSE)
  cfg <- jsonlite::fromJSON(cfg_path, simplifyVector = TRUE)

  read_or_empty <- function(name) {
    p <- file.path(run_dir, name)
    if (file.exists(p)) utils::read.csv(p, stringsAsFactors = FALSE) else data.frame()
  }
  colors <- read_or_empty("colors.csv")
  skipped <- read_or_empty("skipped.csv")

  paths <- as.character(cfg$image_list$path)
  groups <- as.character(cfg$image_list$group)
  names_ <- basename(paths)
  stems <- tools::file_path_sans_ext(names_)

  items <- data.frame(
    image = names_, group = groups, path = paths,
    mask = file.path(run_dir, "json", "masks", paste0(groups, "__", stems, "_000.png")),
    stringsAsFactors = FALSE
  )
  items$has_mask <- file.exists(items$mask) & items$image %in% colors$image
  items$score <- NA_real_
  items$px <- NA_integer_
  if (nrow(colors)) {
    first <- colors[!duplicated(colors$image), c("image", "score", "mask_pixels")]
    idx <- match(items$image, first$image)
    items$score <- first$score[idx]
    items$px <- first$mask_pixels[idx]
  }
  items$reason <- NA_character_
  if (nrow(skipped)) items$reason <- skipped$reason[match(items$image, skipped$image)]

  structure(list(run_dir = run_dir, items = items, colors = colors, config = cfg),
            class = "argiope_gallery")
}


## -----------------------------------------------------------------------------
## 3.3  STEP 3 — inspect: the table and the palettes
##      Read these before looking at pictures. Images with no mask are never
##      dropped silently; they carry their reason.
## -----------------------------------------------------------------------------

argiope_items <- function(g) g$items

argiope_palette_of <- function(g, image) {
  rows <- g$colors[g$colors$image == image, ]
  if (!nrow(rows)) return(data.frame(hex = character(), coverage = numeric()))
  rows <- rows[order(-rows$cluster_size), ]
  data.frame(hex = rows$hex_color, coverage = rows$cluster_frac,
             lab_l = rows$lab_l, lab_a = rows$lab_a, lab_b = rows$lab_b,
             size = rows$cluster_size, stringsAsFactors = FALSE)
}

print.argiope_gallery <- function(x, ...) {
  it <- x$items
  cat("<argiope gallery>", x$run_dir, "\n")
  cat("  ", nrow(it), " images · ", sum(it$has_mask), " with a mask · ",
      sum(!it$has_mask), " empty\n", sep = "")
  if (any(it$has_mask)) {
    cat("  median score ", format(stats::median(it$score, na.rm = TRUE), digits = 3),
        " · pages of ", CFG$per_page, ": ", argiope_pages(x, CFG$per_page), "\n", sep = "")
  } else {
    reasons <- it$reason[!is.na(it$reason) & nzchar(it$reason)]
    if (length(reasons)) {
      tb <- sort(table(reasons), decreasing = TRUE)
      cat("  NOTHING SEGMENTED - most common reason: \"", names(tb)[1], "\" (",
          tb[[1]], " of ", length(reasons), ")\n", sep = "")
    }
  }
  invisible(x)
}


## -----------------------------------------------------------------------------
## 3.4  STEP 4 — choose which specimens to draw
##      Interactively these open a real selection list; in a script they return
##      everything, so batch use keeps working.
## -----------------------------------------------------------------------------

argiope_pick <- function(g, only_with_mask = TRUE, preselect = NULL) {
  it <- g$items
  if (only_with_mask) it <- it[it$has_mask, ]
  if (!nrow(it)) return(character())
  labels <- sprintf("%s  [%s]%s", it$image, it$group,
                    ifelse(is.na(it$score), "", sprintf("  score %.3f", it$score)))
  if (!interactive()) {
    message("Not an interactive session: selecting all ", nrow(it), " items.")
    return(it$image)
  }
  chosen <- utils::select.list(labels, preselect = preselect, multiple = TRUE,
                               title = "Select images to plot", graphics = TRUE)
  if (!length(chosen)) return(character())
  it$image[match(chosen, labels)]
}

argiope_pick_one <- function(g, only_with_mask = TRUE) {
  it <- g$items
  if (only_with_mask) it <- it[it$has_mask, ]
  if (!nrow(it)) return(NA_character_)
  labels <- sprintf("%s  [%s]%s", it$image, it$group,
                    ifelse(is.na(it$score), "", sprintf("  score %.3f", it$score)))
  if (!interactive()) return(it$image[1])
  chosen <- utils::select.list(labels, multiple = FALSE, graphics = TRUE,
                               title = "Select one image")
  if (!nzchar(chosen)) return(NA_character_)
  it$image[match(chosen, labels)]
}


## -----------------------------------------------------------------------------
## 3.5  STEP 5 — the gallery, as a paginated grid
##      One cell per specimen: photograph with the mask outlined, the palette bar
##      proportional to coverage, the leading HEX values, score and mask area.
## -----------------------------------------------------------------------------

#' Draw one cell. Called by argiope_plot; not useful on its own.
.draw_cell <- function(g, item, maxdim) {
  graphics::plot.new()
  graphics::plot.window(c(0, 1), c(0, 1))
  comp <- tryCatch(.load_composite(item, maxdim),
                   error = function(e) list(img = NULL, empty = TRUE,
                                            note = conditionMessage(e)))

  band <- c(0.23, 0.925)                       # vertical room for the photograph
  pin <- graphics::par("pin")
  if (!is.null(comp$img)) {
    d <- dim(comp$img)
    ar_img <- d[2] / d[1]
    bw <- pin[1]; bh <- pin[2] * diff(band)
    if (ar_img > bw / bh) { w <- 1; h <- (bw / ar_img) / pin[2] } else
                          { h <- diff(band); w <- (bh * ar_img) / pin[1] }
    x0 <- 0.5 - w / 2; y0 <- band[1] + (diff(band) - h) / 2
    graphics::rasterImage(comp$img, x0, y0, x0 + w, y0 + h, interpolate = TRUE)
  }

  graphics::text(0, 0.98, sub("^argiope_", "A. ", item$group), adj = c(0, 1),
                 cex = 0.78, font = 3, col = INK)
  graphics::text(1, 0.98, if (is.na(item$score)) "sin mascara" else
                 sprintf("score %.3f", item$score), adj = c(1, 1), cex = 0.7,
                 col = if (is.na(item$score)) "#8C5A3C" else "#8A6E12")
  graphics::text(0, 0.935, item$image, adj = c(0, 1), cex = 0.62, col = MUTED,
                 family = "mono")

  if (isTRUE(comp$empty)) {
    msg <- if (!is.null(comp$note)) comp$note else
      if (!is.na(item$reason)) item$reason else "no mask"
    graphics::text(0.5, 0.13, msg, cex = 0.72, col = "#8C5A3C")
    return(invisible(NULL))
  }

  pal <- argiope_palette_of(g, item$image)
  if (nrow(pal)) {
    x <- 0.02; y1 <- 0.19; y0 <- 0.12
    for (i in seq_len(nrow(pal))) {
      w <- 0.96 * pal$coverage[i]
      graphics::rect(x, y0, x + w, y1, col = pal$hex[i], border = NA)
      x <- x + w
    }
    graphics::rect(0.02, y0, 0.98, y1, border = "#B3BAA0", lwd = 0.6)
    k <- seq_len(min(3, nrow(pal)))
    lab <- paste(sprintf("%s %.0f%%", pal$hex[k], 100 * pal$coverage[k]), collapse = "   ")
    graphics::text(0.02, 0.075, lab, adj = c(0, 1), cex = 0.6, col = MUTED, family = "mono")
  }
  if (!is.na(comp$frac)) {
    graphics::text(0.98, 0.075,
                   sprintf("%s px · %.2f%%",
                           formatC(item$px, format = "d", big.mark = ".",
                                   decimal.mark = ","),
                           100 * comp$frac),
                   adj = c(1, 1), cex = 0.6, col = MUTED, family = "mono")
  }
  invisible(NULL)
}

#' How many pages the gallery needs. Zero when there is nothing to draw.
argiope_pages <- function(g, per_page = 6, select = NULL, include_empty = FALSE) {
  n <- nrow(.selected(g, select, include_empty))
  if (!n) return(0L)
  as.integer(ceiling(n / per_page))
}

#' Draw one page of the gallery.
argiope_plot <- function(g, page = 1, per_page = 6, select = NULL, include_empty = FALSE,
                         ncol = NULL, maxdim = 520) {
  it <- .selected(g, select, include_empty)
  n <- nrow(it)
  if (!n) { .explain_empty(g); return(invisible(NULL)) }
  pages <- max(1L, as.integer(ceiling(n / per_page)))
  page <- max(1L, min(as.integer(page), pages))
  idx <- seq((page - 1) * per_page + 1, min(page * per_page, n))
  it <- it[idx, , drop = FALSE]

  if (is.null(ncol)) ncol <- max(1L, min(nrow(it), as.integer(ceiling(sqrt(nrow(it) * 1.4)))))
  nrow_ <- as.integer(ceiling(nrow(it) / ncol))

  op <- graphics::par(mfrow = c(nrow_, ncol), mar = c(0.2, 0.4, 0.2, 0.4),
                      oma = c(1.6, 0.6, 2.2, 0.6), bg = "#F7F8F3", xaxs = "i", yaxs = "i")
  on.exit(graphics::par(op), add = TRUE)
  for (i in seq_len(nrow(it))) .draw_cell(g, it[i, ], maxdim)
  for (i in seq_len(nrow_ * ncol - nrow(it))) graphics::plot.new()

  graphics::mtext("Argiope · opisthosoma + pantone", side = 3, outer = TRUE, line = 0.7,
                  adj = 0, cex = 0.95, font = 2, col = INK)
  graphics::mtext(sprintf("page %d of %d · %d of %d images", page, pages, nrow(it), n),
                  side = 3, outer = TRUE, line = 0.7, adj = 1, cex = 0.7, col = MUTED)
  graphics::mtext(basename(g$run_dir), side = 1, outer = TRUE, line = 0.4, adj = 1,
                  cex = 0.6, col = MUTED)
  invisible(page)
}


## -----------------------------------------------------------------------------
## 3.6  STEP 6 — the dashboard for one specimen
##      Four panels: photograph with box and contour · dominant colours ·
##      RGB histogram of the masked pixels · the mask recoloured by centroid.
## -----------------------------------------------------------------------------

argiope_dashboard <- function(g, image = NULL, file = NULL, maxdim = 700,
                              width = 1500, height = 1000, res = 130) {
  .need("jsonlite")
  if (is.null(image)) image <- argiope_pick_one(g)
  if (is.na(image) || !nzchar(image)) { message("No image selected."); return(invisible(NULL)) }
  it <- g$items[g$items$image == image, , drop = FALSE]
  if (!nrow(it)) stop("no such image in this gallery: ", image, call. = FALSE)
  item <- it[1, ]
  if (!isTRUE(item$has_mask)) {
    stop("this image has no mask (", if (is.na(item$reason)) "empty" else item$reason,
         "), so there is nothing to describe", call. = FALSE)
  }

  ## full resolution for the numbers; decimated copies only for display
  rgb_full <- .read_image(item$path)
  if (length(dim(rgb_full)) == 2L) rgb_full <- array(rgb_full, c(dim(rgb_full), 3))
  rgb_full <- rgb_full[, , 1:3, drop = FALSE]
  mask_full <- .as_gray(.read_image(item$mask)) > 0.5
  if (!all(dim(mask_full) == dim(rgb_full)[1:2])) {
    stop("mask and image dimensions disagree for ", image, call. = FALSE)
  }

  pal <- argiope_palette_of(g, image)
  centers_lab <- as.matrix(pal[, c("lab_l", "lab_a", "lab_b")])
  centers_rgb <- pmin(pmax(grDevices::convertColor(centers_lab, from = "Lab",
                                                   to = "sRGB"), 0), 1)
  px <- cbind(rgb_full[, , 1][mask_full], rgb_full[, , 2][mask_full],
              rgb_full[, , 3][mask_full])
  n_px <- nrow(px)
  frac <- n_px / prod(dim(mask_full))
  assign_ <- .assign_clusters(px, centers_lab)

  det <- .read_detection(g, item)
  row0 <- g$colors[g$colors$image == image, ][1, ]

  if (!is.null(file)) {
    grDevices::png(file, width = width, height = height, res = res)
    on.exit(grDevices::dev.off(), add = TRUE)
  }
  op <- graphics::par(mfrow = c(2, 2), mar = c(3.2, 3.2, 2.4, 1.0),
                      oma = c(0.4, 0.4, 2.2, 0.4), bg = "#FFFFFF")
  on.exit(graphics::par(op), add = TRUE)

  ## panel 1 — photograph, detection box, mask contour
  k <- .decimate_k(dim(rgb_full)[1:2], maxdim)
  disp <- .decimate(rgb_full, k)
  dmask <- .decimate(mask_full, k)
  edge <- .dilate(dmask, 2L) & !dmask
  for (ch in 1:3) { v <- disp[, , ch]; v[edge] <- ACCENT_RGB[ch]; disp[, , ch] <- v }
  h <- dim(disp)[1]; w <- dim(disp)[2]
  graphics::plot.new(); graphics::plot.window(c(0, w), c(h, 0), asp = 1)
  graphics::rasterImage(disp, 0, h, w, 0, interpolate = TRUE)
  if (!is.null(det)) {
    b <- det$box / k
    graphics::rect(b[["xmin"]], b[["ymin"]], b[["xmax"]], b[["ymax"]],
                   border = "#3B4CC0", lwd = 1.6)
    graphics::text(b[["xmin"]], b[["ymin"]] - 4, sprintf("%s: %.2f", det$label, det$score),
                   adj = c(0, 1), cex = 0.62, col = "#3B4CC0")
  }
  graphics::title("detections + mask contours", cex.main = 0.95, font.main = 1, line = 0.6)

  ## panel 2 — dominant colours
  graphics::plot.new(); graphics::plot.window(c(0, nrow(pal)), c(-0.42, 1))
  for (i in seq_len(nrow(pal))) {
    graphics::rect(i - 1, 0, i, 1, col = pal$hex[i], border = NA)
    graphics::text(i - 0.5, -0.05,
                   sprintf("%s\n%.1f%%\nL%.0f a%.0f b%.0f", pal$hex[i],
                           100 * pal$coverage[i], pal$lab_l[i], pal$lab_a[i], pal$lab_b[i]),
                   adj = c(0.5, 1), cex = 0.6, family = "mono", col = INK)
  }
  graphics::title(sprintf("dominant colours (k-means in CIELAB)   mean %s  median %s",
                          row0$mean_color, row0$median_color),
                  cex.main = 0.85, font.main = 1, line = 0.6)

  ## panel 3 — RGB histogram of the masked pixels
  brk <- seq(0, 255, length.out = 65)
  hs <- lapply(1:3, function(ch) graphics::hist(px[, ch] * 255, breaks = brk, plot = FALSE))
  ymax <- max(vapply(hs, function(x) max(x$counts), numeric(1)))
  graphics::plot.new(); graphics::plot.window(c(0, 255), c(0, ymax * 1.04))
  for (ch in 1:3) {
    graphics::lines(c(0, rep(hs[[ch]]$breaks[-1], each = 2)),
                    c(0, rep(hs[[ch]]$counts, each = 2), 0)[seq_len(2 * length(brk) - 1)],
                    type = "s", col = c("red", "green3", "blue")[ch], lwd = 1)
  }
  graphics::axis(1, cex.axis = 0.75); graphics::axis(2, cex.axis = 0.75, las = 1)
  graphics::box(col = "#333333")
  graphics::title(sprintf("RGB histogram of masked pixels (n=%d)", n_px),
                  cex.main = 0.95, font.main = 1, line = 0.6)
  graphics::mtext("channel value", side = 1, line = 2, cex = 0.7)

  ## panel 4 — the mask recoloured by cluster centroid
  rec <- array(1, dim(rgb_full))
  for (ch in 1:3) {
    v <- rec[, , ch]; v[mask_full] <- centers_rgb[assign_, ch]; rec[, , ch] <- v
  }
  rec <- .decimate(rec, k)
  graphics::plot.new(); graphics::plot.window(c(0, w), c(h, 0), asp = 1)
  graphics::rasterImage(rec, 0, h, w, 0, interpolate = FALSE)
  graphics::title("mask recoloured by cluster centroid", cex.main = 0.95, font.main = 1,
                  line = 0.6)

  graphics::mtext(sprintf("%s / %s    mask=%.2f%% of frame",
                          item$group, item$image, 100 * frac),
                  side = 3, outer = TRUE, line = 0.4, cex = 0.85, col = INK)

  invisible(list(image = image, palette = pal, mask_px = n_px, mask_frac = frac,
                 mean_color = row0$mean_color, median_color = row0$median_color,
                 cluster_sizes = tabulate(assign_, nbins = nrow(pal))))
}


## -----------------------------------------------------------------------------
## 3.7  STEP 7 — export every page to one PDF
##      Rasters are embedded uncompressed, so lower maxdim for a large gallery.
## -----------------------------------------------------------------------------

argiope_pdf <- function(g, file = "argiope_gallery.pdf", per_page = 6, select = NULL,
                        include_empty = FALSE, width = 11, height = 8, maxdim = 520) {
  pages <- argiope_pages(g, per_page, select, include_empty)
  if (!pages) { .explain_empty(g); return(invisible(NULL)) }
  grDevices::pdf(file, width = width, height = height)
  on.exit(grDevices::dev.off(), add = TRUE)
  for (p in seq_len(pages)) {
    argiope_plot(g, page = p, per_page = per_page, select = select,
                 include_empty = include_empty, maxdim = maxdim)
  }
  message("Wrote ", pages, " page(s) -> ",
          normalizePath(file, winslash = "/", mustWork = FALSE))
  invisible(file)
}


## #############################################################################
##  SECTION 4 — RUN IT, ONE STEP AT A TIME
##
##  Sections 1-3 only DEFINE things; nothing has run yet. From here down the code
##  executes. Run it the way you prefer:
##
##    * step by step  - put the cursor on a line (or select a block) and press
##                      Ctrl+Enter in RStudio / VS Code. This is the intended way:
##                      every step prints something you should look at before
##                      moving on.
##    * all at once   - source("argiope_pipeline_segmentR.R"), which runs the
##                      whole of section 4 top to bottom.
##
##  Steps 1 and 2 are the only ones you must run in order. After that, 4 to 8 are
##  independent: re-run any of them as often as you like, they only read `g`.
## #############################################################################


## ---- STEP 0 · is the machine ready? -----------------------------------------
## Look for READY. If the Python side is incomplete every image will fail with an
## import error, which reads like a segmentation failure and is not one.

install_r_packages()
argiope_status()


## ---- STEP 1 · segment the folder --------------------------------------------
## The only slow step: it loads the model once and walks the whole folder. A
## finished matching run is reused, so running this line again is cheap.
## Look at: how many images came back with a mask.

g <- argiope_gallery(dir      = CFG$images,
                     out      = CFG$out,
                     run_id   = CFG$run_id,
                     n        = CFG$n,
                     seed     = CFG$seed,
                     n_colors = CFG$n_colors,
                     weights  = CFG$weights,
                     device   = CFG$device,
                     reuse    = CFG$reuse)

g                                    # the summary line


## ---- STEP 2 · the table, before any picture ---------------------------------
## One row per image. `score` is the segmenter's confidence, `px` the mask size.
## Look at: whether has_mask is TRUE where you expect it.

it <- argiope_items(g)
it[, c("image", "group", "has_mask", "score", "px")]


## ---- STEP 3 · what failed, and why ------------------------------------------
## Images with no mask are never dropped silently. If this is empty, good.
## A reason mentioning "No module named" is an environment problem, not the model.

sum(it$has_mask)                     # how many worked
subset(it, !has_mask)[, c("image", "reason")]


## ---- STEP 4 · the palette of one specimen -----------------------------------
## `coverage` is the share of the mask's pixels in that cluster; lab_* are the
## CIELAB coordinates of its centroid. HEX + Lab + coverage is the "pantone".
## Change `first` to any file name from step 2 to inspect a different specimen.

first <- it$image[it$has_mask][1]
first
argiope_palette_of(g, first)


## ---- STEP 5 · the gallery, page by page -------------------------------------
## Each cell: the photograph with the mask outlined, the palette bar proportional
## to coverage, the leading HEX values, the score and the mask area.
## Run the second line again with page = 2, 3, ... to walk through the pages.

argiope_pages(g, CFG$per_page)       # how many pages there are

argiope_plot(g, page = 1, per_page = CFG$per_page, maxdim = CFG$maxdim)


## ---- STEP 6 · draw only the specimens you choose ----------------------------
## In an interactive session argiope_pick() opens a real selection list; in a
## script it returns everything. You can also select by name or by index.

sel <- argiope_pick(g)
## sel <- it$image[it$has_mask][1:3]            # ... or by name
## sel <- 1:3                                   # ... or by position

argiope_plot(g, select = sel, per_page = CFG$per_page, maxdim = CFG$maxdim)


## ---- STEP 7 · the four-panel dashboard for one specimen ---------------------
## Photograph with box and contour · dominant colours · RGB histogram of the
## masked pixels · the mask recoloured by centroid. It also RETURNS the numbers,
## so the panel is not the only place they exist.

res <- argiope_dashboard(g, first)

res$mask_px
res$mean_color
res$cluster_sizes

## argiope_dashboard(g)                          # ... or pick one from a list
## argiope_dashboard(g, first, file = "card.png")  # ... or write it to a PNG


## ---- STEP 8 · export every page to one PDF ----------------------------------
## Rasters go into the PDF uncompressed, so lower maxdim for a large gallery.

argiope_pdf(g, CFG$pdf_file, per_page = CFG$per_page, maxdim = CFG$maxdim)


## ---- WHERE IT ALL LANDED ----------------------------------------------------
## run_config.json records every parameter: a run is reproducible from it alone.

file.path(CFG$out, CFG$run_id)
head(list.files(file.path(CFG$out, CFG$run_id), recursive = TRUE), 12)

## Your own images:
##   g <- argiope_gallery("C:/path/to/images", out = "runs", run_id = "mine")
