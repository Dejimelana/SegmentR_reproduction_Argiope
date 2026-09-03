## Argiope — thin R interface to the Python pipeline.
##
## Wraps, never reimplements. The opisthosoma segmenter is a trained U-Net living in torch;
## there is nothing to port to R. R calls the published CLI contract
## (`argiope describe <image> --json <out.json> --mask <mask.png>`) and reads the result.
## That boundary is a process boundary, not reticulate, so R needs no Python configuration
## and nothing breaks when the CUDA environment changes.
##
## Requires: jsonlite. Optional: png (only for `argiope_mask()`).
##
## Usage:
##   source("R/argiope.R")
##   options(argiope.executable = "C:/Users/.../envs/argiope/Scripts/argiope.exe")
##   d <- argiope_describe("spider.jpg")
##   argiope_palette(d)

.argiope_require <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(sprintf('package "%s" is required: install.packages("%s")', pkg, pkg), call. = FALSE)
  }
}

#' Locate the `argiope` executable.
#'
#' Resolution order: explicit argument, `options(argiope.executable=)`, the `ARGIOPE_EXE`
#' environment variable, then `PATH`. The PATH lookup only succeeds when R was launched
#' from the activated conda environment, which is the usual reason this fails.
argiope_executable <- function(executable = NULL) {
  if (!is.null(executable) && nzchar(executable)) return(executable)
  opt <- getOption("argiope.executable")
  if (!is.null(opt) && nzchar(opt)) return(opt)
  env <- Sys.getenv("ARGIOPE_EXE", unset = "")
  if (nzchar(env)) return(env)
  found <- unname(Sys.which("argiope"))
  if (nzchar(found)) return(found)
  stop(
    "Could not find the `argiope` executable.\n",
    "Either start R from the activated conda env, or point R at it:\n",
    '  options(argiope.executable = ".../envs/argiope/Scripts/argiope.exe")',
    call. = FALSE
  )
}

#' Describe one image: opisthosoma mask + CIELAB palette.
#'
#' @param image Path to the image.
#' @param outdir Where the JSON and mask PNG are written (default: a temp directory).
#' @param config Optional path to an alternative config YAML.
#' @param taxon Optional known taxon label.
#' @param executable Optional explicit path to the `argiope` executable.
#' @return A list with the parsed contract plus `json_path` and `mask_path`.
argiope_describe <- function(image, outdir = NULL, config = NULL, taxon = NULL,
                             executable = NULL) {
  .argiope_require("jsonlite")
  if (!file.exists(image)) stop("no such image: ", image, call. = FALSE)
  if (is.null(outdir)) outdir <- tempfile("argiope_")
  if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

  stem <- tools::file_path_sans_ext(basename(image))
  json_path <- file.path(outdir, paste0(stem, ".json"))
  mask_path <- file.path(outdir, paste0(stem, "_mask.png"))

  args <- c("describe", shQuote(normalizePath(image, winslash = "/")),
            "--json", shQuote(json_path), "--mask", shQuote(mask_path))
  if (!is.null(config)) args <- c(args, "--config", shQuote(config))
  if (!is.null(taxon))  args <- c(args, "--taxon", shQuote(taxon))

  exe <- argiope_executable(executable)
  out <- suppressWarnings(system2(exe, args, stdout = TRUE, stderr = TRUE))
  status <- attr(out, "status")
  if (!is.null(status) && status != 0) {
    stop("argiope describe failed (exit ", status, "):\n",
         paste(utils::tail(out, 20), collapse = "\n"), call. = FALSE)
  }
  if (!file.exists(json_path)) {
    stop("argiope describe wrote no JSON:\n", paste(utils::tail(out, 20), collapse = "\n"),
         call. = FALSE)
  }

  desc <- jsonlite::fromJSON(json_path, simplifyVector = TRUE)
  desc$json_path <- json_path
  desc$mask_path <- if (file.exists(mask_path)) mask_path else NA_character_
  desc$stdout <- out
  class(desc) <- c("argiope_description", "list")
  desc
}

#' The dominant-colour table as a tidy data.frame.
#'
#' Columns: hex, name, coverage, ci_low, ci_high, delta_e. Zero rows when the pipeline
#' produced no palette (an empty mask), which is a valid outcome and not an error.
argiope_palette <- function(desc) {
  empty <- data.frame(hex = character(), name = character(), coverage = numeric(),
                      ci_low = numeric(), ci_high = numeric(), delta_e = numeric(),
                      stringsAsFactors = FALSE)
  pal <- desc$palette
  if (is.null(pal) || length(pal) == 0L) return(empty)

  if (is.data.frame(pal)) {
    ci <- pal$coverage_ci
    ci_low  <- if (is.matrix(ci)) ci[, 1] else vapply(ci, function(x) as.numeric(x)[1], numeric(1))
    ci_high <- if (is.matrix(ci)) ci[, 2] else vapply(ci, function(x) as.numeric(x)[2], numeric(1))
    return(data.frame(hex = pal$hex, name = pal$name, coverage = pal$coverage,
                      ci_low = ci_low, ci_high = ci_high, delta_e = pal$delta_e,
                      stringsAsFactors = FALSE))
  }
  do.call(rbind, lapply(pal, function(s) data.frame(
    hex = s$hex, name = s$name, coverage = s$coverage,
    ci_low = as.numeric(s$coverage_ci)[1], ci_high = as.numeric(s$coverage_ci)[2],
    delta_e = s$delta_e, stringsAsFactors = FALSE)))
}

#' Opisthosoma summary as a one-row data.frame: score, area fraction, bbox, backend.
argiope_opisthosoma <- function(desc) {
  o <- desc$opisthosoma
  if (is.null(o)) {
    return(data.frame(score = NA_real_, area_frac = NA_real_, x = NA_integer_, y = NA_integer_,
                      w = NA_integer_, h = NA_integer_, backend = NA_character_,
                      stringsAsFactors = FALSE))
  }
  bb <- as.integer(unlist(o$bbox))
  data.frame(score = o$score, area_frac = o$area_frac,
             x = bb[1], y = bb[2], w = bb[3], h = bb[4],
             backend = o$backend, stringsAsFactors = FALSE)
}

#' Read the opisthosoma mask as a logical matrix. Requires the `png` package.
argiope_mask <- function(desc) {
  .argiope_require("png")
  if (is.na(desc$mask_path)) stop("this description carries no mask", call. = FALSE)
  img <- png::readPNG(desc$mask_path)
  if (length(dim(img)) == 3L) img <- img[, , 1]
  img > 0.5
}

#' Describe many images, returning one tidy palette table.
#'
#' Failures are collected rather than raised: a broken image should not abandon the batch.
#' The `error` column is NA for successful rows.
argiope_describe_batch <- function(images, outdir = NULL, config = NULL, executable = NULL,
                                   verbose = TRUE) {
  if (is.null(outdir)) outdir <- tempfile("argiope_batch_")
  dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
  rows <- list()
  for (i in seq_along(images)) {
    img <- images[[i]]
    if (verbose) message(sprintf("Processing image %d of %d: %s", i, length(images),
                                 basename(img)))
    res <- tryCatch({
      d <- argiope_describe(img, outdir = outdir, config = config, executable = executable)
      pal <- argiope_palette(d)
      if (nrow(pal) == 0L) {
        data.frame(image = basename(img), hex = NA_character_, name = NA_character_,
                   coverage = NA_real_, ci_low = NA_real_, ci_high = NA_real_,
                   delta_e = NA_real_, score = argiope_opisthosoma(d)$score,
                   error = NA_character_, stringsAsFactors = FALSE)
      } else {
        cbind(image = basename(img), pal, score = argiope_opisthosoma(d)$score,
              error = NA_character_, stringsAsFactors = FALSE)
      }
    }, error = function(e) data.frame(
      image = basename(img), hex = NA_character_, name = NA_character_, coverage = NA_real_,
      ci_low = NA_real_, ci_high = NA_real_, delta_e = NA_real_, score = NA_real_,
      error = conditionMessage(e), stringsAsFactors = FALSE))
    rows[[length(rows) + 1L]] <- res
  }
  do.call(rbind, rows)
}

print.argiope_description <- function(x, ...) {
  o <- argiope_opisthosoma(x)
  cat("<argiope description>", basename(x$image), "\n")
  cat("  opisthosoma:", if (is.na(o$score)) "none" else
    sprintf("score %.3f, %.2f%% of frame, %s", o$score, 100 * o$area_frac, o$backend), "\n")
  pal <- argiope_palette(x)
  if (nrow(pal)) {
    cat("  palette:\n")
    for (i in seq_len(nrow(pal))) {
      cat(sprintf("    %-8s %-16s %5.1f%%  (dE %.1f)\n",
                  pal$hex[i], pal$name[i], 100 * pal$coverage[i], pal$delta_e[i]))
    }
  }
  invisible(x)
}
