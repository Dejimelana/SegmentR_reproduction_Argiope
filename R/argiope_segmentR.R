## argiope_segmentR.R — browse the adapted SegmentR pipeline's results from R.
##
## Takes a directory of images, runs the pipeline over it once, and draws the gallery as a
## paginated grid: photograph with the opisthosoma mask outlined, plus the CIELAB palette
## stage E extracted. Items can be picked from a list before plotting.
##
## Wraps, never reimplements. The U-Net stays in torch; R calls `adapt_unet.py` across a
## process boundary (one model load for the whole directory, unlike one `argiope describe`
## per image) and then reads the run's own artefacts: mask PNGs and `colors.csv`. Compositing
## and drawing are done in R.
##
## Requires: jsonlite, jpeg, png. All base graphics beyond that.
##
##   source("R/argiope_segmentR.R")
##   options(argiope.python = ".../envs/argiope/python.exe")
##   g <- argiope_gallery("data/raw/gbif/argiope_bruennichi", n = 40)
##   argiope_plot(g, page = 1)                 # first page of the grid
##   argiope_plot(g, select = argiope_pick(g)) # choose items from a list, then plot
##   argiope_pdf(g, "galeria.pdf")             # every page into one PDF
##   argiope_dashboard(g)                      # pick one image -> 4-panel dashboard
##   argiope_dashboard(g, "0cb8b75b7c53.jpg", file = "ficha.png")
##
## Non-interactive:  Rscript R/argiope_segmentR.R <dir_de_imagenes> [salida.pdf] [n]

.ARGIOPE_ACCENT <- c(232, 198, 74) / 255   # the gallery's mask outline
.ARGIOPE_INK    <- "#191C11"
.ARGIOPE_MUTED  <- "#7C8270"

.need <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(sprintf('package "%s" is required: install.packages("%s")', pkg, pkg), call. = FALSE)
  }
}

# Resolved WHEN THIS FILE IS LOADED, not when a function is later called: the source()
# frame that carries $ofile is gone by then, and the --file= fallback would point at
# whichever outer script did the sourcing.
.ARGIOPE_SCRIPT <- local({
  for (i in seq_len(sys.nframe())) {                 # source()d
    ofile <- sys.frame(i)$ofile
    if (!is.null(ofile)) return(normalizePath(ofile, winslash = "/", mustWork = FALSE))
  }
  ca <- commandArgs(trailingOnly = FALSE)            # Rscript argiope_segmentR.R
  m <- grep("^--file=", ca, value = TRUE)
  if (length(m)) {
    return(normalizePath(sub("^--file=", "", m[1]), winslash = "/", mustWork = FALSE))
  }
  NA_character_
})

.repo_root <- function() {
  # .../repro/segmentr/R/argiope_segmentR.R -> .../repro/segmentr
  this <- .ARGIOPE_SCRIPT
  if (is.na(this) || !nzchar(this)) return(normalizePath(".", winslash = "/"))
  dirname(dirname(this))
}

#' Locate the Python interpreter of the `argiope` environment.
#'
#' Resolution order: argument, `options(argiope.python=)`, `ARGIOPE_PYTHON`, the interpreter
#' sitting next to the `argiope` executable, then `python` on PATH.
argiope_python <- function(python = NULL) {
  if (!is.null(python) && nzchar(python)) return(python)
  opt <- getOption("argiope.python")
  if (!is.null(opt) && nzchar(opt)) return(opt)
  env <- Sys.getenv("ARGIOPE_PYTHON", unset = "")
  if (nzchar(env)) return(env)
  exe <- getOption("argiope.executable", default = unname(Sys.which("argiope")))
  if (!is.null(exe) && nzchar(exe)) {
    cand <- file.path(dirname(dirname(exe)), "python.exe")          # env/Scripts/.. -> env
    if (file.exists(cand)) return(cand)
    cand <- file.path(dirname(exe), "python")
    if (file.exists(cand)) return(cand)
  }
  found <- unname(Sys.which("python"))
  if (nzchar(found)) return(found)
  stop("Could not find the argiope environment's Python.\n",
       '  options(argiope.python = ".../envs/argiope/python.exe")', call. = FALSE)
}

#' Run the adapted pipeline over a directory and load the results.
#'
#' @param dir Directory of images (searched recursively).
#' @param out Where the run directory is written. Defaults to a temp directory.
#' @param run_id Name of the run directory.
#' @param n Optional cap on the number of images (seeded sample).
#' @param seed Sampling seed.
#' @param n_colors Clusters per mask.
#' @param adapter Path to `adapt_unet.py`; found next to this script by default.
#' @param reuse If TRUE and the run directory already holds results, skip the run.
#' @return A gallery object; see [argiope_items()].
argiope_gallery <- function(dir, out = NULL, run_id = "r-gallery", n = NULL, seed = 42,
                            n_colors = 5, python = NULL, adapter = NULL, reuse = TRUE,
                            quiet = FALSE) {
  .need("jsonlite")
  if (!dir.exists(dir) && !file.exists(dir)) stop("no such directory: ", dir, call. = FALSE)
  if (is.null(out)) out <- file.path(tempdir(), "argiope_gallery")
  if (is.null(adapter)) adapter <- file.path(.repo_root(), "adapt_unet.py")
  if (!file.exists(adapter)) stop("adapt_unet.py not found: ", adapter, call. = FALSE)

  run_dir <- file.path(out, run_id)
  done <- file.exists(file.path(run_dir, "summary.json"))
  if (!(reuse && done)) {
    args <- c(shQuote(normalizePath(adapter, winslash = "/")),
              "--images", shQuote(normalizePath(dir, winslash = "/")),
              "--out", shQuote(out), "--run-id", shQuote(run_id),
              "--n-colors", n_colors, "--seed", seed, "--no-qa")
    if (!is.null(n)) args <- c(args, "--n", n)
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

#' Load a finished run directory as a gallery object.
argiope_load_gallery <- function(run_dir) {
  .need("jsonlite")
  cfg_path <- file.path(run_dir, "run_config.json")
  if (!file.exists(cfg_path)) stop("not a run directory: ", run_dir, call. = FALSE)
  cfg <- jsonlite::fromJSON(cfg_path, simplifyVector = TRUE)

  colors <- if (file.exists(file.path(run_dir, "colors.csv")))
    utils::read.csv(file.path(run_dir, "colors.csv"), stringsAsFactors = FALSE) else
      data.frame()
  skipped <- if (file.exists(file.path(run_dir, "skipped.csv")))
    utils::read.csv(file.path(run_dir, "skipped.csv"), stringsAsFactors = FALSE) else
      data.frame()

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

#' The per-image table: image, group, mask presence, score, mask pixels, skip reason.
argiope_items <- function(g) g$items

#' The palette of one image, as a data.frame ordered by coverage.
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
        " · pages of 6: ", argiope_pages(x), "\n", sep = "")
  }
  invisible(x)
}

#' Pick images from a list.
#'
#' Uses `utils::select.list`, which opens a real multi-selection list in an interactive
#' session. Non-interactively it returns every item, so scripts keep working.
#'
#' @return A character vector of image names, suitable for `argiope_plot(select = )`.
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

# ---------------------------------------------------------------- image helpers

.read_image <- function(path) {
  ext <- tolower(tools::file_ext(path))
  if (ext %in% c("jpg", "jpeg")) { .need("jpeg"); jpeg::readJPEG(path) }
  else { .need("png"); png::readPNG(path) }
}

.as_gray <- function(a) if (length(dim(a)) == 3L) a[, , 1] else a

.decimate_k <- function(dims, maxdim) max(1L, as.integer(ceiling(max(dims) / maxdim)))

.decimate <- function(a, k) {
  if (k <= 1L) return(a)
  d <- dim(a)
  ri <- seq(1L, d[1], by = k); ci <- seq(1L, d[2], by = k)
  if (length(d) == 3L) a[ri, ci, , drop = FALSE] else a[ri, ci, drop = FALSE]
}

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

#' Photograph with the mask outlined and the background pushed back.
.compose <- function(rgb, mask, dim_mul = 0.55, dim_add = 0.10, ring = 2L) {
  if (length(dim(rgb)) == 2L) rgb <- array(rgb, c(dim(rgb), 3))
  out <- rgb[, , 1:3, drop = FALSE]
  outside <- !mask
  for (ch in 1:3) {
    v <- out[, , ch]; v[outside] <- v[outside] * dim_mul + dim_add; out[, , ch] <- v
  }
  edge <- .dilate(mask, ring) & !mask          # drawn outside, so no mask pixel is hidden
  for (ch in 1:3) { v <- out[, , ch]; v[edge] <- .ARGIOPE_ACCENT[ch]; out[, , ch] <- v }
  out
}

.load_composite <- function(item, maxdim = 520) {
  rgb <- .read_image(item$path)
  k <- .decimate_k(dim(rgb)[1:2], maxdim)
  frame_px <- prod(dim(rgb)[1:2])
  rgb <- .decimate(rgb, k)
  if (!isTRUE(item$has_mask)) return(list(img = rgb, frac = NA_real_, empty = TRUE))
  mask <- .as_gray(.read_image(item$mask)) > 0.5
  if (!all(dim(mask) == dim(rgb)[1:2])) mask <- .decimate(mask, k)
  if (!all(dim(mask) == dim(rgb)[1:2])) {
    # last resort: the mask and the photo disagree; report rather than draw a lie
    return(list(img = rgb, frac = NA_real_, empty = TRUE, note = "mask/image size mismatch"))
  }
  list(img = .compose(rgb, mask), frac = sum(mask) * (k^2) / frame_px, empty = FALSE)
}

# ---------------------------------------------------------------- drawing

.draw_cell <- function(g, item, maxdim) {
  graphics::plot.new()
  graphics::plot.window(c(0, 1), c(0, 1))
  comp <- tryCatch(.load_composite(item, maxdim),
                   error = function(e) list(img = NULL, empty = TRUE, note = conditionMessage(e)))

  band <- c(0.23, 0.925)                                  # vertical room for the photograph
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
                 cex = 0.78, font = 3, col = .ARGIOPE_INK)
  graphics::text(1, 0.98, if (is.na(item$score)) "sin máscara" else
                 sprintf("score %.3f", item$score), adj = c(1, 1), cex = 0.7,
                 col = if (is.na(item$score)) "#8C5A3C" else "#8A6E12")
  graphics::text(0, 0.935, item$image, adj = c(0, 1), cex = 0.62, col = .ARGIOPE_MUTED,
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
    lab <- paste(sprintf("%s %.0f%%", pal$hex[seq_len(min(3, nrow(pal)))],
                         100 * pal$coverage[seq_len(min(3, nrow(pal)))]), collapse = "   ")
    graphics::text(0.02, 0.075, lab, adj = c(0, 1), cex = 0.6, col = .ARGIOPE_MUTED,
                   family = "mono")
  }
  if (!is.na(comp$frac)) {
    graphics::text(0.98, 0.075, sprintf("%s px · %.2f%%",
                   formatC(item$px, format = "d", big.mark = ".", decimal.mark = ","), 100 * comp$frac),
                   adj = c(1, 1), cex = 0.6, col = .ARGIOPE_MUTED, family = "mono")
  }
  invisible(NULL)
}

.selected <- function(g, select, include_empty) {
  it <- g$items
  if (!include_empty) it <- it[it$has_mask, ]
  if (!is.null(select)) {
    it <- if (is.numeric(select)) it[select, , drop = FALSE] else
      it[it$image %in% select, , drop = FALSE]
  }
  it[stats::complete.cases(it$image), , drop = FALSE]
}

#' How many pages the gallery needs.
argiope_pages <- function(g, per_page = 6, select = NULL, include_empty = FALSE) {
  n <- nrow(.selected(g, select, include_empty))
  max(1L, as.integer(ceiling(n / per_page)))
}

#' Draw one page of the gallery as a grid.
#'
#' @param page Which page (1-based).
#' @param per_page Cells per page; the grid is laid out to fit them.
#' @param select Image names or indices to restrict to — e.g. from [argiope_pick()].
#' @param include_empty Show images the segmenter returned nothing for, labelled.
argiope_plot <- function(g, page = 1, per_page = 6, select = NULL, include_empty = FALSE,
                         ncol = NULL, maxdim = 520) {
  it <- .selected(g, select, include_empty)
  n <- nrow(it)
  if (!n) { message("Nothing to plot."); return(invisible(NULL)) }
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
  for (i in seq_len(nrow_ * ncol - nrow(it))) { graphics::plot.new() }

  graphics::mtext("Argiope · opisthosoma + pantone", side = 3, outer = TRUE, line = 0.7,
                  adj = 0, cex = 0.95, font = 2, col = .ARGIOPE_INK)
  graphics::mtext(sprintf("página %d de %d · %d de %d imágenes", page, pages, nrow(it), n),
                  side = 3, outer = TRUE, line = 0.7, adj = 1, cex = 0.7, col = .ARGIOPE_MUTED)
  graphics::mtext(basename(g$run_dir), side = 1, outer = TRUE, line = 0.4, adj = 1,
                  cex = 0.6, col = .ARGIOPE_MUTED)
  invisible(page)
}

#' Write every page to a PDF, one page per sheet.
argiope_pdf <- function(g, file = "argiope_galeria.pdf", per_page = 6, select = NULL,
                        include_empty = FALSE, width = 11, height = 8, maxdim = 520) {
  pages <- argiope_pages(g, per_page, select, include_empty)
  grDevices::pdf(file, width = width, height = height)
  on.exit(grDevices::dev.off(), add = TRUE)
  for (p in seq_len(pages)) {
    argiope_plot(g, page = p, per_page = per_page, select = select,
                 include_empty = include_empty, maxdim = maxdim)
  }
  message("Wrote ", pages, " page(s) -> ", normalizePath(file, winslash = "/", mustWork = FALSE))
  invisible(file)
}


# ================================================================ single-image dashboard
# The four-panel QA figure, rebuilt in R from the run's artefacts: the photograph with the
# detection box and mask contour, the dominant colours, the RGB histogram of the masked
# pixels, and the mask recoloured by cluster centroid.
#
# The recolouring reproduces stage E's assignment rule rather than approximating it: masked
# pixels are converted to CIELAB with grDevices::convertColor (base R, the same call the
# original SegmentR used) and assigned to the nearest centroid by Euclidean distance in Lab.

#' The detection stored beside the mask, as a one-row list (label, score, box).
.read_detection <- function(g, item) {
  stem <- tools::file_path_sans_ext(item$image)
  jp <- file.path(g$run_dir, "json", paste0(item$group, "__", stem, ".json"))
  if (!file.exists(jp)) return(NULL)
  recs <- jsonlite::fromJSON(jp, simplifyVector = FALSE)
  if (!length(recs)) return(NULL)
  d <- recs[[1]]
  list(label = d$label, score = d$score, box = unlist(d$box)[c("xmin", "ymin", "xmax", "ymax")])
}

#' Assign each masked pixel to its nearest palette centroid, in Lab.
.assign_clusters <- function(rgb_px, centers_lab) {
  lab <- grDevices::convertColor(rgb_px, from = "sRGB", to = "Lab")
  d2 <- vapply(seq_len(nrow(centers_lab)),
               function(i) colSums((t(lab) - centers_lab[i, ])^2),
               numeric(nrow(lab)))
  if (is.null(dim(d2))) d2 <- matrix(d2, nrow = nrow(lab))
  max.col(-d2, ties.method = "first")
}

#' Pick exactly one image from a list.
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

#' Four-panel dashboard for one image.
#'
#' @param g A gallery from [argiope_gallery()] or [argiope_load_gallery()].
#' @param image Image file name. Omit to choose from a list.
#' @param file Optional PNG path; when given the figure is written there instead of the
#'   current device.
#' @param maxdim Longest side used for the displayed rasters. Statistics (histogram,
#'   cluster sizes) are always computed at full resolution.
#' @return Invisibly, a list with the palette, the mask pixel count and the mean/median
#'   colours — so the panel's numbers are available to the caller, not just drawn.
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

  rgb_full <- .read_image(item$path)
  if (length(dim(rgb_full)) == 2L) rgb_full <- array(rgb_full, c(dim(rgb_full), 3))
  rgb_full <- rgb_full[, , 1:3, drop = FALSE]
  mask_full <- .as_gray(.read_image(item$mask)) > 0.5
  if (!all(dim(mask_full) == dim(rgb_full)[1:2])) {
    stop("mask and image dimensions disagree for ", image, call. = FALSE)
  }

  pal <- argiope_palette_of(g, image)
  centers_lab <- as.matrix(pal[, c("lab_l", "lab_a", "lab_b")])
  centers_rgb <- pmin(pmax(grDevices::convertColor(centers_lab, from = "Lab", to = "sRGB"), 0), 1)

  px <- cbind(rgb_full[, , 1][mask_full], rgb_full[, , 2][mask_full], rgb_full[, , 3][mask_full])
  n_px <- nrow(px)
  frac <- n_px / prod(dim(mask_full))
  assign_ <- .assign_clusters(px, centers_lab)

  det <- .read_detection(g, item)
  row0 <- g$colors[g$colors$image == image, ][1, ]
  mean_hex <- row0$mean_color
  median_hex <- row0$median_color

  if (!is.null(file)) {
    grDevices::png(file, width = width, height = height, res = res)
    on.exit(grDevices::dev.off(), add = TRUE)
  }
  op <- graphics::par(mfrow = c(2, 2), mar = c(3.2, 3.2, 2.4, 1.0),
                      oma = c(0.4, 0.4, 2.2, 0.4), bg = "#FFFFFF")
  on.exit(graphics::par(op), add = TRUE)

  # ---- panel 1: photograph, detection box, mask contour -------------------------
  k <- .decimate_k(dim(rgb_full)[1:2], maxdim)
  disp <- .decimate(rgb_full, k)
  dmask <- .decimate(mask_full, k)
  edge <- .dilate(dmask, 2L) & !dmask
  for (ch in 1:3) { v <- disp[, , ch]; v[edge] <- .ARGIOPE_ACCENT[ch]; disp[, , ch] <- v }
  h <- dim(disp)[1]; w <- dim(disp)[2]
  graphics::plot.new(); graphics::plot.window(c(0, w), c(h, 0), asp = 1)
  graphics::rasterImage(disp, 0, h, w, 0, interpolate = TRUE)
  if (!is.null(det)) {
    b <- det$box / k
    graphics::rect(b[["xmin"]], b[["ymin"]], b[["xmax"]], b[["ymax"]],
                   border = "#3B4CC0", lwd = 1.6)
    graphics::text(b[["xmin"]], b[["ymin"]] - 4,
                   sprintf("%s: %.2f", det$label, det$score),
                   adj = c(0, 1), cex = 0.62, col = "#3B4CC0")
  }
  graphics::title("detections + mask contours", cex.main = 0.95, font.main = 1, line = 0.6)

  # ---- panel 2: dominant colours ------------------------------------------------
  graphics::plot.new(); graphics::plot.window(c(0, nrow(pal)), c(-0.42, 1))
  for (i in seq_len(nrow(pal))) {
    graphics::rect(i - 1, 0, i, 1, col = pal$hex[i], border = NA)
    graphics::text(i - 0.5, -0.05,
                   sprintf("%s\n%.1f%%\nL%.0f a%.0f b%.0f", pal$hex[i], 100 * pal$coverage[i],
                           pal$lab_l[i], pal$lab_a[i], pal$lab_b[i]),
                   adj = c(0.5, 1), cex = 0.6, family = "mono", col = .ARGIOPE_INK)
  }
  graphics::title(sprintf("dominant colours (k-means in CIELAB)   mean %s  median %s",
                          mean_hex, median_hex), cex.main = 0.85, font.main = 1, line = 0.6)

  # ---- panel 3: RGB histogram of the masked pixels ------------------------------
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

  # ---- panel 4: mask recoloured by cluster centroid -----------------------------
  rec <- array(1, dim(rgb_full))
  for (ch in 1:3) { v <- rec[, , ch]; v[mask_full] <- centers_rgb[assign_, ch]; rec[, , ch] <- v }
  rec <- .decimate(rec, k)
  graphics::plot.new(); graphics::plot.window(c(0, w), c(h, 0), asp = 1)
  graphics::rasterImage(rec, 0, h, w, 0, interpolate = FALSE)
  graphics::title("mask recoloured by cluster centroid", cex.main = 0.95, font.main = 1,
                  line = 0.6)

  graphics::mtext(sprintf("%s / %s    unet:%s    mask=%.2f%% of frame",
                          item$group, item$image, basename(g$config$weights), 100 * frac),
                  side = 3, outer = TRUE, line = 0.4, cex = 0.85, col = .ARGIOPE_INK)

  invisible(list(image = image, palette = pal, mask_px = n_px, mask_frac = frac,
                 mean_color = mean_hex, median_color = median_hex,
                 cluster_sizes = tabulate(assign_, nbins = nrow(pal))))
}

# ---------------------------------------------------------------- script entry point

if (!interactive() && sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  if (!length(args)) {
    cat("Usage: Rscript argiope_segmentR.R <image_dir> [output.pdf] [n_images]\n")
  } else {
    dir_in <- args[1]
    out_pdf <- if (length(args) >= 2) args[2] else "argiope_galeria.pdf"
    n_max <- if (length(args) >= 3) as.integer(args[3]) else NULL
    g <- argiope_gallery(dir_in, n = n_max)
    print(g)
    argiope_pdf(g, out_pdf)
  }
}
