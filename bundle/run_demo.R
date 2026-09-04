## Argiope x SegmentR — the whole workflow on the bundled sample images.
##
##   source("run_demo.R")
##
## Run setup.R first if you have not. This writes runs/demo/ and demo_gallery.pdf next to
## this file, and draws two figures.

## Refuse to start against a half-built environment: the symptom would be every image
## skipped with a Python import error, which reads as a segmentation failure and is not.
if (!isTRUE(source("setup.R")$value)) {   # setup.R ends with argiope_status()
  stop("Setup is incomplete - see the report above. Fix that, then re-run this file.",
       call. = FALSE)
}

source("R/argiope_segmentR.R")

cat("Python:", argiope_python(), "\n\n")

## 1. process a folder of images (one model load for the whole folder)
g <- argiope_gallery(dir = "sample_images", out = "runs", run_id = "demo")
print(g)

## 2. the table, before the pictures
it <- argiope_items(g)
print(it[, c("image", "group", "has_mask", "score", "px")])
cat("\nwith a mask:", sum(it$has_mask), "of", nrow(it), "\n")
if (any(!it$has_mask)) print(subset(it, !has_mask)[, c("image", "reason")])

## 3. the palette of the first specimen
first <- it$image[it$has_mask][1]
cat("\npalette of", first, "\n")
print(argiope_palette_of(g, first))

## 4. the gallery as a paginated grid
cat("\npages of 6:", argiope_pages(g), "\n")
argiope_plot(g, page = 1)

## 5. the four-panel dashboard for one specimen
res <- argiope_dashboard(g, first)
cat("\nmask:", res$mask_px, "px  mean:", res$mean_color, " median:", res$median_color, "\n")

## 6. every page into one PDF
argiope_pdf(g, "demo_gallery.pdf")

cat("\nDone. Artefacts in runs/demo/ and demo_gallery.pdf\n")
cat("Point it at your own images with:\n")
cat('  g <- argiope_gallery("C:/path/to/images", out = "runs", run_id = "mine")\n')
