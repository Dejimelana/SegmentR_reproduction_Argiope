## =============================================================================
##  example_argiopeSegmentR.R — the worked example, one step at a time
## =============================================================================
##
##  Install once, from the tarball:
##
##      install.packages("argiopeSegmentR_0.1.0.tar.gz", repos = NULL, type = "source")
##
##  Then work through this file. Put the cursor on a line (or select a block) and
##  press Ctrl+Enter: every step prints something you should look at before moving
##  on. Sourcing the file runs the lot.
##
##  Steps 1 and 2 must run in order. After that, 4 to 8 only read `g`, so re-run
##  them in any order, as often as you like.
## =============================================================================

library(argiopeSegmentR)

## Just re-installed the package? Restart R before this line. A session keeps the
## version it already loaded and library() will not replace it, which surfaces later
## as "could not find function".


## ---- STEP 0 · is the machine ready? -----------------------------------------
## The segmenter is a U-Net that runs in Python; the package ships the weights but
## not an interpreter. Look for READY.
##
## If the Python side is missing, either point the package at one you already have
## or let it build one (about 500 MB, CPU-only PyTorch, which is plenty here):
##
##      argiope_use_python("C:/path/to/python.exe")
##      argiope_install_python()

install_r_packages()
argiope_status()


## ---- STEP 1 · segment a folder of photographs -------------------------------
## The only slow step: it loads the model once and walks the whole folder. A
## matching finished run is reused, so running this again is cheap.
## Look at: how many images came back with a mask.

images <- system.file("sample_images", package = "argiopeSegmentR")   # or your own folder
outdir <- file.path(tempdir(), "argiope")                             # or "runs"

g <- argiope_gallery(dir = images, out = outdir, run_id = "example", n_colors = 5)

g                                     # the summary line


## ---- STEP 2 · the table, before any picture ---------------------------------
## One row per image. `score` is the segmenter's confidence, `px` the mask size.

it <- argiope_items(g)
it[, c("image", "group", "has_mask", "score", "px")]


## ---- STEP 3 · what failed, and why ------------------------------------------
## Images with no mask are never dropped silently. On unseen field photographs
## expect roughly one in five to come back empty: that is the model, not a bug.
## A reason mentioning "No module named" is an environment problem instead.

sum(it$has_mask)
subset(it, !has_mask)[, c("image", "reason")]


## ---- STEP 4 · the palette of one specimen -----------------------------------
## `coverage` is the share of the mask's pixels in that cluster; lab_* are the
## CIELAB coordinates of its centroid. HEX + Lab + coverage is the "pantone".
## Change `first` to any file name from step 2.

first <- it$image[it$has_mask][1]
first
argiope_palette_of(g, first)


## ---- STEP 5 · the gallery, page by page -------------------------------------
## Each cell: the photograph with the mask outlined, the palette bar proportional
## to coverage, the leading HEX values, the score and the mask area.
## Re-run the last line with page = 2, 3, ... to walk through.

argiope_pages(g)                      # how many pages of 6

argiope_plot(g, page = 1)


## ---- STEP 6 · draw only the specimens you choose ----------------------------
## Interactively argiope_pick() opens a real selection list; in a script it
## returns everything. You can also select by name or by position.

sel <- argiope_pick(g)
## sel <- it$image[it$has_mask][1:3]        # ... or by name
## sel <- 1:3                               # ... or by position

argiope_plot(g, select = sel, per_page = 3)


## ---- STEP 7 · the four-panel dashboard for one specimen ---------------------
## Photograph with box and contour · dominant colours · RGB histogram of the
## masked pixels · the mask recoloured by centroid. It also RETURNS the numbers.

res <- argiope_dashboard(g, first)

res$mask_px
res$mean_color
res$cluster_sizes

## argiope_dashboard(g)                                  # ... or pick from a list
## argiope_dashboard(g, first, file = "specimen.png")    # ... or write a PNG


## ---- STEP 7b · the specimen card, and a page of cards -----------------------
## A denser alternative to the grid: one card per specimen, with the cut-out and the
## palette listed by nearest colour name. argiope_card_grid() lays several on a page,
## calling argiope_card() per panel so the two cannot diverge.

argiope_card(g, first)
## argiope_card(g, first, file = "card.png")        # ... or to a PNG

argiope_card_grid(g, page = 1, per_page = 6, ncol = 3)
## argiope_card_grid(g, select = sel, file = "cards.png")


## ---- STEP 8 · export every page to one PDF ----------------------------------
## Rasters go in uncompressed, so lower maxdim for a large gallery.

argiope_pdf(g, file.path(outdir, "gallery.pdf"))


## ---- WHERE IT LANDED --------------------------------------------------------
## run_config.json records every parameter: a run is reproducible from it alone,
## and colour can be recomputed from the artefacts without loading the model:
##
##      python <inst>/adapt_unet.py --from-json <run_dir>

g$run_dir
head(list.files(g$run_dir, recursive = TRUE), 12)


## ---- YOUR OWN IMAGES --------------------------------------------------------
## Any folder of .jpg/.jpeg/.png, searched recursively. `n` takes a seeded sample.
##
##   g <- argiope_gallery("C:/path/to/images", out = "runs", run_id = "mine", n = 50)
