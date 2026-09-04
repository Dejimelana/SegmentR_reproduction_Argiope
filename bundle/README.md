<!-- Lives at the ROOT of a hand-over folder. NOT the repository copy. -->

# Argiope × SegmentR — opisthosoma masks and CIELAB palettes, from R

Point this at a folder of spider photographs and get, for each one, a mask of the
**opisthosoma** (the abdomen), the dominant colours of that region in CIELAB, a paginated
gallery and a four-panel dashboard per specimen.

Everything needed is in this folder, including the trained model. The only thing that cannot
be shipped is a Python interpreter — and `setup.R` will build one for you from R if you have
none.

## Three ways in

**As an installed R package:** `package/argiopeSegmentR_0.1.0.tar.gz` installs with one line
and carries everything, weights included. Work through `package/example_argiopeSegmentR.R`.
See `package/README.md`.

**As a cheatsheet:** `cheatsheet_argiopeSegmentR.html` — two printable pages with the usage,
the plotting functions, the measured performance of every segmenter tried, and the three
failure modes.

## Two ways in

**Reading the code, and stepping through it:** `argiope_pipeline_segmentR.R` is the whole
thing in one file, in reading order — auxiliary functions, then configuration, then the
functions themselves, then a walkthrough. Sections 1-3 only *define* things; section 4 is a
numbered walkthrough you run **one step at a time** (put the cursor on a line, Ctrl+Enter),
each printing something to look at before you move on. Sourcing the file runs the whole
walkthrough instead. Start here if you want to understand what happens.

**Checking the machine first:** `setup.R` installs the R packages and reports whether the
Python side is ready, without running anything. Sourcing the pipeline file runs the
walkthrough, so this is the way to look before you leap.

## Quick start

```r
setwd("path/to/this/folder")
source("setup.R")     # installs 3 R packages, then tells you what is still missing
```

`setup.R` ends with a status report. If it says **READY**:

```r
source("argiope_pipeline_segmentR.R")   # runs the walkthrough end to end
```

Or open that file and run section 4 one step at a time, which is the intended way.

If it says the Python side is missing, either point the bundle at a Python you already have:

```r
argiope_use_python("C:/path/to/python.exe")   # must have torch + the packages listed below
argiope_status()
```

…or let R build one (downloads roughly 500 MB, CPU-only PyTorch, which is fine here):

```r
argiope_install_python()
```

Then `source("argiope_pipeline_segmentR.R")`.

## The three bundled image folders

| Folder | What it is | Why it is here |
|---|---|---|
| `sample_images/` | 6 photographs | the quickest thing to run |
| `crops/` | 40 crops cut for hand annotation | pre-cropped spiders: the segmenter's easy case |
| `random_ArTaxOr+GBIF/` | 20 ArTaxOr + 20 GBIF, drawn at random | uncropped field photographs: the honest case |

None of the 40 in `random_ArTaxOr+GBIF` is among the 150 images used to train the
segmenter, so what you see there is genuine unseen behaviour. Expect a good fraction to
come back with no mask at all — on unseen field photographs that happens on roughly one
image in five, and it is a known limit rather than a fault in this code.

```r
g <- argiope_gallery("crops",               out = "runs", run_id = "crops")
g <- argiope_gallery("random_ArTaxOr+GBIF", out = "runs", run_id = "random")
```

## Your own images

One argument changes:

```r
source("argiope_pipeline_segmentR.R")
g <- argiope_gallery("C:/path/to/your/images", out = "runs", run_id = "mine")
```

Any folder of `.jpg` / `.jpeg` / `.png`, searched recursively. Add `n = 50` to take a seeded
random sample instead of everything.

## What you get

```r
g                                  # 40 images · 31 with a mask · pages of 6: 6
argiope_items(g)                   # one row per image: mask? score? size? why not?
argiope_palette_of(g, "x.jpg")     # HEX + Lab + coverage per colour cluster
argiope_plot(g, page = 2)          # the gallery grid, one page
argiope_plot(g, select = argiope_pick(g))   # choose from a list first
argiope_dashboard(g, "x.jpg")      # photo + palette + histogram + recoloured mask
argiope_pdf(g, "gallery.pdf")      # every page into one PDF
```

Images the model finds nothing in are **never dropped silently** — they are listed with a
reason, in `argiope_items(g)` and in `runs/<id>/skipped.csv`.

## What is in this folder

| | |
|---|---|
| `argiope_pipeline_segmentR.R` | **the whole workflow in one file — start here** |
| `setup.R` | installs the R side and checks the Python side, without running anything |
| `adapt_unet.py`, `repro_segmentr.py`, `argiope_unet.py` | the Python side the R layer calls |
| `checkpoints/opistho_unet.pt` | the trained U-Net, 130 MB |
| `sample_images/` | 9 photographs, with `ATTRIBUTION.md` |
| `crops/` | 40 crops made for hand annotation, with `PROVENANCE.md` — the easy case |
| `random_ArTaxOr+GBIF/` | 40 spiders drawn at random from ArTaxOr and GBIF, with `PROVENANCE.md` |

## Requirements

**R** ≥ 4.1 with `jsonlite`, `jpeg`, `png` — `setup.R` installs them.

**Python** ≥ 3.9 with `torch`, `segmentation-models-pytorch`, `numpy`, `opencv-python`,
`scikit-learn`, `scikit-image`, `scipy`, `matplotlib`, `pillow`. A GPU is optional: the model
runs on CPU, just slower (a few seconds per image).

## How it fits together

The segmenter is a U-Net trained in PyTorch. R does not reimplement it — it calls
`adapt_unet.py` once for the whole folder (one model load, not one per image), then reads the
run's artefacts and does the compositing and drawing itself.

```
R  ──►  adapt_unet.py  ──►  U-Net (torch)  ──►  mask
                       └──►  CIELAB palette, JSON artefact, cut-outs
R  ◄──  runs/<id>/  (masks, colors.csv, per-image JSON)  ──►  grid, dashboard, PDF
```

`runs/<id>/run_config.json` records every parameter, so a run is reproducible from it alone.
Colour can be recomputed from the artefacts without loading the model:

```bash
python adapt_unet.py --from-json runs/demo
```

## Troubleshooting

**`Could not find the argiope environment's Python`** — run `source("setup.R")` and then either
`argiope_use_python(...)` or `argiope_install_python()`.

**`missing checkpoint`** — `checkpoints/opistho_unet.pt` is not where it should be. It ships
with this folder; if your copy is missing it, ask for it (130 MB, too large for email).

**Everything comes back "no mask"** — check `argiope_status()` first. If the setup is fine,
this is the model: on unseen field photographs it returns nothing on roughly one image in
five, which is a known limit, not a bug in this code.

**`package "jpeg" is required`** — `install.packages("jpeg")`, likewise `png`, `jsonlite`.

**The PDF is enormous** — rasters are embedded uncompressed. Use
`argiope_pdf(g, "x.pdf", maxdim = 360)`.

## Honest limits

- A mask is **not** automatically correct. On a 100-image unseen sample, 81 produced a mask and
  3 of those were fragments under 0.1% of the frame — all three scoring below 0.70 against a
  median of 0.959, so a score filter separates them. Look at the pictures.
- Measured against hand-drawn masks on a held-out split, the median IoU is 0.765, and the
  colour taken from the predicted mask is indistinguishable from the colour taken from the
  hand-drawn one (ΔE 1.51) wherever IoU ≥ 0.7.

## Credits and licence

The colour, artefact and QA stages are a port of **SegmentR** — Boyko, J. D. (2025),
*Ecological Informatics* 90:103259, MIT licence. The segmenter and the R layer are from the
Argiope project. See `THIRD_PARTY_NOTICES.md`.

The six sample photographs are third-party GBIF observations, reproduced with attribution
under Creative Commons licences. Every image folder carries its own attribution file; the
totals are in `THIRD_PARTY_NOTICES.md`. Most are **CC-BY-NC**,
which forbids commercial use.

Colours are reported as HEX, Lab coordinates and coverage. No PANTONE® compatibility is
claimed.
