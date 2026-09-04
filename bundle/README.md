<!-- These files live at the ROOT of a hand-over folder, beside adapt_unet.py,
     repro_segmentr.py, argiope_unet.py, R/, checkpoints/ and sample_images/. -->

# Argiope × SegmentR — opisthosoma masks and CIELAB palettes, from R

Point this at a folder of spider photographs and get, for each one, a mask of the
**opisthosoma** (the abdomen), the dominant colours of that region in CIELAB, a paginated
gallery and a four-panel dashboard per specimen.

Everything needed is in this folder, including the trained model. The only thing that cannot
be shipped is a Python interpreter — and `setup.R` will build one for you from R if you have
none.

## Two ways in

**Reading the code:** `argiope_pipeline_segmentR.R` is the whole thing in one file, in
reading order — auxiliary functions, then configuration, then the pipeline in seven numbered
steps. Sourcing it runs everything; set `CFG$autorun <- FALSE` in its configuration section
to load the functions without running. Start here if you want to understand what happens.

**Using it:** `setup.R` then `run_demo.R`, which split the same code across files. Identical
behaviour — verified on the same folder: same masks, same palettes, same counts.

## Quick start

```r
setwd("path/to/this/folder")
source("setup.R")     # installs 3 R packages, then tells you what is still missing
```

`setup.R` ends with a status report. If it says **READY**:

```r
source("run_demo.R")  # runs the six bundled images end to end
```

If it says the Python side is missing, either point the bundle at a Python you already have:

```r
argiope_use_python("C:/path/to/python.exe")   # must have torch + the packages listed below
argiope_status()
```

…or let R build one (downloads roughly 500 MB, CPU-only PyTorch, which is fine here):

```r
argiope_install_python()
```

Then `source("run_demo.R")`.

## Your own images

One argument changes:

```r
source("R/argiope_segmentR.R")
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
| `setup.R` | installs the R side, checks the Python side, can build it |
| `run_demo.R` | the whole workflow on the bundled images |
| `Argiope_SegmentR_R.ipynb` | the same thing as a Jupyter notebook (needs an R kernel) |
| `R/argiope_segmentR.R` | all the R functions |
| `adapt_unet.py`, `repro_segmentr.py`, `argiope_unet.py` | the Python side the R layer calls |
| `checkpoints/opistho_unet.pt` | the trained U-Net, 130 MB |
| `sample_images/` | six photographs, with `ATTRIBUTION.md` |
| `docs/` | a step-by-step tutorial (Spanish) |
| `extras/argiope.R` | **does not work standalone** — see below |

## Requirements

**R** ≥ 4.1 with `jsonlite`, `jpeg`, `png` — `setup.R` installs them.

**Python** ≥ 3.9 with `torch`, `segmentation-models-pytorch`, `numpy`, `opencv-python`,
`scikit-learn`, `scikit-image`, `scipy`, `matplotlib`, `pillow`. A GPU is optional: the model
runs on CPU, just slower (a few seconds per image).

For the notebook only, an R kernel for Jupyter:

```r
install.packages("IRkernel"); IRkernel::installspec(name = "ir", displayname = "R")
```

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
- `extras/argiope.R` is a different entry point that wraps the full `argiope describe` command
  line. It needs the whole Argiope project installed — package, configs, taxonomy files and a
  second checkpoint — so it will **not** run from this folder. It is included for reference.

## Credits and licence

The colour, artefact and QA stages are a port of **SegmentR** — Boyko, J. D. (2025),
*Ecological Informatics* 90:103259, MIT licence. The segmenter and the R layer are from the
Argiope project. See `THIRD_PARTY_NOTICES.md`.

The six sample photographs are third-party GBIF observations, reproduced with attribution
under Creative Commons licences — see `sample_images/ATTRIBUTION.md`. Most are **CC-BY-NC**,
which forbids commercial use.

Colours are reported as HEX, Lab coordinates and coverage. No PANTONE® compatibility is
claimed.
