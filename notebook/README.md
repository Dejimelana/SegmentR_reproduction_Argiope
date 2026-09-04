# Executable R notebook

`Argiope_SegmentR_R.ipynb` — the whole workflow in R, from a folder of images to masks,
CIELAB palettes, a paginated gallery and a per-specimen dashboard. It ships **already
executed**, so the figures are visible on GitHub without running anything.

```
notebook/
├── Argiope_SegmentR_R.ipynb   the notebook (R kernel), with its output
├── install_R_packages.R       the three CRAN packages it needs
├── sample_images/             six attributed photographs, so it runs out of the box
│   └── ATTRIBUTION.md         photographer, licence and GBIF link for each
└── runs/                      written when you execute it (git-ignored)
```

## What you must supply

Two things cannot live in a repository:

| | Why | How |
|---|---|---|
| the `argiope` conda environment | runs the U-Net (torch + CUDA) | `conda env create -f environment.yml` in the parent project |
| `checkpoints/opistho_unet.pt` | the trained weights, ~130 MB | `argiope train-segmenter`, or copy it in |

Everything else — the adapter, the R layer, the images — is here or one directory up.

## Setup

```r
# 1. the R packages the workflow uses
Rscript install_R_packages.R          # jsonlite, jpeg, png

# 2. an R kernel for Jupyter, so the notebook can run R cells
Rscript -e 'install.packages("IRkernel"); IRkernel::installspec(name="ir", displayname="R")'
```

Then tell R where the environment's Python is — either launch Jupyter from the activated
conda environment, or set the variable the notebook reads:

```bash
set ARGIOPE_PYTHON=C:\path\to\envs\argiope\python.exe    # Windows
export ARGIOPE_PYTHON=/path/to/envs/argiope/bin/python   # macOS / Linux
jupyter lab Argiope_SegmentR_R.ipynb
```

## Running it headless

```bash
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=ir Argiope_SegmentR_R.ipynb
```

## Pointing it at your own images

Change one argument in section 3:

```r
g <- argiope_gallery(dir = "/path/to/your/images", out = "runs", run_id = "mine", n = 50)
```

Any folder of `.jpg`/`.jpeg`/`.png`, searched recursively. `n` takes a seeded random sample.

## The prose version

`../readme_Argiope_SegmentR.md` is the same workflow as a step-by-step tutorial, in Spanish,
including a troubleshooting section.

## Sample images

The six photographs in `sample_images/` are third-party GBIF observations, downscaled to
1024 px and reproduced **with attribution** under their Creative Commons licences — see
`sample_images/ATTRIBUTION.md`. Most are `CC-BY-NC`, which forbids commercial use.
