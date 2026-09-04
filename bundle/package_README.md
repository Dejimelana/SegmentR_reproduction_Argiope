# argiopeSegmentR — the installable package

```r
install.packages("argiopeSegmentR_0.1.0.tar.gz", repos = NULL, type = "source")
library(argiopeSegmentR)
argiope_status()          # checks the machine; argiope_install_python() builds the Python side
```

Then work through `example_argiopeSegmentR.R` one step at a time — either the copy beside
this file, or the one the package installs with itself. They are the same file:

```r
file.edit(system.file("examples", "example_argiopeSegmentR.R", package = "argiopeSegmentR"))
```

The tarball is self-contained: it carries the Python side (`adapt_unet.py`,
`repro_segmentr.py`, `argiope_unet.py`), the trained U-Net (125 MB), nine sample images and
the worked example, all reachable through `system.file(package = "argiopeSegmentR")`. The only thing it cannot
carry is a Python interpreter.

**This duplicates the loose copies in the parent folder.** `argiope_pipeline_segmentR.R` plus
`adapt_unet.py`, `checkpoints/` and the rest are the same code and the same weights, kept for
reading and for running without installing anything. Once you have settled on one of the two
routes, the other can be deleted — they are about 125 MB each.

The R code in the package is lifted verbatim from `argiope_pipeline_segmentR.R`; only two path
helpers differ, because an installed package finds its files through `system.file()` and writes
its config to a user directory rather than beside a script.
