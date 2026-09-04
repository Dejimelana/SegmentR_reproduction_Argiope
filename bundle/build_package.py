"""Build the argiopeSegmentR R package from the single-file pipeline.

The R code is the same code, not a rewrite: sections 1-3 of
argiope_pipeline_segmentR.R are lifted verbatim, with exactly two path helpers
re-pointed at the installed package (`.here()`) and at a writable config
directory (`python_path.txt`). Keeping it a lift rather than a fork is what stops
the package and the script drifting apart.
"""
import re
import shutil
from pathlib import Path

ROOT = Path(r"C:\Users\Usuario\Documents\Research_Projects\Argiope")
SRC = ROOT / "Argiope_SegmenteR.AlbertoJimenez"
PKG = Path(r"C:\Users\Usuario\.claude\jobs\0a25c1ec\tmp\pkg") / "argiopeSegmentR"

if PKG.parent.exists():
    shutil.rmtree(PKG.parent)
(PKG / "R").mkdir(parents=True)
(PKG / "man").mkdir()
(PKG / "inst" / "checkpoints").mkdir(parents=True)
(PKG / "inst" / "examples").mkdir(parents=True)

# ---------------------------------------------------------------- the R code
text = (SRC / "argiope_pipeline_segmentR.R").read_text(encoding="utf-8")
# anchored to a line start: the file's own header quotes "##  SECTION 4" inside the
# snippet that tells a reader how to load sections 1-3, and an unanchored search
# finds THAT first and cuts the package down to nothing.
body = text[:text.index("\n##  SECTION 4") + 1 - 4]
assert body.count("\n##  SECTION ") == 3, "expected sections 1-3, got " + str(
    body.count("\n##  SECTION "))

# drop the file-header block: a package has DESCRIPTION for that
body = body[body.index("## #####"):]

prelude = '''## argiopeSegmentR — opisthosoma masks and CIELAB palettes for Argiope photographs.
##
## The R code below is lifted verbatim from argiope_pipeline_segmentR.R, the single-file
## version of the same workflow, so the two cannot drift. Only two things differ, and both
## are about where files live once the code is installed rather than sourced:
##
##   .here()      the installed package directory, which is where adapt_unet.py and the
##                checkpoint end up (inst/ at build time)
##   .cfg_file()  a writable per-user config path for the recorded interpreter; the install
##                directory may be read-only

.here <- function() system.file(package = "argiopeSegmentR")

.cfg_file <- function() {
  d <- tools::R_user_dir("argiopeSegmentR", which = "config")
  if (!dir.exists(d)) dir.create(d, recursive = TRUE, showWarnings = FALSE)
  file.path(d, "python_path.txt")
}

'''

# remove the script-path machinery and the sourced-file .here(); the package has its own
start = body.index(".ARGIOPE_SCRIPT <- local({")
end = body.index("dirname(.ARGIOPE_SCRIPT)\n}")
body = body[:start] + body[end + len("dirname(.ARGIOPE_SCRIPT)\n}") + 1:]

n = body.count('file.path(.here(), "python_path.txt")')
body = body.replace('file.path(.here(), "python_path.txt")', ".cfg_file()")
assert n == 3, f"expected 3 python_path.txt sites, found {n}"

# the CFG list is a script-level convenience; a package must not carry global state
cfg_start = body.index("## What to process, and where the results go")
cfg_end = body.index("## Packages ---")
body = body[:cfg_start] + body[cfg_end:]
body = body.replace('CARD$per_page, ": ", argiope_pages(x, CARD$per_page)',
                    '6, ": ", argiope_pages(x, 6)')
body = body.replace('" · pages of ", CFG$per_page, ": ", argiope_pages(x, CFG$per_page)',
                    '" · pages of 6: ", argiope_pages(x, 6)')

(PKG / "R" / "argiopeSegmentR.R").write_text(prelude + body, encoding="utf-8")
print(f"R/argiopeSegmentR.R  {len((prelude + body).splitlines())} lineas")

# ---------------------------------------------------------------- assets
for f in ("adapt_unet.py", "repro_segmentr.py", "argiope_unet.py"):
    shutil.copy2(SRC / f, PKG / "inst" / f)
shutil.copy2(SRC / "checkpoints" / "opistho_unet.pt",
             PKG / "inst" / "checkpoints" / "opistho_unet.pt")
shutil.copytree(SRC / "sample_images", PKG / "inst" / "sample_images")

# the worked example ships INSIDE the package: installing the tarball alone, with no
# hand-over folder around it, still gets you the walkthrough. It lives in two places,
# so refuse to build on a drifted pair rather than pick one silently.
example = Path(__file__).resolve().parent / "example_argiopeSegmentR.R"
beside_tarball = SRC / "package" / "example_argiopeSegmentR.R"
if beside_tarball.exists():
    assert (example.read_text(encoding="utf-8")
            == beside_tarball.read_text(encoding="utf-8")), (
        f"{example} and {beside_tarball} have drifted apart")
shutil.copy2(example, PKG / "inst" / "examples" / example.name)
print("inst/: python + checkpoint + sample_images + examples/" + example.name)

# ---------------------------------------------------------------- DESCRIPTION / NAMESPACE
(PKG / "DESCRIPTION").write_text("""Package: argiopeSegmentR
Type: Package
Title: Opisthosoma Masks and CIELAB Palettes for Argiope Photographs
Version: 0.1.0
Authors@R: person("Alberto", "Jimenez", role = c("aut", "cre"),
                  email = "de.jimenez.dataprojects@gmail.com")
Description: Segments the opisthosoma (abdomen) of orb-weaving spiders of the genus
    Argiope with a trained U-Net, then extracts a dominant-colour palette in CIELAB,
    draws a paginated gallery and a four-panel dashboard per specimen. The colour,
    artefact and quality-control stages are a port of SegmentR (Boyko 2025, MIT); the
    segmenter is a U-Net trained on hand-labelled masks. The model itself runs in Python
    via PyTorch: this package calls it across a process boundary, once per folder, and
    does the reading and drawing in R.
License: MIT + file LICENSE
Encoding: UTF-8
Depends: R (>= 4.1)
Imports: jsonlite, jpeg, png, grDevices, graphics, stats, utils, tools
Suggests: reticulate
SystemRequirements: Python (>= 3.9) with torch, segmentation-models-pytorch,
    numpy, opencv-python, scikit-learn, scikit-image, scipy, matplotlib, pillow.
    argiope_install_python() can build this from R.
""", encoding="utf-8")

(PKG / "LICENSE").write_text("""YEAR: 2026
COPYRIGHT HOLDER: Alberto Jimenez

The colour, artefact and QA stages are ported from SegmentR (Boyko 2025),
also MIT licensed; see inst/repro_segmentr.py for the ported code.
""", encoding="utf-8")

exports = ["argiope_status", "argiope_python", "argiope_use_python", "argiope_install_python",
           "install_r_packages", "argiope_gallery", "argiope_load_gallery", "argiope_items",
           "argiope_palette_of", "argiope_pick", "argiope_pick_one", "argiope_pages",
           "argiope_plot", "argiope_dashboard", "argiope_pdf",
           "argiope_card", "argiope_card_grid"]
ns = "\n".join(f"export({e})" for e in exports)
ns += "\nS3method(print, argiope_gallery)\n"
ns += "importFrom(grDevices, convertColor, col2rgb, dev.off, pdf, png)\n"
ns += "importFrom(graphics, par, plot.new, plot.window, rasterImage, rect, text, title,\n"
ns += "           axis, box, hist, lines, mtext, segments, strwidth)\n"
ns += "importFrom(stats, median, complete.cases)\n"
ns += "importFrom(utils, read.csv, select.list, install.packages)\n"
(PKG / "NAMESPACE").write_text(ns, encoding="utf-8")
print(f"NAMESPACE: {len(exports)} funciones exportadas")

# ---------------------------------------------------------------- minimal docs
def rd(name, title, aliases, usage, desc, value="", seealso=""):
    a = "\n".join(f"\\alias{{{x}}}" for x in aliases)
    v = f"\\value{{{value}}}\n" if value else ""
    s = f"\\seealso{{{seealso}}}\n" if seealso else ""
    (PKG / "man" / f"{name}.Rd").write_text(
        f"\\name{{{name}}}\n{a}\n\\title{{{title}}}\n"
        f"\\description{{{desc}}}\n\\usage{{\n{usage}\n}}\n{v}{s}", encoding="utf-8")

rd("argiopeSegmentR-package", "Opisthosoma masks and CIELAB palettes",
   ["argiopeSegmentR-package", "argiopeSegmentR"],
   "## See argiope_status() first, then argiope_gallery().",
   "Segments the abdomen of Argiope spiders with a trained U-Net and describes its colour. "
   "Run \\code{argiope_status()} first: it reports whether the Python side is ready and, if "
   "not, \\code{argiope_install_python()} builds it from R.")

rd("setup", "Check and build the Python side",
   ["argiope_status", "argiope_python", "argiope_use_python", "argiope_install_python",
    "install_r_packages"],
   "argiope_status()\nargiope_python(python = NULL)\nargiope_use_python(path)\n"
   "argiope_install_python(envname = \"argiope-segmentr\", gpu = FALSE)\ninstall_r_packages()",
   "The segmenter runs in Python. \\code{argiope_status} reports whether an interpreter with "
   "the needed packages is reachable; \\code{argiope_install_python} creates one with "
   "reticulate and records it; \\code{argiope_use_python} records one you already have.",
   "\\code{argiope_status} returns TRUE when everything is ready.")

rd("gallery", "Run the segmenter over a folder and read the result",
   ["argiope_gallery", "argiope_load_gallery", "argiope_items", "argiope_palette_of"],
   "argiope_gallery(dir, out = NULL, run_id = \"r-gallery\", n = NULL, seed = 42,\n"
   "                n_colors = 5, python = NULL, adapter = NULL, reuse = TRUE,\n"
   "                quiet = FALSE, weights = NULL, device = NULL)\n"
   "argiope_load_gallery(run_dir)\nargiope_items(g)\nargiope_palette_of(g, image)",
   "\\code{argiope_gallery} loads the model once and walks the whole folder, then returns a "
   "gallery object. \\code{argiope_items} is one row per image, including why an image "
   "produced no mask. \\code{argiope_palette_of} is that image's colours: HEX, CIELAB "
   "coordinates and coverage.",
   "A gallery object, or a data frame.")

rd("plots", "Draw the gallery",
   ["argiope_plot", "argiope_pages", "argiope_dashboard", "argiope_pdf", "argiope_pick",
    "argiope_pick_one"],
   "argiope_plot(g, page = 1, per_page = 6, select = NULL, include_empty = FALSE,\n"
   "             ncol = NULL, maxdim = 520)\nargiope_pages(g, per_page = 6, select = NULL,\n"
   "             include_empty = FALSE)\n"
   "argiope_dashboard(g, image = NULL, file = NULL, maxdim = 700,\n"
   "                  width = 1500, height = 1000, res = 130)\n"
   "argiope_pdf(g, file = \"argiope_gallery.pdf\", per_page = 6, select = NULL,\n"
   "            include_empty = FALSE, width = 11, height = 8, maxdim = 520)\n"
   "argiope_pick(g, only_with_mask = TRUE, preselect = NULL)\n"
   "argiope_pick_one(g, only_with_mask = TRUE)",
   "\\code{argiope_plot} draws one page of a grid: photograph with the mask outlined, the "
   "palette bar, score and mask area. \\code{argiope_dashboard} draws four panels for one "
   "specimen. \\code{argiope_pick} opens a selection list in an interactive session and "
   "returns everything in a script.")
rd("cards", "The specimen card, and pages of cards",
   ["argiope_card", "argiope_card_grid"],
   "argiope_card(g, image = NULL, file = NULL, maxdim = 900, width = 5.2,\n"
   "             height = 7.1, res = 150, mar = c(0, 0, 0, 0))\n"
   "argiope_card_grid(g, page = 1, per_page = 6, ncol = NULL, select = NULL,\n"
   "                  file = NULL, card_w = 3.5, card_h = 4.8, res = 150,\n"
   "                  maxdim = 600, gap = 0.35, page_bg = \"#0B0D07\")",
   "\\code{argiope_card} draws one specimen as a card: header, the photograph with the "
   "mask outlined, the isolated opisthosoma beside its palette listed by nearest colour "
   "name, and a footer with the mask size. \\code{argiope_card_grid} lays several on a "
   "page, calling \\code{argiope_card} per panel so the two cannot diverge. The card is "
   "designed tall, so the page is sized from columns and rows times that shape.",
   "Invisibly, the palette and mask statistics of the card drawn.")
print(f"man/: 5 ficheros Rd con alias para las {len(exports)} funciones")
