# SegmentR reproduction — *Argiope*

An independent reproduction of the method in:

> Boyko, J. D. (2025). *SegmentR: Deep learning for automated segmentation with an R
> interface.* **Ecological Informatics** 90:103259. doi:10.1016/j.ecoinf.2025.103259
> Preprint (as *SegColR*): bioRxiv 10.1101/2024.07.28.605475 ·
> Code: <https://github.com/jboyko/SegmentR>

applied to photographs of orb-weaving spiders of the genus *Argiope*.

## What this is — and what it is not

An **exploratory probe**, not a package. Its deliverable is an answer: what does a zero-shot,
text-prompted GroundedSAM pipeline (GroundingDINO → SlimSAM → colour extraction in CIELAB)
recover from *Argiope* photographs, and which parts of that method are worth adopting?

It deliberately carries **no validation** — no IoU, no ground truth, no quantitative comparison
against another pipeline. The upstream paper reports none either; measuring segmentation
quality against hand-labelled masks is a separate piece of work.

## Relationship to the parent project

This repository sits nested inside the working tree of the Argiope project
(<https://github.com/Dejimelana/OphistoHEX>), whose images it reads **read-only**. It imports
nothing from that codebase: a reproduction that leans on the pipeline it will eventually be
compared against is not a reproduction. The parent repository ignores this directory, so the
two histories stay independent.

The probe therefore expects to be run from the Argiope project root, or pointed at it with
`--argiope-root`.

## Layout

```
.
├── PROMPT.md                    the executable specification of the reproduction
├── README.md
├── REPORT.md                    findings, deviations, and the §9 correction
├── requirements.txt             transformers / timm / accelerate (not added to the parent project)
├── repro_segmentr.py            the ported pipeline — the reproduction proper, unchanged
├── adapt_unet.py                the adapter: Argiope's trained U-Net -> stages E/F/G
├── R/argiope.R                  thin R interface over the `argiope describe` contract
├── experiments/                 follow-up work that is NOT part of the reproduction
│   ├── candidate_selection.py   measured the mechanism claim in REPORT §3.1
│   └── sanity_gt_vs_unet.py     does a U-Net mask give the same colours as a hand mask?
├── tests/                       pytest: pure functions, the adapter, and the R layer
└── outputs/<run_id>/            run artefacts (git-ignored)
```

## Status

**Complete.** `repro_segmentr.py` ports the method end to end and has been run on 40 seeded,
species-stratified GBIF photographs at the paper's score threshold of 0.5 and again at 0.3.
`pytest repro/segmentr` covers the pure functions (50 tests: no weights, no GPU, no network).

**Finding, in one line:** the method as published does not recover the opisthosoma from our
images. All three text prompts return the same whole-animal mask (median abdomen-to-whole
area ratio 1.000), so part subtraction deletes a median 99.91% of the region it is meant to
refine, and what survives is background — foliage, silk, a wall, a hand. One image in 40
produced a mask I would accept as an abdomen, and only because part subtraction failed to
run on it.

Read `REPORT.md` for the counts, per-prompt hit rates, failure modes, the full DEVIATIONS
list and the recommendation about what earns promotion into `src/argiope/`. The same report
is written into each run directory. Run artefacts live in `outputs/<run_id>/` and are
git-ignored; `run_config.json` reproduces a run on its own.

## What came after the replication

The replication is finished and its script is frozen. Two things were built on top of it,
both kept separate so the reproduction stays a reproduction.

**`adapt_unet.py` — the adapter.** The finding was that SegmentR's *method* (the GroundedSAM
stages A–D that isolate a part from a text prompt) does not recover the opisthosoma, while
everything downstream of the mask is sound and segmenter-agnostic. The adapter feeds those
surviving stages from Argiope's trained U-Net instead:

```
U-Net mask ─► DetectionResult ─► extract_colors   (E, CIELAB)
                               ├► write_image_json (F, re-loadable artefact)
                               └► QA figure + transparent cut-out (G)
```

Stages A–D are not imported. Two deliberate omissions: `combine_masks`/`merge_masks_by_label`
are bypassed because they OR same-label detections into one mask and destroy per-specimen
identity, and part subtraction is gone because a segmenter trained on the part does not need
it. `--from-json` re-runs colour with the U-Net never loaded.

```bash
python repro/segmentr/adapt_unet.py --images data/interim/opistho_seg/images --n 20
python repro/segmentr/adapt_unet.py --from-json repro/segmentr/outputs/<run_id>
```

**`R/argiope.R` — the R layer.** Wraps, never reimplements: the segmenter stays in torch and R
calls the published CLI contract (`argiope describe --json --mask`) across a process boundary,
so R needs no Python configuration. Requires `jsonlite`.

```r
source("repro/segmentr/R/argiope.R")
options(argiope.executable = ".../envs/argiope/Scripts/argiope.exe")
d <- argiope_describe("spider.jpg")
argiope_palette(d)          # hex, name, coverage, ci_low, ci_high, delta_e
```

**Does the segmenter's error move the colours?** `experiments/sanity_gt_vs_unet.py` compares,
over the held-out split, the palette taken from the hand-drawn mask against the palette taken
from the U-Net mask. On the 18 of 30 images with IoU ≥ 0.7 the dominant-cluster ΔE is 0.8
(median) and the coverage-weighted palette ΔE is 1.51, with all 18 inside ΔE 5 — below the
ΔE 2.3 just-noticeable difference, so the colour stage is robust to the segmenter's residual
error. The remaining tail is not: 7 of 30 held-out images produce an **empty** mask, which the
adapter logs and counts as a skip rather than passing an empty region downstream.

## Attribution and licence

The method, and the reference implementation being ported, are James Boyko's. SegmentR is
released under the **MIT licence**, so code ported verbatim is permitted provided its copyright
notice and licence text travel with it — see `THIRD_PARTY_NOTICES.md` (added alongside the
first ported code). Cite the paper, not this repository, for the method itself.
