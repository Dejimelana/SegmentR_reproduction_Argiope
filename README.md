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
├── requirements.txt             transformers / timm / accelerate (not added to the parent project)
├── repro_segmentr.py            the ported pipeline
├── tests/test_repro_segmentr.py pure-function tests: no weights, no GPU, no network
└── outputs/<run_id>/            run artefacts (git-ignored)
```

## Status

`PROMPT.md` is written; the script is not implemented yet. To build it, execute `PROMPT.md` in
a Claude Code session started from the Argiope project root.

## Attribution and licence

The method, and the reference implementation being ported, are James Boyko's. SegmentR is
released under the **MIT licence**, so code ported verbatim is permitted provided its copyright
notice and licence text travel with it — see `THIRD_PARTY_NOTICES.md` (added alongside the
first ported code). Cite the paper, not this repository, for the method itself.
