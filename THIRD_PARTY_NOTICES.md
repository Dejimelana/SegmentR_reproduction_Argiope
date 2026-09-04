# Third-party notices

## SegmentR (also released as SegColR)

`repro_segmentr.py` contains code ported — in places verbatim — from **SegmentR**, the
reference implementation accompanying:

> Boyko, J. D. (2025). *SegmentR: Deep learning for automated segmentation with an R
> interface.* **Ecological Informatics** 90:103259. doi:10.1016/j.ecoinf.2025.103259
> Preprint (as *SegColR*): bioRxiv 10.1101/2024.07.28.605475

- Upstream: <https://github.com/jboyko/SegmentR>
- Author/maintainer: James Boyko <jboyko@umich.edu>
- Licence: MIT (`LICENSE`: `YEAR: 2024`, `COPYRIGHT HOLDER: SegColR authors`)

Ported units, each marked in the source with a `# --- ported from <upstream file> ---`
comment naming its origin:

| Ported into `repro_segmentr.py`                                 | Upstream file                       |
| --------------------------------------------------------------- | ----------------------------------- |
| `BoundingBox`, `DetectionResult`, `load_image`, `mask_to_polygon`, `polygon_to_mask`, `get_boxes`, `refine_masks` | `inst/python/rseg/utils.py`         |
| `detect`                                                          | `inst/python/rseg/detection.py`     |
| `segment`                                                         | `inst/python/rseg/segmentation.py`  |
| `annotate`, `plot_detections`                                     | `inst/python/rseg/visualization.py` |
| JSON artefact schema and the batch entry point                    | `inst/python/main.py`               |
| `combine_masks`, `exclude_masks`, `extract_colors`, `process_masks_and_extract_colors` | `R/image_analysis.R`                |
| `export_transparent_png`                                          | `R/export.R`                        |

Sources were retrieved from the repository's default branch (`HEAD`) on 2026-09-02.

### MIT Licence

Copyright (c) 2024 SegColR authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Model checkpoints

Downloaded at runtime from the HuggingFace Hub; neither is redistributed here.

- `IDEA-Research/grounding-dino-tiny` — Grounding DINO, Apache-2.0.
- `Zigeng/SlimSAM-uniform-77` — SlimSAM, Apache-2.0. Distilled from Meta's
  Segment Anything (SAM), itself Apache-2.0.

## Images

Input photographs under `data/raw/gbif/` belong to the parent Argiope project and are read
**read-only**. Run artefacts (masks, QA figures, cut-outs) live in `outputs/` and are
git-ignored.

**One exception, added deliberately:** `docs/GALERIA.md` and `docs/img/` reproduce
**downscaled thumbnails** of the 81 photographs for which the pipeline produced a mask, so the
gallery is viewable in the repository. These are third-party photographs obtained via GBIF
(largely iNaturalist observations) and are reproduced **with per-image attribution** — the
photographer, the licence and a link to the GBIF occurrence appear beside every image in
`docs/GALERIA.md`. Licences of the published thumbnails:

| Licence | Images |
| --- | ---: |
| [CC-BY-NC](https://creativecommons.org/licenses/by-nc/4.0/) | 72 |
| [CC-BY](https://creativecommons.org/licenses/by/4.0/) | 7 |
| [CC0](https://creativecommons.org/publicdomain/zero/1.0/) | 2 |

All three permit redistribution with attribution; **CC-BY-NC forbids commercial use**, which
therefore constrains that part of this repository regardless of the MIT licence covering the
code. Full-resolution originals are not redistributed — follow the GBIF link for the source
image. The derived masks and palettes are outputs of this analysis; the photographs remain
their authors'. If any rights holder objects, the thumbnail will be removed on request.
