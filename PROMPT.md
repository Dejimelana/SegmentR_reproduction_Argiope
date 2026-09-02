# Reproduction probe — SegmentR (Boyko 2025) on *Argiope*

Paste this at the start of a session dedicated to reproducing one paper's method on our
images. This is **not** `docs/INITIAL_PROMPT.md`: that prompt drives normal increments of the
product; this one drives a **throwaway probe** whose deliverable is an answer and a
recommendation, not a feature.

---

You are working on **Argiope**, a computer-vision project that describes and identifies the
**opisthosoma** (abdomen) of *Argiope* spp. from images. Our own pipeline is YOLO (whole
spider) → SAM 3 (opisthosoma mask, text-prompted) → CIELAB palette → pattern terms → ledger.
**You are not touching it today.**

## Mission

Reproduce, faithfully and in isolation, the method of:

> Boyko, J. D. (2025). *SegmentR: Deep learning for automated segmentation with an R
> interface.* **Ecological Informatics** 90:103259. doi:10.1016/j.ecoinf.2025.103259
> Preprint (as *SegColR*): bioRxiv 10.1101/2024.07.28.605475
> Code: <https://github.com/jboyko/SegmentR>

…on real *Argiope* images from `data/raw/gbif/`, and report what it does with our spiders.

Its method is: **GroundingDINO** with a text prompt → boxes → **SlimSAM** prompted by those
boxes → masks → score/label filtering → **colour extraction in CIELAB** → **re-loadable JSON
artefacts** → QA plots → batch over a folder. (The R layer is the paper's accessibility
contribution; it is not part of what you reproduce — port the method to one Python script.)

The question you are answering: **what does a zero-shot, text-prompted GroundedSAM pipeline
recover from our images, and which parts of it are worth promoting into `src/argiope/`?**

You may not "improve" the method. Reproduce it, then write down where it fails.

## Hard rule — two repositories, one folder

`repro/segmentr/` is **its own git repository**, versioned at
<https://github.com/Dejimelana/SegmentR_reproduction_Argiope>. It sits nested inside the working
tree of the Argiope project (remote `Dejimelana/OphistoHEX`), which ignores it, so the two
histories never mix.

All work goes under `repro/segmentr/`, and every commit lands in the reproduction repository.
Do not create or modify a single file outside that folder — the Argiope working tree must be
clean when you are done. Explicitly off-limits: `src/argiope/**`, `configs/default.yaml`,
`pyproject.toml`, `data/taxonomy/*.yaml`, `data/processed/ledger.jsonl`, `docs/**`, the root
`tests/**` and the root `.gitignore`. Everything under `data/` is **read-only** input.

The script must **not import `argiope`** — no `build_segmenter`, no `extract_palette`, no
`TaxonomyRegistry`. A reproduction that leans on the pipeline it will eventually be compared
against is not a reproduction; that comparison is a separate, later job.

Because the probe lives in a nested repository but reads the parent project's images, resolve
paths explicitly: accept `--argiope-root`, defaulting to `Path(__file__).resolve().parents[2]`,
and build every input path from it. Never depend on the current working directory.

Layout (`PROMPT.md`, `README.md` and `.gitignore` already exist):

```
repro/segmentr/
├── PROMPT.md                    (this file — do not edit)
├── README.md                    (update its Status section once the script runs)
├── .gitignore
├── requirements.txt
├── repro_segmentr.py
├── tests/test_repro_segmentr.py
├── THIRD_PARTY_NOTICES.md       (see below)
└── outputs/<run_id>/            (git-ignored)
```

**Licensing — this repository is public.** SegmentR is MIT-licensed, so porting its code
verbatim is permitted, but the notice has to travel with it. When you port, create
`THIRD_PARTY_NOTICES.md` holding SegmentR's copyright line and the full MIT text, and mark each
ported function in the source with a comment naming the upstream file it came from.

## Read the original before writing code

Port these files; where this prompt and the source disagree, **the source wins**:

- `https://raw.githubusercontent.com/jboyko/SegmentR/HEAD/inst/python/rseg/detection.py`
- `https://raw.githubusercontent.com/jboyko/SegmentR/HEAD/inst/python/rseg/segmentation.py`
- `https://raw.githubusercontent.com/jboyko/SegmentR/HEAD/inst/python/rseg/utils.py`
- `https://raw.githubusercontent.com/jboyko/SegmentR/HEAD/inst/python/rseg/visualization.py`
- `https://raw.githubusercontent.com/jboyko/SegmentR/HEAD/inst/python/main.py`
- `https://raw.githubusercontent.com/jboyko/SegmentR/HEAD/R/image_analysis.R` (colour)
- `https://raw.githubusercontent.com/jboyko/SegmentR/HEAD/R/export.R` (transparent PNG)

## The method, stage by stage, with the original's values

**A — Detection (GroundingDINO).** `transformers.pipeline(task="zero-shot-object-detection")`
with `detector_id="IDEA-Research/grounding-dino-tiny"` and `threshold=0.1`. Labels are
suffixed with `"."` if they do not already end in one — the detector requires it. Keep the
original's dataclasses and names: `BoundingBox(xmin, ymin, xmax, ymax)` with an `.xyxy`
property, and `DetectionResult(score, label, box, mask=None)` with `.from_dict()`. Device:
CUDA if available, else CPU.

**B — Segmentation (SlimSAM).** `AutoModelForMaskGeneration` + `AutoProcessor` with
`segmenter_id="Zigeng/SlimSAM-uniform-77"`. Boxes go in as `input_boxes=get_boxes(detections)`,
which wraps them one level deeper than you expect: `[[[xmin,ymin,xmax,ymax], ...]]`. Then
`processor.post_process_masks(...)` with the returned `original_sizes` and
`reshaped_input_sizes`, then `refine_masks(masks, polygon_refinement)` — copy that function
verbatim from `utils.py` (float → permute → mean over the channel axis → threshold → uint8),
including its optional contour pass (`mask_to_polygon` = largest `cv2.findContours` contour,
`polygon_to_mask` = `cv2.fillPoly`). Default `polygon_refinement=False`.

**C — Filtering and merging.** Apply a post-hoc score threshold, default **0.5** (the value
used in the paper's Example 2), plus `--include-labels` / `--exclude-labels`. Multiple
detections carrying the same label are **merged into one mask** (logical OR) before colour
extraction.

**D — Part subtraction.** This is the paper's Example 3 (prompting `"fins of a fish"` to
exclude them). Run detection+segmentation a second time with the exclusion prompt, merge those
masks, and subtract: `roi &= ~excluded`. For us this is the analogue of isolating the
opisthosoma from the rest of the animal, so make it first-class, not an afterthought.

**E — Colour in CIELAB** (port of `extract_colors(image, mask, n_colors = 5, custom_colors = NULL)`):

- masked pixels only: `image[mask.astype(bool)]` → `(N, 3)` float sRGB in [0, 1];
- sRGB → Lab with `skimage.color.rgb2lab` (D65, 2°), which matches R's
  `convertColor(from = "sRGB", to = "Lab")`;
- default path: k-means on the Lab pixels with `n_colors = 5`;
- supervised path (`--custom-colors`): reference hex codes → RGB/255 → Lab, then assign every
  pixel to the nearest reference by **Euclidean distance in Lab**;
- per-cluster output columns, with the original's names: `lab_l`, `lab_a`, `lab_b`,
  `hex_color`, `cluster_size`. Hex comes from converting the Lab centroid back to RGB;
- also report `mean_color` and `median_color`, computed **in RGB** over the masked pixels
  (`colMeans` / per-channel `median`), not in Lab. That is what the original does — reproduce
  it, do not fix it.

**F — The JSON artefact.** One JSON per image, in the original's schema:

```json
[{"label": "…", "score": 0.0, "box": {"xmin": 0, "ymin": 0, "xmax": 0, "ymax": 0}, "mask": null}]
```

Masks are cropped to the image dimensions before serialising. The *point* of this artefact is
that colour analysis re-runs without re-running the models — so implement `--from-json <run_dir>`
and prove it.

Two documented deviations are expected here, and both go in the report:

1. the original serialises masks as 2-D integer lists; at ~1 MP × 40 images that is hundreds of
   MB of text, so by default write `mask_<i>.png` beside the JSON and put its filename in
   `"mask"`. Keep `--faithful-json` to restore the literal format;
2. R's `kmeans()` defaults to Hartigan-Wong with `nstart = 1` and no seed. Use
   `sklearn.cluster.KMeans(n_clusters=n_colors, n_init=10, random_state=seed)` so runs are
   reproducible, and say so.

**G — QA visuals.** Mask overlay with score labels, a colour-swatch strip, an RGB histogram, a
recoloured mask (every pixel replaced by its cluster centroid), and transparent-PNG export with
the original's `crop` and `remove_overlap` options.

**H — Batch.** Walk the folder printing `Processing image X of Y`. In the original this loop
lives in R; here it goes in the script.

## Argiope-specific settings

- ROI prompt: `"the abdomen of a spider."` — **do not re-derive this.** `configs/default.yaml`
  already records, from real runs, that `"opisthosoma"` is out-of-vocabulary for text-prompted
  models and that richer phrasing recalls the abdomen far better than `"spider abdomen"`. Read
  that comment before choosing prompts.
- Whole-animal prompt: `"a spider."`
- Subtraction prompts: `"the legs of a spider."`, optionally `"the head of a spider."`
- Score threshold: keep the paper's **0.5** as the default — that is the value being
  reproduced. But the same `configs/default.yaml` comment block records that for us `0.5`
  *dropped valid abdomen masks*, because anatomical concepts score lower than whole objects,
  which is why our own segmenter runs at `0.3`. That was measured on SAM 3, not on
  GroundingDINO + SlimSAM, so do not assume it transfers: run the probe at `0.5`, then re-run
  at `0.3`, and report the difference in hit rate. Two runs, one paragraph in the report.
- Reference palette for `--custom-colors` (define it **inside the script**; do not touch
  `data/taxonomy/features.yaml`): yellow, black, silver/white, brown.
- Images: seeded, species-stratified sample of `data/raw/gbif/<taxon_id>/*.jpg` (~2999 images;
  ID targets are *A. bruennichi*, *A. aurantia*, *A. argentata*). Default `--n 40 --seed 42`.
- Also accept `--images-from data/interim/annotate/manifest.jsonl` (read-only) so the probe can
  be re-run over the exact images already cropped for hand annotation — that gives a future IoU
  job something to line up against.

Target CLI:

```
python repro/segmentr/repro_segmentr.py \
  --images data/raw/gbif --n 40 --seed 42 \
  --prompt "the abdomen of a spider." \
  --exclude-prompt "the legs of a spider." \
  --detector-id IDEA-Research/grounding-dino-tiny --box-threshold 0.1 \
  --segmenter-id Zigeng/SlimSAM-uniform-77 --score-threshold 0.5 \
  --n-colors 5 [--custom-colors argiope] [--polygon-refinement] \
  [--faithful-json] [--from-json repro/segmentr/outputs/<run_id>] \
  [--argiope-root <path to the Argiope checkout>] \
  --out repro/segmentr/outputs
```

## Environment

Check imports first (`transformers`, `timm`, `accelerate`); if any are missing, install them
into the `argiope` conda env from `repro/segmentr/requirements.txt`. **Do not add them to
`pyproject.toml`** — the probe must not change the project's dependency contract. `torch`,
`opencv-python`, `scikit-learn`, `scikit-image` and `matplotlib` are already available.

Windows note: use the HuggingFace `transformers` implementations of GroundingDINO and SAM
(pure Python). Do **not** install the original `groundingdino` package — it requires a
CUDA/MSVC build step. Both checkpoints are ungated, so unlike SAM 3 no `hf auth login` is
needed; they download on first run.

If the environment cannot be prepared, stop and report the blocker instead of forcing it.

## Repo conventions that still apply

- **Never silently drop data.** Every skipped image — unreadable, no detection above threshold,
  empty mask after subtraction — is logged with its reason and counted in the report.
- Seed everything; write every parameter and seed to `run_config.json`.
- Parameters through the CLI; no module-level globals.
- No PANTONE claims: HEX + Lab coordinates + cluster sizes, nothing more.

## Definition of done

1. `python repro/segmentr/repro_segmentr.py --images data/raw/gbif --n 40 --seed 42` runs end
   to end.
2. `repro/segmentr/outputs/<run_id>/` holds: per-image JSON (+ mask PNGs), `colors.csv` (one row
   per image × cluster: image, species, label, score, `lab_l`, `lab_a`, `lab_b`, `hex_color`,
   `cluster_size`, plus mean/median hex), the QA PNGs, `run_config.json`, and `skipped.csv`.
3. `--from-json` re-runs colour extraction over a finished run **without loading either model**.
4. `pytest repro/segmentr` passes: pure functions only (mask merge, part subtraction, colour
   extraction on synthetic arrays) — no weights, no GPU, no network. Follow the style of
   `tests/test_segmentation_backend.py`, which tests only the pure `_select_opisthosoma`.
   The root suite will not collect these (`pyproject.toml` sets `testpaths = ["tests"]`); run
   plain `pytest` too and confirm it is still green and untouched.
5. `repro/segmentr/outputs/<run_id>/REPORT.md` exists and contains:
   - counts — processed / skipped, with reasons;
   - hit rate per prompt (fraction of images where that prompt produced any mask above threshold);
   - **failure modes, concretely**: whole-spider vs abdomen granularity, bleed into legs or into
     web/stabilimentum, shadow-driven colour distortion (a caveat the paper itself raises for
     citizen-science images), background dominating the palette;
   - a `DEVIATIONS` section listing every departure from the original;
   - a closing **recommendation**: which single piece — most likely part-subtraction, the
     re-loadable JSON artefact, or supervised colour assignment — earns promotion into
     `src/argiope/`, and which do not.
6. `git status` in the Argiope working tree is **clean**; inside `repro/segmentr/` it shows only
   the probe's own files, committed to the reproduction repository. `THIRD_PARTY_NOTICES.md`
   exists if any upstream code was ported verbatim.

## Non-goals

No IoU or any other mask metric. No ground truth, no annotation work. No ledger writes. No
changes to the package. No training or fine-tuning. No quantitative comparison against our
YOLO + SAM 3 pipeline. No improvements to the paper's method — if something is wrong with it,
that belongs in `REPORT.md`, not in the code.

## Start here

Restate the probe and its done-criteria in one paragraph. Then read the seven original source
files above **before** writing any code. Port them into
`repro/segmentr/repro_segmentr.py`, and run with `--n 4` first to confirm both models load and
the artefacts are shaped correctly, before committing to the full 40 images.
