# Reproduction report — SegmentR (Boyko 2025) on *Argiope*

**Run:** `n40-thr050` (primary) and `n40-thr030` (threshold re-run), 2026-09-02.
**Method reproduced:** Boyko, J. D. (2025). *SegmentR: Deep learning for automated
segmentation with an R interface.* Ecological Informatics 90:103259.
**Port:** `repro/segmentr/repro_segmentr.py` (see `THIRD_PARTY_NOTICES.md`).
**Scope:** no validation. No IoU, no ground truth, no comparison against the parent
project's YOLO + SAM 3 pipeline. Every number below is one the method itself produced
(mask areas, subtraction residuals, cluster statistics) or one I counted by eye from the
QA images, which is stated wherever it applies.

---

## The answer, up front

A zero-shot, text-prompted GroundedSAM pipeline, reproduced faithfully at the paper's own
settings, **does not recover the opisthosoma from our images**. Across 40 seeded,
species-stratified GBIF photographs it produced a mask I would accept as "the abdomen and
not much else" on **one** image — and that one succeeded only because the part-subtraction
stage failed to execute on it. Where subtraction did run, it removed a median of **99.91%**
of the region it was supposed to refine, and the pixels left behind were background:
foliage, silk, a wall, and in one case the photographer's hand.

This is a negative result about *this method on these images*, not about text-prompted
segmentation in general. The limits of that claim are in **Skeptic's note** below.

---

## 1. Counts

Both runs used the same seed (42), the same 40-image species-stratified sample, and the
same detector settings; only `--score-threshold` differs. The detector produced identical
box counts in both runs (575 abdomen boxes, 981 leg boxes, 144 whole-spider boxes over the
40 images), which is the reproducibility check on the sampling and seeding.

| | `n40-thr050` (paper's value) | `n40-thr030` |
|---|---|---|
| images sampled | 40 (14 argentata / 13 aurantia / 13 bruennichi) | 40 (same images) |
| produced a palette | **31** | **39** |
| skipped | **9** | **1** |
| skip reason | `No masks meet the score threshold` — all 9 | same — 1 |
| skips by species | argentata 1, aurantia 4, bruennichi 4 | aurantia 1 |

No image was skipped as unreadable, and none was dropped silently: every skip is a row in
`skipped.csv` with its stage and reason.

## 2. Hit rate per prompt

Fraction of the 40 images on which that prompt produced at least one mask above the score
threshold.

| prompt | role | hit rate @ 0.5 | hit rate @ 0.3 | median boxes/image @ 0.1 |
|---|---|---|---|---|
| `the abdomen of a spider.` | ROI | **78%** (31/40) | **98%** (39/40) | 13 |
| `the legs of a spider.` | subtraction | **75%** (30/40) | **100%** (40/40) | 24 |
| `a spider.` | reference | **80%** (32/40) | **98%** (39/40) | 2 |

Every prompt produced at least one box on all 40 images at the paper's box threshold of
0.1; the hit rates above are what survives the *post-hoc* score filter.

## 3. Failure modes

### 3.1 There is no part-level granularity: all three prompts return the same mask

At 0.5, on the 29 images where both the abdomen prompt and the whole-animal prompt fired,
the ratio of *abdomen-mask area* to *whole-spider-mask area* was:

```
min 0.589   q25 0.998   median 1.000   q75 1.002   max 1.251
within +/-5% of the whole-animal mask:  23 / 29
```

A median of exactly 1.000 is the finding. `"the abdomen of a spider."` and `"a spider."`
return the same region. So does `"the legs of a spider."` — on image
`argiope_argentata/f93be823004c0461.jpg` the three prompts returned 215,473 / 215,962 /
216,741 px respectively, a spread of 0.6%.

**Mechanism, visible in the source rather than inferred from outputs.** `refine_masks` in
`inst/python/rseg/utils.py` receives a tensor shaped `(n_boxes, n_candidates, H, W)` —
SlimSAM emits three candidate masks per box precisely to resolve the whole/part/subpart
ambiguity — and does:

```python
masks = masks.permute(0, 2, 3, 1)
masks = masks.mean(axis=-1)
masks = (masks > 0).int()
```

Averaging over the candidate axis and thresholding at `> 0` is a **union**. The step whose
job is to choose among "abdomen / abdomen+cephalothorax / whole animal" instead returns
all three. The part prompt cannot express a part. This is reproduced verbatim and pinned by
`test_refine_masks_takes_the_union_of_slimsams_three_candidates`.

At 0.3 the ratio's median rises to 1.144 (max 2.112): more boxes clear the filter, the
union grows, and the "abdomen" mask becomes *larger* than the whole-animal mask.

### 3.2 Part subtraction annihilates the region it is meant to refine

Because the ROI mask and the exclusion mask are near-identical, `roi &= ~excluded` leaves
their boundary noise. Final ROI as a fraction of the abdomen-prompt mask, per image:

| | @ 0.5 (n=31) | @ 0.3 (n=39) |
|---|---|---|
| median retention | **0.0009** (0.09%) | **0.0009** |
| retaining < 1% | 25/31 | 31/39 |
| retaining < 5% | 28/31 | 35/39 |
| max retention | 1.0000 *(subtraction did not run)* | **0.1588** |
| final ROI < 1000 px | 24/31 | 29/39 |

At 0.3 **no image retains even 16%** of its abdomen mask. The two images at 0.5 with
retention 1.0000 are the two where the legs prompt produced nothing above threshold, so
stage D never executed.

### 3.3 What survives is background, not spider

I looked at the QA composites rather than trusting the areas. The largest surviving ROI in
the whole 0.5 run — `argiope_argentata/346c97219a2ca91a.jpg`, 162,766 px, 13% retention —
is a shredded lace of **foliage**: all five dominant colours are leaf-green (`#3B5A23`,
`#8FC735`, `#6D9733`, `#1F2916`, `#B2B779`), and the spider, plainly visible in the frame,
is absent from the mask. `argiope_aurantia/2eb88eabe99929eb.jpg` masks the entire serrated
**leaf** the spider sits on and keeps its outline. On
`argiope_bruennichi/af3f59c452c25026.jpg` the spider is held on a human hand, 250 px
survive, and the five "abdomen" colours (`#D0AB87`, `#BF836A`, `#C89580`, `#A37148`,
`#7D4627`) are the photographer's **skin**.

### 3.4 Bleed into web and stabilimentum

The arachnologist's specific worry is confirmed. On
`argiope_argentata/90b24824c0a2b164.jpg` — one of the two images where subtraction did not
run — the mask is the whole spider *plus* a large irregular blob of **silk**, and the
palette is five near-neutral greys (`#D6D8D8`, `#A3A6A5`, `#757A74`, `#4B4E4E`, `#CEBBA5`,
max chroma 13.9). *Argiope* stabilimenta are bright, high-contrast and directly adjacent to
the animal, and a text-prompted model swallows them exactly as feared.

### 3.5 Background dominates the palette

Of the 31 palettes at 0.5, only **7** contain a bright-yellow-ish cluster (b\* > 35 and
L\* > 55) and **3** are effectively achromatic (max chroma < 12). For a genus whose
diagnostic abdominal colours are saturated yellow, silver-white and black, a palette set in
which three quarters of images show no yellow at all is measuring the habitat, not the
spider.

### 3.6 Shadow-driven colour distortion — the paper's own caveat, made worse by stage E

Boyko raises uncontrolled lighting as a limitation of citizen-science imagery, and the
method's own colour statistics amplify it. `extract_colors` clusters in CIELAB but computes
`mean_color` and `median_color` in **RGB** (`colMeans` / per-channel `median`). An RGB mean
of a shaded, high-contrast abdomen lands between the clusters rather than on any of them:
on the silk-contaminated argentata above the reported mean is `#9E9E9A`, which is not near
any of its five centroids. A per-channel RGB median is not even guaranteed to be a colour
present in the image. Reproduced, not fixed; pinned by
`test_mean_and_median_colour_are_taken_in_rgb_not_lab`.

### 3.7 The detector sprays boxes at threshold 0.1

Over 40 images the abdomen prompt produced 575 boxes (median 13/image, max 33) and the legs
prompt 981 (median 24, max 49) — boxes covering leaves, twigs and empty background. The
paper's design leans entirely on the post-hoc score filter to clean this up, and a single
scalar threshold cannot separate "a box on the abdomen" from "a box on a leaf" when both
score in the same range.

### 3.8 Lowering the threshold raises the hit rate and destroys the result

The parent project's `configs/default.yaml` records that 0.5 dropped valid abdomen masks
for SAM 3 and that our own segmenter therefore runs at 0.3. That does **not** transfer here.
Dropping to 0.3 raised the abdomen-prompt hit rate from 78% to 98% — and converted the
run's single correct result into grass:

| `argiope_bruennichi/ce285fafdfd82efb.jpg` | @ 0.5 | @ 0.3 |
|---|---|---|
| abdomen masks above threshold | 1 (6,156 px) | 3 (48,865 px) |
| leg masks above threshold | **0** | 4 (47,669 px) |
| final ROI | **6,156 px** | **298 px** |
| palette | `#5D5120 #EDEAB2 #A4945D #27220A #F6F098` (banded yellow/cream/black) | `#383D1A #9CA765 #737A44 #C1D67F #DADEA9` (grass) |

At 0.5 this is the one image in the run whose cut-out I would accept as an opisthosoma: a
clean *A. bruennichi* abdomen, no legs, no vegetation. It is correct *because* the legs
prompt scored nothing and stage D was skipped. At 0.3 the legs prompt fires, subtraction
runs, and the palette becomes the background. **Hit rate is not a proxy for correctness in
this pipeline**, and reporting only the former would have inverted the conclusion.

---

## 4. Objections to the method (reproduced, not fixed)

Each of these is in the port exactly as upstream has it; the objection lives here.

1. **`refine_masks` takes the union of SlimSAM's three candidates** (§3.1). This alone
   defeats part-prompted segmentation.
2. **Colour means are taken in RGB while clustering is perceptual** (§3.6).
3. **`kmeans()` is unseeded** (`R/image_analysis.R`: Hartigan-Wong, `nstart = 1`, no seed),
   so the paper's own colour output is not reproducible run to run. Deviation D2.
4. **The k-means branch of `extract_colors` is partly dead code.** After computing centres,
   control falls through to the nearest-centroid `apply(...)` loop and
   `cluster_sizes <- tabulate(cluster_assignments, nbins = n_colors)` **overwrites**
   `km_result$size`. k-means supplies centres only; sizes always come from a re-assignment
   pass. Reproduced deliberately — it changes the reported `cluster_size` values.
5. **The per-pixel `hex_colors` field is computed with a double-scaling bug**:
   `rgb(masked_pixels[, 1] / 255, ...)` divides an array already in [0, 1] by 255 again, so
   every entry is `#010101` or darker. Reproduced in `upstream_pixel_hex` and pinned by
   `test_upstream_per_pixel_hex_reproduces_the_double_scaling_bug`.
6. **The shipped `inst/python/main.py` cannot write its JSON artefact.** At `HEAD`,
   `detections_dict = [...]` is indented *inside* `if args.save_plot or args.show_plot:`
   (confirmed by parsing the file: the assignment is a child of that `If` node), while
   `json.dump(detections_dict, f)` sits at function level. Unless a plot is also requested
   the script raises `NameError`; and `--save_json` defaults to `None`, so `open(None, 'w')`
   raises `TypeError` when it is omitted. The re-loadable artefact that the paper's
   accessibility argument rests on is unreachable from the shipped CLI. The port implements
   the documented schema instead.
7. **The paper reports no segmentation validation.** No IoU, no ground truth, no
   inter-observer agreement. Nothing in the original tells a reader how often the mask is
   the thing it was asked for — which, on this evidence, is the number that matters most.

---

## 5. DEVIATIONS from the original

Every departure, however small.

| # | Deviation | Why |
|---|---|---|
| D1 | Masks serialised as `mask_<i>.png` beside the JSON, filename stored in `"mask"`, instead of upstream's 2-D integer lists. `--faithful-json` restores the literal format. | Measured: the faithful format is **236 MB for one image** (25 detections × 2048×1536), ≈9 GB for 40. The default run is 0.32 MB of JSON + 15.7 MB of PNGs for all 40. Reproduce with `--n 1 --faithful-json`. |
| D2 | `sklearn.cluster.KMeans(n_clusters=n, n_init=10, random_state=seed)` instead of R's unseeded Hartigan-Wong with `nstart = 1`. | Reproducibility; also changes the algorithm and the restart count. |
| D3 | Both checkpoints are cached in a `ModelCache` for the batch instead of being constructed inside `detect()`/`segment()` on every call. | Upstream would load both models 80 times for 40 images. Identical weights and outputs; performance only. |
| D4 | `lab2rgb` output is clipped into [0, 1]. | R's `convertColor` does not clip and R's `rgb()` then errors on out-of-gamut centroids. Clipping is the only way the supervised path returns at all. |
| D5 | Hex conversion spells out R's round-**half-up** rule `(int)(255*x + 0.5)`. | Python's `round()` is half-to-even and differs on exact .5 boundaries. Found by a failing test before any model ran. |
| D6 | Detection labels are matched case- and period-insensitively (`normalise_label`) instead of R's exact `%in%`. | Grounding DINO returns the matched *phrase*, and whether the terminating "." survives is transformers-version dependent. Unmatched labels are logged, never dropped silently. |
| D7 | Detection+segmentation runs as separate passes per prompt (ROI, exclusion, reference) rather than one multi-label `detect()` call filtered by label, as `R/image_analysis.R` does. | Directed by `PROMPT.md` stage D, and it is what makes a per-prompt hit rate measurable. Same models, same thresholds. |
| D8 | Inference wrapped in `torch.no_grad()`. | Upstream leaves autograd on. No effect on outputs. |
| D9 | Upstream's unused `reshaped_height, reshaped_width` locals in `segmentation.py` are omitted; `load_image`'s `http` branch is dropped. | Dead code; the probe only reads local files. |
| D10 | The per-pixel `hex_colors` field is computed only on request (`include_pixel_hex=True`). | One string per masked pixel (~10^5/image) for a field that is unused downstream and, upstream, always near-black (objection 5). |
| D11 | Stage G's five visuals are emitted as one 4-panel composite (`*_qa.png`) plus the transparent PNG cut-out, rather than five separate files. | Auditability: the arachnologist check needs mask, palette, histogram and recolour side by side. |
| D12 | A third, non-ROI `"a spider."` reference pass was added. | Required by `PROMPT.md`'s Argiope settings and it is what makes §3.1 measurable. It never contributes to the ROI. |
| D13 | This report is written to the run directory as specified **and** committed at the repository root. | `outputs/` is git-ignored, so the specified location alone would not survive a clone. |
| D14 | Environment: Python 3.11.15, `transformers` 4.57.6, torch 2.11.0+cu128, CUDA. | The original pins nothing (reticulate resolves at runtime), so an exact environment match is not available. |
| D15 | Installing `transformers` 4.x **downgraded `huggingface_hub` 1.26.1 → 0.36.2** in the shared `argiope` conda env (transformers 4.x requires `<1.0`). | Sanctioned by the probe spec, but it touches the shared environment; `pip install -e ".[segment]"` restores whatever the parent project pins. Nothing was added to `pyproject.toml`. |

---

## 6. Skeptic's note — what would have to be true for this to be wrong

- **The sample is small and uncontrolled.** 40 citizen-science photographs across three
  species, with uncontrolled lighting, pose, occlusion and background. That supports
  "this pipeline failed here", not "this pipeline fails". Confidence intervals on a 78%
  hit rate from n=40 are wide, and I have not computed them because the hit rate is not the
  quantity that decided the conclusion.
- **The single success is n=1.** "One acceptable mask in 40" is my own count, by eye, with
  no second observer and no ground truth (both out of scope). A different arachnologist
  might accept two, or none.
- **I did not tune the prompts**, by instruction — `"the abdomen of a spider."` comes from
  the parent project's recorded runs. Better phrasing may exist; this probe does not
  exclude it.
- **The checkpoints are the smallest in their families.** `grounding-dino-tiny` and
  `SlimSAM-uniform-77` (a heavily distilled SAM) are the paper's defaults, so using them is
  the reproduction; larger backbones might separate parts better.
- **What would make the negative result wrong:** if the near-identical mask areas were an
  artefact of my port rather than the method. Two things argue against that — the mechanism
  is legible in upstream's `refine_masks` before any output is examined, and the effect is
  consistent across three species, two thresholds and 40 images.
- **What this says about our own pipeline: nothing.** No comparison against YOLO + SAM 3
  was run, by instruction. Any statement that one is better than the other would be
  unsupported by anything measured here.

---

## 7. Recommendation

**Promote: the re-loadable JSON artefact (stage F).** This is the piece that earns its
place. `--from-json` re-derived the finished run's colour table **byte-identically** with
`transformers` never imported (asserted in-process, and in
`test_from_json_reruns_colour_without_loading_a_model`). Separating "run the models once"
from "analyse colour many times" is worth having in `src/argiope/` regardless of which
segmenter produces the mask: it makes palette parameters cheap to revisit and makes a run
auditable after the fact. Adopt the schema with mask PNGs (D1), not the literal 2-D lists.

**Promote, with the mask problem solved elsewhere: supervised colour assignment
(`--custom-colors`).** Assigning every pixel to the nearest reference colour in Lab yields
a fixed-length, directly comparable vector per specimen — a better fit for a taxonomic
ledger than free k-means, whose cluster identities are not comparable across images. Its
value is entirely downstream of a correct mask, so it is worth porting but it fixes nothing
about segmentation.

**Do not promote: part subtraction as specified.** `roi &= ~excluded` is sound only when
the two masks genuinely delineate different parts. Fed whole-object masks it deleted a
median 99.91% of the ROI and, at 0.3, destroyed the run's only correct palette. A
containment or overlap test — keep the exclusion only if it covers less than some fraction
of the ROI — would be the obvious guard, and writing it is exactly the "improvement" this
replication is forbidden to make. It belongs in a design note, not in this port.

**Do not promote: zero-shot GroundingDINO + SlimSAM for part-level segmentation**, nor
`refine_masks`' union of candidates (§3.1), nor RGB mean/median colour (§3.6). If the union
were replaced by a per-candidate choice, the granularity question would deserve re-asking —
but that is a different experiment, and it would not be a reproduction of this paper.

---

## 8. Reproducing this

```bash
pip install -r repro/segmentr/requirements.txt

# primary run (the paper's score threshold)
python repro/segmentr/repro_segmentr.py --images data/raw/gbif --n 40 --seed 42 \
    --score-threshold 0.5 --run-id n40-thr050

# threshold re-run
python repro/segmentr/repro_segmentr.py --images data/raw/gbif --n 40 --seed 42 \
    --score-threshold 0.3 --run-id n40-thr030

# colour re-analysis, no checkpoint loaded
python repro/segmentr/repro_segmentr.py --from-json repro/segmentr/outputs/n40-thr050

# the same 40 images already cropped for hand annotation
python repro/segmentr/repro_segmentr.py \
    --images-from data/interim/annotate/manifest.jsonl --n 40 --seed 42

pytest repro/segmentr        # 50 pure-function tests: no weights, no GPU, no network
```

Every parameter, seed, package version and the resolved image list are in
`run_config.json`; the run is reproducible from that file alone.
