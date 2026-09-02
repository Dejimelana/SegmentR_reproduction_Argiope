#!/usr/bin/env python
"""Reproduction of the SegmentR method (Boyko 2025) on *Argiope* photographs.

    Boyko, J. D. (2025). SegmentR: Deep learning for automated segmentation with an
    R interface. Ecological Informatics 90:103259. doi:10.1016/j.ecoinf.2025.103259
    Code: https://github.com/jboyko/SegmentR   (MIT -- see THIRD_PARTY_NOTICES.md)

This is a REPLICATION, not a product. Fidelity to the original beats output quality.
Where the original does something questionable -- colour means taken in RGB while the
clustering happens in Lab, an unseeded k-means, the union of SlimSAM's three candidate
masks, no validation at all -- it is reproduced here and the objection is written into
the run's REPORT.md. Nothing is "fixed".

The R layer of SegmentR is the paper's accessibility contribution and is not part of
what is reproduced; the method is ported to this one Python script. Every ported unit
carries a `# --- ported from <upstream file> ---` marker.

Pipeline (upstream stage letters follow repro/segmentr/PROMPT.md):
    A  GroundingDINO  zero-shot detection from a text prompt        -> boxes
    B  SlimSAM        box-prompted segmentation + refine_masks       -> masks
    C  score filter (default 0.5) + include/exclude labels + merge by label
    D  part subtraction: a second prompt's masks are removed (roi &= ~excluded)
    E  colour: k-means in CIELAB; mean/median in RGB (as upstream)
    F  re-loadable JSON artefact per image (--from-json re-runs E with no models)
    G  QA visuals: overlay, swatches, RGB histogram, recolour, transparent PNG
    H  batch over a folder

This script MUST NOT import `argiope`; it reads the parent project's images read-only
and resolves every path from --argiope-root.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

# transformers / matplotlib are imported lazily, inside the functions that need them, so
# that --from-json and the test-suite never pull in a model or a display backend.

# --------------------------------------------------------------------------------------
# Argiope-specific settings (defined here on purpose: the probe must not read or write
# the parent project's configs/default.yaml or data/taxonomy/features.yaml).
# --------------------------------------------------------------------------------------

# configs/default.yaml already records, from real runs, that "opisthosoma" is
# out-of-vocabulary for text-prompted models and that richer phrasing recalls the abdomen
# far better than "spider abdomen". These are not re-derived here.
DEFAULT_ROI_PROMPT = "the abdomen of a spider."
DEFAULT_WHOLE_PROMPT = "a spider."
DEFAULT_EXCLUDE_PROMPT = "the legs of a spider."

# Reference palette for --custom-colors argiope: yellow, black, silver/white, brown.
# A deliberately coarse, four-way vocabulary -- this is the supervised branch of the
# original's extract_colors(), not a colour standard. No PANTONE claim is made.
ARGIOPE_PALETTE: Dict[str, str] = {
    "yellow": "#FFD700",
    "black": "#000000",
    "silver": "#F0F0F0",
    "brown": "#8B5A2B",
}

SPECIES_DIRS = ("argiope_argentata", "argiope_aurantia", "argiope_bruennichi")


def log(msg: str) -> None:
    """Single logging point. Repo convention: never silently drop data."""
    print(msg, flush=True)


# ======================================================================================
# A/B support -- ported from inst/python/rseg/utils.py
# ======================================================================================

# --- ported from inst/python/rseg/utils.py (verbatim) ---
@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


# --- ported from inst/python/rseg/utils.py (verbatim) ---
@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.ndarray] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> "DetectionResult":
        return cls(
            score=detection_dict["score"],
            label=detection_dict["label"],
            box=BoundingBox(
                xmin=detection_dict["box"]["xmin"],
                ymin=detection_dict["box"]["ymin"],
                xmax=detection_dict["box"]["xmax"],
                ymax=detection_dict["box"]["ymax"],
            ),
        )


# --- ported from inst/python/rseg/utils.py ---
# (the upstream http branch is dropped: this probe only ever reads local files)
def load_image(image_str: str) -> Image.Image:
    return Image.open(image_str).convert("RGB")


# --- ported from inst/python/rseg/utils.py (verbatim) ---
def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    largest_contour = max(contours, key=cv2.contourArea)
    polygon = largest_contour.reshape(-1, 2).tolist()
    return polygon


# --- ported from inst/python/rseg/utils.py (verbatim) ---
def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], color=(255,))
    return mask


# --- ported from inst/python/rseg/utils.py (verbatim) ---
def get_boxes(results: List[DetectionResult]) -> List[List[List[float]]]:
    """Note the extra nesting level: SlimSAM's processor wants [[[x0,y0,x1,y1], ...]]."""
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)
    return [boxes]


# --- ported from inst/python/rseg/utils.py (verbatim) ---
def refine_masks(masks: "torch.BoolTensor", polygon_refinement: bool = False) -> List[np.ndarray]:
    """Upstream mask post-processing, reproduced exactly.

    OBJECTION (reproduced, not fixed): `masks` arrives shaped
    (n_boxes, n_candidates, H, W) -- SlimSAM emits three candidate masks per box to
    resolve the whole/part/subpart ambiguity. Averaging over that axis and thresholding
    at `> 0` takes the UNION of all three. For a part prompt such as "the abdomen of a
    spider" the three candidates are typically {abdomen, abdomen+cephalothorax, whole
    animal}, so the union is the whole animal: the step that exists to pick a part
    systematically discards the part. See REPORT.md.
    """
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks


# ======================================================================================
# Stage A -- detection (GroundingDINO), ported from inst/python/rseg/detection.py
# ======================================================================================

class ModelCache:
    """Holds the two checkpoints for the length of a batch run.

    DEVIATION (logged): upstream `detect()` and `segment()` construct their model inside
    the function, so a 40-image batch would load both checkpoints 80 times. The cache is
    a pure performance change -- identical models, identical weights, identical outputs --
    and it is the only reason a batch finishes in minutes rather than hours. Constructed
    by the caller and passed in, so there are no module-level globals; `--from-json`
    never constructs one, which is what makes the reload provably model-free.
    """

    def __init__(self, detector_id: str, segmenter_id: str) -> None:
        self.detector_id = detector_id
        self.segmenter_id = segmenter_id
        self._detector = None
        self._segmenter = None
        self._processor = None

    # upstream: device = "cuda" if torch.cuda.is_available() else "cpu"   (detection.py)
    @staticmethod
    def device() -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def detector(self):
        if self._detector is None:
            from transformers import pipeline

            log(f"  loading detector {self.detector_id} on {self.device()}")
            self._detector = pipeline(
                model=self.detector_id,
                task="zero-shot-object-detection",
                device=self.device(),
            )
        return self._detector

    def segmenter(self):
        if self._segmenter is None:
            from transformers import AutoModelForMaskGeneration, AutoProcessor

            log(f"  loading segmenter {self.segmenter_id} on {self.device()}")
            self._segmenter = AutoModelForMaskGeneration.from_pretrained(
                self.segmenter_id
            ).to(self.device())
            self._processor = AutoProcessor.from_pretrained(self.segmenter_id)
        return self._segmenter, self._processor


# --- ported from inst/python/rseg/detection.py ---
def detect(
    image: Image.Image,
    labels: List[str],
    threshold: float = 0.1,
    detector_id: str = "IDEA-Research/grounding-dino-tiny",
    models: Optional[ModelCache] = None,
) -> List[DetectionResult]:
    """Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion."""
    if models is None:  # upstream behaviour: build the pipeline per call
        models = ModelCache(detector_id, "")
    object_detector = models.detector()

    # upstream, verbatim -- Grounding DINO requires each phrase to be terminated
    labels = [label if label.endswith(".") else label + "." for label in labels]

    results = object_detector(image, candidate_labels=labels, threshold=threshold)
    results = [DetectionResult.from_dict(result) for result in results]

    return results


# ======================================================================================
# Stage B -- segmentation (SlimSAM), ported from inst/python/rseg/segmentation.py
# ======================================================================================

# --- ported from inst/python/rseg/segmentation.py ---
def segment(
    image: Image.Image,
    detection_results: List[DetectionResult],
    polygon_refinement: bool = False,
    segmenter_id: str = "Zigeng/SlimSAM-uniform-77",
    models: Optional[ModelCache] = None,
) -> List[DetectionResult]:
    if models is None:  # upstream behaviour: build the model per call
        models = ModelCache("", segmenter_id)
    segmentator, processor = models.segmenter()
    device = models.device()

    boxes = get_boxes(detection_results)

    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt")
    inputs = {k: v.to(torch.float32) if torch.is_tensor(v) else v for k, v in inputs.items()}
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    with torch.no_grad():  # DEVIATION (logged): inference-only; upstream keeps autograd on
        outputs = segmentator(**inputs)

    original_sizes = [(int(s[0].item()), int(s[1].item())) for s in inputs["original_sizes"].cpu()]
    reshaped_sizes = [
        (int(s[0].item()), int(s[1].item())) for s in inputs["reshaped_input_sizes"].cpu()
    ]

    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        original_sizes=original_sizes,
        reshaped_input_sizes=reshaped_sizes,
    )[0]

    masks = refine_masks(masks, polygon_refinement)

    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask

    return detection_results


# ======================================================================================
# Stage C/D -- filtering, merging, part subtraction. Ported from R/image_analysis.R.
# Pure array code: no torch, no models, no I/O. This is what tests/ exercises.
# ======================================================================================

def normalise_label(label: str) -> str:
    """Compare Grounding DINO's returned phrase with the prompt that asked for it.

    Not upstream. R compares labels with `%in%`, i.e. exact string equality, because in
    the paper's examples the label is a single word ("flower", "bee"). Grounding DINO
    returns the matched *phrase*, and whether the terminating "." survives depends on the
    transformers version, so the probe compares case- and period-insensitively and logs
    anything it cannot match. Logged as a deviation.
    """
    return label.strip().rstrip(".").strip().lower()


# --- ported from R/image_analysis.R :: combine_masks ---
def combine_masks(
    masks: Sequence[np.ndarray], scores: Sequence[float], threshold: float = 0.5
) -> np.ndarray:
    """Combine every mask whose score meets the threshold (logical OR).

    Upstream calls stop() when nothing survives; the ValueError here is that stop(), and
    the batch driver turns it into a logged, counted skip rather than a crash.
    """
    selected = [m for m, s in zip(masks, scores) if s >= threshold]
    if len(selected) == 0:
        raise ValueError("No masks meet the score threshold")
    combined = np.zeros_like(np.asarray(selected[0]), dtype=bool)
    for m in selected:
        combined |= np.asarray(m) > 0
    return combined


# --- ported from R/image_analysis.R :: exclude_masks ---
def exclude_masks(combined_mask: np.ndarray, masks_to_exclude: Sequence[np.ndarray]) -> np.ndarray:
    """Stage D, part subtraction: roi &= ~excluded (upstream: combined[mask > 0] <- FALSE)."""
    out = np.asarray(combined_mask).astype(bool).copy()
    for mask in masks_to_exclude:
        out[np.asarray(mask) > 0] = False
    return out


def merge_masks_by_label(
    detections: Sequence[DetectionResult], score_threshold: float
) -> Dict[str, Dict]:
    """Stage C: OR together every surviving detection that carries the same label.

    Upstream ORs all included labels into a single mask in one step (combine_masks over
    `which(labels %in% include_labels)`); splitting the OR per label first is equivalent
    for the union, and is what lets colors.csv carry a `label` column and the report a
    per-prompt hit rate. `score` is the max over the detections merged into that label.
    """
    merged: Dict[str, Dict] = {}
    for det in detections:
        if det.mask is None or det.score < score_threshold:
            continue
        key = normalise_label(det.label)
        m = np.asarray(det.mask) > 0
        if key not in merged:
            merged[key] = {"mask": m.copy(), "score": float(det.score), "n_detections": 1}
        else:
            merged[key]["mask"] |= m
            merged[key]["score"] = max(merged[key]["score"], float(det.score))
            merged[key]["n_detections"] += 1
    return merged


# ======================================================================================
# Stage E -- colour in CIELAB. Ported from R/image_analysis.R :: extract_colors.
# ======================================================================================

def hex_to_rgb01(hex_code: str) -> np.ndarray:
    """R's col2rgb(hex) / 255."""
    h = hex_code.lstrip("#")
    return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0


def rgb01_to_hex(rgb: Sequence[float]) -> str:
    """R's rgb(r, g, b) with the default maxColorValue = 1.

    R's C implementation scales with (unsigned int)(255 * x + 0.5), i.e. round-half-UP.
    Python's round() is round-half-to-EVEN, which differs on exact .5 boundaries, so the
    R rule is spelled out here rather than inherited from the language.
    """
    vals = [int(np.floor(float(np.clip(v, 0.0, 1.0)) * 255 + 0.5)) for v in rgb]
    return "#{:02X}{:02X}{:02X}".format(*vals)


def rgb01_to_lab(rgb: np.ndarray) -> np.ndarray:
    """R's convertColor(from = "sRGB", to = "Lab"); skimage's rgb2lab is D65 / 2 deg."""
    from skimage.color import rgb2lab

    arr = np.asarray(rgb, dtype=float).reshape(-1, 1, 3)
    return rgb2lab(arr).reshape(-1, 3)


def lab_to_rgb01(lab: np.ndarray) -> np.ndarray:
    """R's convertColor(from = "Lab", to = "sRGB").

    DEVIATION (logged): skimage's lab2rgb clips out-of-gamut results into [0, 1];
    R's convertColor does not, and R's rgb() then errors. Clipping is the only way the
    supervised path returns at all for centroids near the gamut boundary.
    """
    from skimage.color import lab2rgb

    arr = np.asarray(lab, dtype=float).reshape(-1, 1, 3)
    return np.clip(lab2rgb(arr).reshape(-1, 3), 0.0, 1.0)


def upstream_pixel_hex(masked_pixels: np.ndarray) -> List[str]:
    """Reproduction of upstream's per-pixel `hex_colors` field, bug included.

    R/image_analysis.R returns
        hex_colors = rgb(masked_pixels[, 1] / 255, masked_pixels[, 2] / 255, ...)
    but `masked_pixels` is already in [0, 1] -- process_masks_and_extract_colors divides
    the image by 255 before calling. Dividing again drives every channel to <= 1/255, so
    every pixel comes back near-black. Reproduced here, and tested, rather than
    corrected; the field is unused downstream so it is off by default (it is one string
    per masked pixel, ~10^5 per image).
    """
    scaled = np.asarray(masked_pixels, dtype=float) / 255.0
    return [rgb01_to_hex(px) for px in scaled]


# --- ported from R/image_analysis.R :: extract_colors ---
def extract_colors(
    image: np.ndarray,
    mask: np.ndarray,
    n_colors: int = 5,
    custom_colors: Optional[Sequence[str]] = None,
    seed: int = 42,
    include_pixel_hex: bool = False,
) -> Dict:
    """Dominant colours of the masked region.

    `image` is float sRGB in [0, 1], (H, W, 3). `mask` is boolean, (H, W).

    Faithful to the original in three respects that matter:

    1. clustering happens in CIELAB, on masked pixels only;
    2. the *assignment* step runs in BOTH branches. Upstream computes k-means centres and
       then falls through to the nearest-centroid apply(...) loop, overwriting
       km_result$size with tabulate(cluster_assignments). So k-means supplies centres
       only; sizes always come from a Euclidean-in-Lab nearest-centroid assignment;
    3. OBJECTION (reproduced, not fixed): `mean_color` and `median_color` are computed in
       **RGB** over the masked pixels while everything else is perceptual. A per-channel
       RGB median is not a colour any pixel need have, and an RGB mean of a yellow-and-
       black abdomen lands on a muddy olive that is in neither cluster. See REPORT.md.
    """
    from sklearn.cluster import KMeans

    image = np.asarray(image, dtype=float)
    mask_bool = np.asarray(mask).astype(bool)
    masked_pixels = image[mask_bool]  # (N, 3); upstream: matrix(image[mask], ncol = 3)
    if masked_pixels.size == 0:
        raise ValueError("Mask selects no pixels")

    lab_colors = rgb01_to_lab(masked_pixels)

    if custom_colors is None:
        n_pixels = lab_colors.shape[0]
        k = int(min(n_colors, n_pixels))  # KMeans needs n_clusters <= n_samples
        # DEVIATION (logged): R's kmeans() is Hartigan-Wong, nstart = 1, unseeded.
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(lab_colors)
        centers_lab = km.cluster_centers_
        n_out = k
    else:
        custom_rgb = np.vstack([hex_to_rgb01(c) for c in custom_colors])
        centers_lab = rgb01_to_lab(custom_rgb)
        n_out = len(custom_colors)

    # Upstream runs this for BOTH branches -- see docstring point 2.
    d2 = ((lab_colors[:, None, :] - centers_lab[None, :, :]) ** 2).sum(axis=2)
    cluster_assignments = np.argmin(d2, axis=1)
    cluster_sizes = np.bincount(cluster_assignments, minlength=n_out)[:n_out]

    centers_rgb = lab_to_rgb01(centers_lab)
    hex_codes = [rgb01_to_hex(c) for c in centers_rgb]

    dominant_color_info = [
        {
            "lab_l": float(centers_lab[i, 0]),
            "lab_a": float(centers_lab[i, 1]),
            "lab_b": float(centers_lab[i, 2]),
            "hex_color": hex_codes[i],
            "cluster_size": int(cluster_sizes[i]),
        }
        for i in range(n_out)
    ]

    mean_color = masked_pixels.mean(axis=0)             # upstream: colMeans, in RGB
    median_color = np.median(masked_pixels, axis=0)     # upstream: apply(., 2, median)

    out = {
        "masked_pixels": masked_pixels,
        "dominant_color_info": dominant_color_info,
        "mean_color": rgb01_to_hex(mean_color),
        "median_color": rgb01_to_hex(median_color),
        "km_result": {
            "cluster": cluster_assignments,
            "centers": centers_lab,
            "size": cluster_sizes,
        },
    }
    if include_pixel_hex:
        out["hex_colors"] = upstream_pixel_hex(masked_pixels)
    return out


# --- ported from R/image_analysis.R :: process_masks_and_extract_colors ---
def process_masks_and_extract_colors(
    image: np.ndarray,
    detections: Sequence[DetectionResult],
    include_labels: Sequence[str],
    exclude_labels: Optional[Sequence[str]] = None,
    score_threshold: float = 0.5,
    n_colors: int = 5,
    custom_colors: Optional[Sequence[str]] = None,
    seed: int = 42,
) -> Dict:
    """Stages C + D + E for one image, in the original's order.

    Returns the final ROI mask, the per-label merged masks, and the colour info for each
    included label. Raises ValueError (upstream's stop()) if nothing survives, so the
    batch driver can log and count the skip.
    """
    image = np.asarray(image, dtype=float)
    if image.max() > 1:  # upstream: if (max(image) > 1) image <- image / 255
        image = image / 255.0

    include = {normalise_label(x) for x in include_labels}
    exclude = {normalise_label(x) for x in (exclude_labels or [])}

    merged = merge_masks_by_label(detections, score_threshold)
    unmatched = sorted(set(merged) - include - exclude)
    if unmatched:
        log("    note: detection labels matched no prompt, ignored: " + repr(unmatched))

    include_masks = [v["mask"] for k, v in merged.items() if k in include]
    include_scores = [v["score"] for k, v in merged.items() if k in include]
    combined_mask = combine_masks(include_masks, include_scores, threshold=score_threshold)

    if exclude:
        exclude_list = [v["mask"] for k, v in merged.items() if k in exclude]
        final_mask = exclude_masks(combined_mask, exclude_list)
    else:
        exclude_list = []
        final_mask = combined_mask

    if not final_mask.any():
        raise ValueError("Mask is empty after part subtraction")

    per_label = {}
    for key, entry in merged.items():
        if key not in include:
            continue
        label_mask = exclude_masks(entry["mask"], exclude_list) if exclude else entry["mask"]
        if not label_mask.any():
            log("    note: label " + repr(key) + " emptied by part subtraction")
            continue
        per_label[key] = {
            "score": entry["score"],
            "n_detections": entry["n_detections"],
            "mask": label_mask,
            "color_info": extract_colors(
                image, label_mask, n_colors=n_colors, custom_colors=custom_colors, seed=seed
            ),
        }

    return {
        "final_mask": final_mask,
        "image": image,
        "merged": merged,
        "excluded_masks": exclude_list,
        "per_label": per_label,
        "color_info": extract_colors(
            image, final_mask, n_colors=n_colors, custom_colors=custom_colors, seed=seed
        ),
    }


# ======================================================================================
# Stage F -- the re-loadable JSON artefact. Schema ported from inst/python/main.py.
# ======================================================================================

def detections_to_dicts(
    detections: Sequence[DetectionResult],
    img_height: int,
    img_width: int,
    mask_names: Optional[Sequence[Optional[str]]] = None,
    faithful: bool = False,
) -> List[Dict]:
    """Upstream schema: a bare list of {label, score, box, mask}.

    Upstream (main.py):
        "mask": d.mask[:img_height, :img_width].tolist() if d.mask is not None else None

    DEVIATION (logged, and reversible with --faithful-json): by default `mask` holds the
    filename of a PNG written beside the JSON instead of a 2-D integer list. At ~1 MP a
    single serialised mask is tens of MB of text; 40 images would be several GB.
    """
    out = []
    for i, d in enumerate(detections):
        if d.mask is None:
            mask_field = None
        elif faithful:
            mask_field = np.asarray(d.mask)[:img_height, :img_width].astype(int).tolist()
        else:
            mask_field = mask_names[i] if mask_names else None
        out.append(
            {
                "label": d.label,
                "score": float(d.score),
                "box": {
                    "xmin": int(d.box.xmin),
                    "ymin": int(d.box.ymin),
                    "xmax": int(d.box.xmax),
                    "ymax": int(d.box.ymax),
                },
                "mask": mask_field,
            }
        )
    return out


def dicts_to_detections(records: Sequence[Dict], base_dir: Path) -> List[DetectionResult]:
    """Inverse of detections_to_dicts: rebuild DetectionResults with no model involved."""
    out = []
    for rec in records:
        det = DetectionResult.from_dict(rec)
        m = rec.get("mask")
        if m is None:
            det.mask = None
        elif isinstance(m, str):
            arr = cv2.imread(str(Path(base_dir) / m), cv2.IMREAD_GRAYSCALE)
            if arr is None:
                raise FileNotFoundError("mask PNG missing: " + str(Path(base_dir) / m))
            det.mask = (arr > 0).astype(np.uint8)
        else:
            det.mask = np.asarray(m, dtype=np.uint8)
        out.append(det)
    return out


def write_image_json(
    json_path: Path,
    detections: Sequence[DetectionResult],
    img_height: int,
    img_width: int,
    faithful: bool,
) -> None:
    mask_names: List[Optional[str]] = []
    if not faithful:
        mask_dir = json_path.parent / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        for i, d in enumerate(detections):
            if d.mask is None:
                mask_names.append(None)
                continue
            name = "masks/{}_{:03d}.png".format(json_path.stem, i)
            arr = (np.asarray(d.mask)[:img_height, :img_width] > 0).astype(np.uint8) * 255
            cv2.imwrite(str(json_path.parent / name), arr)
            mask_names.append(name)
    payload = detections_to_dicts(detections, img_height, img_width, mask_names, faithful)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


# ======================================================================================
# Stage G -- QA visuals. annotate/plot_detections ported from rseg/visualization.py,
# export_transparent_png from R/export.R.
# ======================================================================================

# --- ported from inst/python/rseg/visualization.py :: annotate ---
def annotate(image, detection_results: Sequence[DetectionResult]) -> np.ndarray:
    image_cv2 = np.array(image) if isinstance(image, Image.Image) else np.asarray(image)
    image_cv2 = cv2.cvtColor(image_cv2.astype(np.uint8), cv2.COLOR_RGB2BGR).copy()

    for detection in detection_results:
        label = detection.label
        score = detection.score
        box = detection.box
        mask = detection.mask

        color = np.random.randint(0, 256, size=3)

        cv2.rectangle(
            image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax), color.tolist(), 2
        )
        cv2.putText(
            image_cv2,
            "{}: {:.2f}".format(label, score),
            (box.xmin, box.ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color.tolist(),
            2,
        )

        if mask is not None:
            mask_uint8 = (np.asarray(mask) * 255).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(image_cv2, contours, -1, color.tolist(), 2)

    return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)


def recolour_by_cluster(
    image01: np.ndarray, mask: np.ndarray, color_info: Dict
) -> np.ndarray:
    """Every masked pixel replaced by its cluster centroid (stage G)."""
    centers_rgb = lab_to_rgb01(color_info["km_result"]["centers"])
    assignments = color_info["km_result"]["cluster"]
    out = np.ones_like(np.asarray(image01, dtype=float))
    mask_bool = np.asarray(mask).astype(bool)
    out[mask_bool] = centers_rgb[assignments]
    return out


# --- ported from R/export.R :: export_transparent_png ---
def export_transparent_png(
    image01: np.ndarray,
    masks_by_label: Dict[str, np.ndarray],
    output_path: Path,
    prefix: str,
    crop: bool = True,
    remove_overlap: bool = True,
) -> List[Path]:
    """RGBA cut-out per label: alpha = mask, optionally cropped to the mask bounding box.

    remove_overlap reproduces upstream's behaviour of erasing, from each label's mask,
    every region claimed by a mask carrying a *different* label.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    rgb = (np.clip(np.asarray(image01, dtype=float), 0, 1) * 255).astype(np.uint8)

    for idx, (label, mask) in enumerate(sorted(masks_by_label.items()), start=1):
        m = np.asarray(mask).astype(bool)
        if remove_overlap:
            for other_label, other in masks_by_label.items():
                if other_label != label:
                    m = m & ~np.asarray(other).astype(bool)
        if not m.any():
            log("    note: transparent export skipped for " + repr(label) + " (empty mask)")
            continue

        alpha = (m * 255).astype(np.uint8)
        rgba = np.dstack([rgb * m[..., None], alpha])

        if crop:
            rows = np.any(m, axis=1)
            cols = np.any(m, axis=0)
            r0, r1 = np.where(rows)[0][[0, -1]]
            c0, c1 = np.where(cols)[0][[0, -1]]
            rgba = rgba[r0 : r1 + 1, c0 : c1 + 1]

        safe = "".join(ch if ch.isalnum() else "_" for ch in label).strip("_")
        out_file = output_path / "{}_{:03d}_{}.png".format(prefix, idx, safe)
        Image.fromarray(rgba, mode="RGBA").save(out_file)
        written.append(out_file)
    return written


def write_qa_figure(
    out_png: Path,
    image01: np.ndarray,
    detections: Sequence[DetectionResult],
    final_mask: np.ndarray,
    color_info: Dict,
    title: str,
) -> None:
    """Four-panel composite: overlay, swatch strip, RGB histogram, recoloured mask."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgb_u8 = (np.clip(image01, 0, 1) * 255).astype(np.uint8)
    overlay = annotate(rgb_u8, detections)
    masked_pixels = color_info["masked_pixels"]
    info = color_info["dominant_color_info"]
    total = max(1, sum(c["cluster_size"] for c in info))

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(title, fontsize=10)

    axes[0][0].imshow(overlay)
    axes[0][0].set_title("detections + mask contours", fontsize=9)
    axes[0][0].axis("off")

    ax = axes[0][1]
    order = np.argsort([-c["cluster_size"] for c in info])
    for j, i in enumerate(order):
        c = info[int(i)]
        ax.add_patch(plt.Rectangle((j, 0), 1, 1, color=c["hex_color"]))
        ax.text(
            j + 0.5,
            -0.12,
            "{}\n{:.1f}%\nL{:.0f} a{:.0f} b{:.0f}".format(
                c["hex_color"],
                100.0 * c["cluster_size"] / total,
                c["lab_l"],
                c["lab_a"],
                c["lab_b"],
            ),
            ha="center",
            va="top",
            fontsize=7,
        )
    ax.set_xlim(0, max(1, len(info)))
    ax.set_ylim(-0.7, 1.05)
    ax.axis("off")
    ax.set_title(
        "dominant colours (k-means in CIELAB)   mean {}  median {}".format(
            color_info["mean_color"], color_info["median_color"]
        ),
        fontsize=9,
    )

    ax = axes[1][0]
    for ch, colour in enumerate(("red", "green", "blue")):
        ax.hist(
            masked_pixels[:, ch] * 255,
            bins=64,
            range=(0, 255),
            histtype="step",
            color=colour,
        )
    ax.set_title("RGB histogram of masked pixels (n={})".format(len(masked_pixels)), fontsize=9)
    ax.set_xlabel("channel value", fontsize=8)

    axes[1][1].imshow(recolour_by_cluster(image01, final_mask, color_info))
    axes[1][1].set_title("mask recoloured by cluster centroid", fontsize=9)
    axes[1][1].axis("off")

    fig.tight_layout()
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ======================================================================================
# Stage H -- image sampling and the batch driver.
# ======================================================================================

def sample_images(
    images_root: Path, n: int, seed: int
) -> Tuple[List[Tuple[Path, str]], List[str]]:
    """Seeded, species-stratified sample of data/raw/gbif/<taxon>/*.jpg."""
    notes: List[str] = []
    species_dirs = sorted(p for p in Path(images_root).iterdir() if p.is_dir())
    if not species_dirs:
        raise FileNotFoundError("no species subdirectories under " + str(images_root))

    per_species = {}
    for d in species_dirs:
        files = sorted(p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        per_species[d.name] = files
        notes.append("{}: {} images available".format(d.name, len(files)))

    k = len(species_dirs)
    quota = {d.name: n // k + (1 if i < n % k else 0) for i, d in enumerate(species_dirs)}

    chosen: List[Tuple[Path, str]] = []
    for d in species_dirs:
        files = per_species[d.name]
        want = quota[d.name]
        if want > len(files):
            notes.append(
                "{}: asked for {} but only {} available".format(d.name, want, len(files))
            )
            want = len(files)
        rng = random.Random("{}:{}".format(seed, d.name))
        picked = rng.sample(files, want)
        chosen.extend((p, d.name) for p in sorted(picked))
    return chosen, notes


def images_from_manifest(
    manifest: Path, argiope_root: Path, n: Optional[int]
) -> Tuple[List[Tuple[Path, str]], List[str]]:
    """Read data/interim/annotate/manifest.jsonl (read-only) and use its source images."""
    notes: List[str] = []
    chosen: List[Tuple[Path, str]] = []
    with open(manifest, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rel = str(rec.get("source_image", "")).replace("\\", "/")
            if not rel:
                notes.append("manifest line {}: no source_image, skipped".format(line_no))
                continue
            path = (Path(argiope_root) / rel).resolve()
            if not path.exists():
                notes.append("manifest line {}: missing on disk, skipped ({})".format(line_no, rel))
                continue
            chosen.append((path, path.parent.name))
    if n is not None and n < len(chosen):
        notes.append("manifest truncated to the first {} of {} entries".format(n, len(chosen)))
        chosen = chosen[:n]
    return chosen, notes


COLORS_CSV_FIELDS = [
    "image",
    "species",
    "role",
    "label",
    "score",
    "n_detections",
    "cluster",
    "lab_l",
    "lab_a",
    "lab_b",
    "hex_color",
    "cluster_size",
    "cluster_frac",
    "mask_pixels",
    "mean_color",
    "median_color",
]


def colour_rows(
    image_name: str, species: str, role: str, label: str, score: float,
    n_detections: int, color_info: Dict, mask_pixels: int,
) -> List[Dict]:
    total = max(1, sum(c["cluster_size"] for c in color_info["dominant_color_info"]))
    rows = []
    for i, c in enumerate(color_info["dominant_color_info"]):
        rows.append(
            {
                "image": image_name,
                "species": species,
                "role": role,
                "label": label,
                "score": round(float(score), 4),
                "n_detections": n_detections,
                "cluster": i,
                "lab_l": round(c["lab_l"], 3),
                "lab_a": round(c["lab_a"], 3),
                "lab_b": round(c["lab_b"], 3),
                "hex_color": c["hex_color"],
                "cluster_size": c["cluster_size"],
                "cluster_frac": round(c["cluster_size"] / total, 5),
                "mask_pixels": mask_pixels,
                "mean_color": color_info["mean_color"],
                "median_color": color_info["median_color"],
            }
        )
    return rows


# ======================================================================================
# Run configuration and the batch driver (upstream's loop lives in R; here it is stage H).
# ======================================================================================

@dataclass
class RunConfig:
    """Everything needed to reproduce a run. Serialised verbatim to run_config.json."""

    run_id: str
    argiope_root: str
    images: str
    images_from: Optional[str]
    n: int
    seed: int
    roi_prompt: str
    exclude_prompts: List[str]
    reference_prompts: List[str]
    include_labels: List[str]
    exclude_labels: List[str]
    detector_id: str
    box_threshold: float
    segmenter_id: str
    score_threshold: float
    n_colors: int
    custom_colors: Optional[List[str]]
    polygon_refinement: bool
    faithful_json: bool
    qa: bool
    device: str = ""
    started_utc: str = ""
    versions: Dict[str, str] = field(default_factory=dict)
    image_list: List[Dict[str, str]] = field(default_factory=list)
    sampling_notes: List[str] = field(default_factory=list)


def package_versions() -> Dict[str, str]:
    """Recorded without importing the packages (importlib.metadata reads metadata only)."""
    from importlib.metadata import PackageNotFoundError, version

    out = {"python": sys.version.split()[0]}
    for pkg in (
        "torch", "transformers", "numpy", "scikit-learn", "scikit-image",
        "opencv-python", "matplotlib", "pillow", "timm", "accelerate",
    ):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def filter_by_labels(
    detections: Sequence[DetectionResult], keep: Sequence[str]
) -> List[DetectionResult]:
    wanted = {normalise_label(k) for k in keep}
    return [d for d in detections if normalise_label(d.label) in wanted]


def write_csv(path: Path, rows: Sequence[Dict], fields: Sequence[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_batch(cfg: RunConfig, out_dir: Path) -> Dict:
    """Detection -> segmentation -> filter -> subtract -> colour -> artefacts, per image."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "json").mkdir(exist_ok=True)
    if cfg.qa:
        (out_dir / "qa").mkdir(exist_ok=True)
        (out_dir / "cutouts").mkdir(exist_ok=True)

    models = ModelCache(cfg.detector_id, cfg.segmenter_id)
    passes: List[Tuple[str, str]] = [("roi", cfg.roi_prompt)]
    passes += [("exclude", p) for p in cfg.exclude_prompts]
    passes += [("reference", p) for p in cfg.reference_prompts]

    colour_rows_all: List[Dict] = []
    hit_rows: List[Dict] = []
    skipped: List[Dict] = []
    processed = 0

    total = len(cfg.image_list)
    for idx, entry in enumerate(cfg.image_list, start=1):
        img_path = Path(entry["path"])
        species = entry["species"]
        name = img_path.name
        log("Processing image {} of {}: {}/{}".format(idx, total, species, name))

        try:
            image = load_image(str(img_path))
        except Exception as exc:  # unreadable file -- logged, counted, never silent
            log("  SKIP unreadable image: {}".format(exc))
            skipped.append(
                {"image": name, "species": species, "stage": "load", "reason": repr(str(exc))}
            )
            continue

        width, height = image.size
        image01 = np.asarray(image, dtype=float) / 255.0

        all_dets: List[DetectionResult] = []
        for role, prompt in passes:
            dets = detect(
                image=image,
                labels=[prompt],
                threshold=cfg.box_threshold,
                detector_id=cfg.detector_id,
                models=models,
            )
            n_dets = len(dets)
            if dets:
                dets = segment(
                    image=image,
                    detection_results=dets,
                    polygon_refinement=cfg.polygon_refinement,
                    segmenter_id=cfg.segmenter_id,
                    models=models,
                )
            above = [d for d in dets if d.score >= cfg.score_threshold]
            mask_px = int(
                sum(int((np.asarray(d.mask) > 0).sum()) for d in above if d.mask is not None)
            )
            hit_rows.append(
                {
                    "image": name,
                    "species": species,
                    "role": role,
                    "prompt": prompt,
                    "n_detections": n_dets,
                    "n_above_score": len(above),
                    "max_score": round(max([d.score for d in dets], default=0.0), 4),
                    "mask_pixels": mask_px,
                    "mask_frac": round(mask_px / float(width * height), 5),
                }
            )
            log(
                "  {:<9} '{}' -> {} boxes, {} above score {}".format(
                    role, prompt, n_dets, len(above), cfg.score_threshold
                )
            )
            all_dets.extend(dets)

        json_path = out_dir / "json" / "{}__{}.json".format(species, img_path.stem)
        write_image_json(json_path, all_dets, height, width, cfg.faithful_json)

        roi_dets = filter_by_labels(all_dets, cfg.include_labels + cfg.exclude_labels)
        try:
            result = process_masks_and_extract_colors(
                image=image01,
                detections=roi_dets,
                include_labels=cfg.include_labels,
                exclude_labels=cfg.exclude_labels,
                score_threshold=cfg.score_threshold,
                n_colors=cfg.n_colors,
                custom_colors=cfg.custom_colors,
                seed=cfg.seed,
            )
        except ValueError as exc:
            log("  SKIP {}".format(exc))
            skipped.append(
                {"image": name, "species": species, "stage": "mask", "reason": str(exc)}
            )
            continue

        final_mask = result["final_mask"]
        roi_px = int(final_mask.sum())
        colour_rows_all.extend(
            colour_rows(
                name, species, "final_roi", "+".join(cfg.include_labels),
                max([v["score"] for v in result["per_label"].values()], default=0.0),
                sum(v["n_detections"] for v in result["per_label"].values()),
                result["color_info"], roi_px,
            )
        )
        for label, entry_l in result["per_label"].items():
            colour_rows_all.extend(
                colour_rows(
                    name, species, "label", label, entry_l["score"], entry_l["n_detections"],
                    entry_l["color_info"], int(np.asarray(entry_l["mask"]).sum()),
                )
            )

        log(
            "  ROI {} px ({:.1%} of frame) -> {} clusters, mean {}".format(
                roi_px, roi_px / float(width * height),
                len(result["color_info"]["dominant_color_info"]),
                result["color_info"]["mean_color"],
            )
        )

        if cfg.qa:
            try:
                write_qa_figure(
                    out_dir / "qa" / "{}__{}_qa.png".format(species, img_path.stem),
                    image01, all_dets, final_mask, result["color_info"],
                    "{} / {}   roi={:.1%} of frame".format(
                        species, name, roi_px / float(width * height)
                    ),
                )
                export_transparent_png(
                    image01,
                    {"roi": final_mask},
                    out_dir / "cutouts",
                    "{}__{}".format(species, img_path.stem),
                    crop=True,
                    remove_overlap=True,
                )
            except Exception as exc:
                log("  note: QA rendering failed: {}".format(exc))
                skipped.append(
                    {"image": name, "species": species, "stage": "qa", "reason": repr(str(exc))}
                )

        processed += 1

    write_csv(out_dir / "colors.csv", colour_rows_all, COLORS_CSV_FIELDS)
    write_csv(
        out_dir / "prompt_hits.csv", hit_rows,
        ["image", "species", "role", "prompt", "n_detections", "n_above_score",
         "max_score", "mask_pixels", "mask_frac"],
    )
    write_csv(out_dir / "skipped.csv", skipped, ["image", "species", "stage", "reason"])

    summary = {
        "run_id": cfg.run_id,
        "n_images": total,
        "processed": processed,
        "skipped": len(skipped),
        "colour_rows": len(colour_rows_all),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    log("\n{} processed, {} skipped -> {}".format(processed, len(skipped), out_dir))
    return summary


def run_from_json(
    run_dir: Path,
    n_colors: Optional[int],
    custom_colors: Optional[List[str]],
    seed: Optional[int],
) -> Dict:
    """Stage F's payoff: colour re-analysis with neither checkpoint loaded.

    Reads the finished run's per-image JSON + mask PNGs and re-runs stages C, D and E.
    Asserts at the end that `transformers` was never imported.
    """
    run_dir = Path(run_dir)
    with open(run_dir / "run_config.json", "r", encoding="utf-8") as fh:
        saved = json.load(fh)

    n_colors = saved["n_colors"] if n_colors is None else n_colors
    seed = saved["seed"] if seed is None else seed
    if custom_colors is None:
        custom_colors = saved.get("custom_colors")

    rows: List[Dict] = []
    skipped: List[Dict] = []
    entries = saved["image_list"]
    for idx, entry in enumerate(entries, start=1):
        img_path = Path(entry["path"])
        species = entry["species"]
        name = img_path.name
        json_path = run_dir / "json" / "{}__{}.json".format(species, Path(name).stem)
        log("Processing image {} of {}: {} (from JSON)".format(idx, len(entries), name))
        if not json_path.exists():
            log("  SKIP no JSON artefact for this image")
            skipped.append(
                {"image": name, "species": species, "stage": "from_json", "reason": "no JSON"}
            )
            continue
        with open(json_path, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        try:
            dets = dicts_to_detections(records, json_path.parent)
            image01 = np.asarray(load_image(str(img_path)), dtype=float) / 255.0
            dets = filter_by_labels(dets, saved["include_labels"] + saved["exclude_labels"])
            result = process_masks_and_extract_colors(
                image=image01,
                detections=dets,
                include_labels=saved["include_labels"],
                exclude_labels=saved["exclude_labels"],
                score_threshold=saved["score_threshold"],
                n_colors=n_colors,
                custom_colors=custom_colors,
                seed=seed,
            )
        except (ValueError, FileNotFoundError) as exc:
            log("  SKIP {}".format(exc))
            skipped.append(
                {"image": name, "species": species, "stage": "from_json", "reason": str(exc)}
            )
            continue
        rows.extend(
            colour_rows(
                name, species, "final_roi", "+".join(saved["include_labels"]),
                max([v["score"] for v in result["per_label"].values()], default=0.0),
                sum(v["n_detections"] for v in result["per_label"].values()),
                result["color_info"], int(result["final_mask"].sum()),
            )
        )

    out_csv = run_dir / "colors_from_json.csv"
    write_csv(out_csv, rows, COLORS_CSV_FIELDS)
    loaded = "transformers" in sys.modules
    log("\nre-analysed {} images, {} skipped -> {}".format(
        len(entries) - len(skipped), len(skipped), out_csv))
    log("transformers imported during this run: {}".format(loaded))
    assert not loaded, "--from-json must not load a model"
    return {"rows": len(rows), "skipped": len(skipped), "models_loaded": loaded}


# ======================================================================================
# CLI
# ======================================================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reproduction of SegmentR (Boyko 2025) on Argiope images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--argiope-root", type=Path, default=Path(__file__).resolve().parents[2],
                   help="root of the Argiope checkout; every input path is built from it")
    p.add_argument("--images", type=str, default="data/raw/gbif",
                   help="image root, relative to --argiope-root unless absolute")
    p.add_argument("--images-from", type=str, default=None,
                   help="read images from a manifest.jsonl instead of sampling (read-only)")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", type=str, default=DEFAULT_ROI_PROMPT)
    p.add_argument("--exclude-prompt", action="append", default=None,
                   help="repeatable; stage D part subtraction")
    p.add_argument("--reference-prompt", action="append", default=None,
                   help="repeatable; detected and reported but never part of the ROI")
    p.add_argument("--include-labels", type=str, default=None,
                   help="comma-separated override of the labels forming the ROI")
    p.add_argument("--exclude-labels", type=str, default=None,
                   help="comma-separated override of the labels subtracted from the ROI")
    p.add_argument("--detector-id", type=str, default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--box-threshold", type=float, default=0.1)
    p.add_argument("--segmenter-id", type=str, default="Zigeng/SlimSAM-uniform-77")
    p.add_argument("--score-threshold", type=float, default=0.5)
    p.add_argument("--n-colors", type=int, default=5)
    p.add_argument("--custom-colors", type=str, default=None,
                   help="'argiope' for the built-in reference palette, or comma-separated hex")
    p.add_argument("--polygon-refinement", action="store_true")
    p.add_argument("--faithful-json", action="store_true",
                   help="serialise masks as 2-D integer lists, as upstream does")
    p.add_argument("--from-json", type=Path, default=None,
                   help="re-run colour extraction over a finished run, loading no model")
    p.add_argument("--no-qa", action="store_true", help="skip QA rendering")
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--out", type=str, default="repro/segmentr/outputs")
    return p.parse_args(argv)


def resolve_custom_colors(spec: Optional[str]) -> Optional[List[str]]:
    if spec is None:
        return None
    if spec.strip().lower() == "argiope":
        return list(ARGIOPE_PALETTE.values())
    return [c.strip() for c in spec.split(",") if c.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.from_json is not None:
        run_from_json(
            args.from_json,
            n_colors=args.n_colors if "--n-colors" in (argv or sys.argv) else None,
            custom_colors=resolve_custom_colors(args.custom_colors),
            seed=args.seed if "--seed" in (argv or sys.argv) else None,
        )
        return 0

    root = args.argiope_root.resolve()
    seed_everything(args.seed)

    exclude_prompts = args.exclude_prompt if args.exclude_prompt is not None else [
        DEFAULT_EXCLUDE_PROMPT
    ]
    reference_prompts = (
        args.reference_prompt if args.reference_prompt is not None else [DEFAULT_WHOLE_PROMPT]
    )
    include_labels = (
        [s.strip() for s in args.include_labels.split(",")]
        if args.include_labels
        else [args.prompt]
    )
    exclude_labels = (
        [s.strip() for s in args.exclude_labels.split(",")]
        if args.exclude_labels
        else list(exclude_prompts)
    )

    if args.images_from:
        manifest = Path(args.images_from)
        if not manifest.is_absolute():
            manifest = root / manifest
        chosen, notes = images_from_manifest(manifest, root, args.n)
        images_desc = str(manifest)
    else:
        images_root = Path(args.images)
        if not images_root.is_absolute():
            images_root = root / images_root
        chosen, notes = sample_images(images_root, args.n, args.seed)
        images_desc = str(images_root)

    for note in notes:
        log("  sampling: " + note)
    if not chosen:
        log("no images selected -- nothing to do")
        return 1

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = root / out_root
    out_dir = out_root / run_id

    cfg = RunConfig(
        run_id=run_id,
        argiope_root=str(root),
        images=images_desc,
        images_from=args.images_from,
        n=args.n,
        seed=args.seed,
        roi_prompt=args.prompt,
        exclude_prompts=list(exclude_prompts),
        reference_prompts=list(reference_prompts),
        include_labels=include_labels,
        exclude_labels=exclude_labels,
        detector_id=args.detector_id,
        box_threshold=args.box_threshold,
        segmenter_id=args.segmenter_id,
        score_threshold=args.score_threshold,
        n_colors=args.n_colors,
        custom_colors=resolve_custom_colors(args.custom_colors),
        polygon_refinement=args.polygon_refinement,
        faithful_json=args.faithful_json,
        qa=not args.no_qa,
        device=ModelCache.device(),
        started_utc=datetime.now(timezone.utc).isoformat(),
        versions=package_versions(),
        image_list=[{"path": str(p), "species": s} for p, s in chosen],
        sampling_notes=notes,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as fh:
        json.dump(cfg.__dict__, fh, indent=2)
    log("run {} -> {}".format(run_id, out_dir))
    log("{} images, device {}".format(len(chosen), cfg.device))

    run_batch(cfg, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
