#!/usr/bin/env python
"""Adapter: Argiope's trained U-Net -> SegmentR's downstream stages.

The reproduction in `repro_segmentr.py` established that SegmentR's *method* -- the
GroundedSAM stages A-D that isolate a part from a text prompt -- does not recover the
opisthosoma (REPORT.md; median 99.91% of the ROI deleted by part subtraction, and the cause
is the detector, not the collapse rule). What survives that finding is everything downstream
of the mask, which was never segmentation in the first place:

    E  colour in CIELAB          extract_colors
    F  re-loadable JSON artefact write_image_json / dicts_to_detections
    G  QA visuals + cut-outs     write_qa_figure / export_transparent_png

This module feeds those stages from a mask that works: Argiope's trained U-Net
(`UnetSegmenter`, held-out IoU median 0.755). Stages A-D are not imported and not used.

Two deliberate departures from `repro_segmentr.run_batch`, both following the handoff:

* **`combine_masks` / `merge_masks_by_label` are bypassed.** They OR every detection
  carrying the same label into one mask, which destroys per-specimen identity -- two spiders
  in a frame would become one blob, and a taxonomic ledger needs one record per specimen.
  The U-Net already returns exactly one opisthosoma mask, so colour is taken straight from
  `extract_colors`, the perceptual core, without the stage-C/D wrapper around it.
* **No part subtraction.** Its whole purpose was to compensate for not being able to prompt
  a part. A segmenter trained on the part does not need it.

`argiope` is imported lazily, inside the mask source that needs it, so importing this module
(and `--from-json`, which loads no model at all) stays free of torch.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from repro_segmentr import (  # noqa: E402  (reused verbatim from the reproduction)
    COLORS_CSV_FIELDS,
    BoundingBox,
    DetectionResult,
    colour_rows,
    dicts_to_detections,
    export_transparent_png,
    extract_colors,
    load_image,
    log,
    resolve_custom_colors,
    write_csv,
    write_image_json,
    write_qa_figure,
)

OPISTHOSOMA_LABEL = "opisthosoma"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


# ======================================================================================
# Pure helpers -- no models, no I/O. These are what tests/test_adapt_unet.py exercises.
# ======================================================================================

def mask_bounds(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Inclusive (xmin, ymin, xmax, ymax) of a boolean mask. Raises on an empty mask."""
    ys, xs = np.where(np.asarray(mask).astype(bool))
    if xs.size == 0:
        raise ValueError("mask is empty")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def mask_to_detection(
    mask: np.ndarray, score: float = 1.0, label: str = OPISTHOSOMA_LABEL
) -> DetectionResult:
    """Wrap one segmenter mask in the structure SegmentR's downstream stages expect.

    `DetectionResult.box` is the mask's own bounding box, not a detector box: nothing
    downstream uses it as a constraint, it exists so the JSON artefact keeps upstream's
    schema and the QA overlay has something to draw.
    """
    xmin, ymin, xmax, ymax = mask_bounds(mask)
    return DetectionResult(
        score=float(score),
        label=label,
        box=BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax),
        mask=np.asarray(mask).astype(bool).astype(np.uint8),
    )


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Overlap between two masks. Used only by the sanity check, never as a quality claim."""
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    union = int((a | b).sum())
    return float((a & b).sum()) / union if union else 0.0


def collect_images(root: Path, n: Optional[int] = None, seed: int = 42) -> List[Tuple[Path, str]]:
    """Images under `root`, recursively. `group` is the parent directory name."""
    import random

    root = Path(root)
    if root.is_file():
        return [(root, root.parent.name)]
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if n is not None and n < len(files):
        files = sorted(random.Random(seed).sample(files, n))
    return [(p, p.parent.name) for p in files]


# ======================================================================================
# Mask sources -- the only place `argiope` is touched.
# ======================================================================================

class UnetImportSource:
    """In-process `UnetSegmenter`. One model load for the whole batch."""

    name = "import"

    def __init__(self, weights: Path, encoder: str, imgsz: int, threshold: float, device: str):
        self.cfg = {
            "unet_weights": str(weights),
            "unet_encoder": encoder,
            "unet_imgsz": imgsz,
            "unet_threshold": threshold,
            "device": device,
        }
        self._seg = None

    def load(self):
        if self._seg is None:
            from argiope.segmentation.unet_backend import UnetSegmenter

            log(f"  loading UnetSegmenter from {self.cfg['unet_weights']} on {self.cfg['device']}")
            self._seg = UnetSegmenter(self.cfg).load()
        return self._seg

    def mask_for(self, image_rgb: np.ndarray, image_path: Path) -> Tuple[np.ndarray, float, str]:
        res = self.load().segment(image_rgb)
        return np.asarray(res.mask).astype(bool), float(res.score), res.backend


class CliSource:
    """`argiope describe --mask` in a subprocess: the same boundary the R layer uses.

    Slower (a model load per image) but it exercises the published contract end to end,
    which is what makes it worth keeping as an option.
    """

    name = "cli"

    def __init__(self, executable: str = "argiope", config: Optional[Path] = None):
        self.executable = executable
        self.config = config

    def load(self):
        return self

    def mask_for(self, image_rgb: np.ndarray, image_path: Path) -> Tuple[np.ndarray, float, str]:
        with tempfile.TemporaryDirectory() as td:
            mask_png = Path(td) / "mask.png"
            json_out = Path(td) / "out.json"
            cmd = [self.executable, "describe", str(image_path),
                   "--json", str(json_out), "--mask", str(mask_png)]
            if self.config:
                cmd += ["--config", str(self.config)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"argiope describe failed: {proc.stderr.strip()[:400]}")
            arr = cv2.imread(str(mask_png), cv2.IMREAD_GRAYSCALE)
            if arr is None:
                raise RuntimeError("argiope describe wrote no mask")
            payload = json.loads(json_out.read_text(encoding="utf-8"))
            opistho = payload.get("opisthosoma") or {}
            return (arr > 127), float(opistho.get("score", 1.0)), str(opistho.get("backend", "cli"))


# ======================================================================================
# Run configuration and the batch driver
# ======================================================================================

@dataclass
class AdapterConfig:
    run_id: str
    argiope_root: str
    images: str
    n: Optional[int]
    seed: int
    source: str
    weights: str
    encoder: str
    imgsz: int
    threshold: float
    device: str
    n_colors: int
    custom_colors: Optional[List[str]]
    faithful_json: bool
    qa: bool
    label: str = OPISTHOSOMA_LABEL
    started_utc: str = ""
    versions: Dict[str, str] = field(default_factory=dict)
    image_list: List[Dict[str, str]] = field(default_factory=list)


def package_versions() -> Dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out = {"python": sys.version.split()[0]}
    for pkg in ("torch", "segmentation-models-pytorch", "numpy", "scikit-learn",
                "scikit-image", "opencv-python", "matplotlib"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def run_batch(cfg: AdapterConfig, out_dir: Path, source) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "json").mkdir(exist_ok=True)
    if cfg.qa:
        (out_dir / "qa").mkdir(exist_ok=True)
        (out_dir / "cutouts").mkdir(exist_ok=True)

    rows: List[Dict] = []
    skipped: List[Dict] = []
    processed = 0
    total = len(cfg.image_list)

    for idx, entry in enumerate(cfg.image_list, start=1):
        img_path = Path(entry["path"])
        group, name = entry["group"], img_path.name
        log(f"Processing image {idx} of {total}: {group}/{name}")

        try:
            image = load_image(str(img_path))
        except Exception as exc:
            log(f"  SKIP unreadable image: {exc}")
            skipped.append({"image": name, "group": group, "stage": "load", "reason": str(exc)})
            continue

        width, height = image.size
        rgb = np.asarray(image, dtype=np.uint8)
        image01 = rgb.astype(float) / 255.0

        try:
            mask, score, backend = source.mask_for(rgb, img_path)
        except Exception as exc:
            log(f"  SKIP segmenter failed: {exc}")
            skipped.append({"image": name, "group": group, "stage": "segment", "reason": str(exc)})
            continue

        try:
            det = mask_to_detection(mask, score, cfg.label)
        except ValueError as exc:
            log(f"  SKIP {exc}")
            skipped.append({"image": name, "group": group, "stage": "mask", "reason": str(exc)})
            continue

        # --- stage F: the re-loadable artefact -------------------------------------
        json_path = out_dir / "json" / f"{group}__{img_path.stem}.json"
        write_image_json(json_path, [det], height, width, cfg.faithful_json)

        # --- stage E: colour, straight from the perceptual core ---------------------
        try:
            info = extract_colors(image01, mask, n_colors=cfg.n_colors,
                                  custom_colors=cfg.custom_colors, seed=cfg.seed)
        except ValueError as exc:
            log(f"  SKIP colour: {exc}")
            skipped.append({"image": name, "group": group, "stage": "colour", "reason": str(exc)})
            continue

        px = int(np.asarray(mask).sum())
        rows.extend(colour_rows(name, group, "opisthosoma", cfg.label, score, 1, info, px))
        log(f"  mask {px} px ({px / float(width * height):.2%} of frame) score {score:.3f} "
            f"-> {len(info['dominant_color_info'])} clusters, mean {info['mean_color']}")

        # --- stage G: QA + cut-outs ------------------------------------------------
        if cfg.qa:
            try:
                write_qa_figure(
                    out_dir / "qa" / f"{group}__{img_path.stem}_qa.png",
                    image01, [det], np.asarray(mask).astype(bool), info,
                    f"{group} / {name}   {backend}   "
                    f"mask={px / float(width * height):.2%} of frame",
                )
                export_transparent_png(
                    image01, {cfg.label: np.asarray(mask).astype(bool)},
                    out_dir / "cutouts", f"{group}__{img_path.stem}",
                    crop=True, remove_overlap=False,
                )
            except Exception as exc:
                log(f"  note: QA rendering failed: {exc}")
                skipped.append({"image": name, "group": group, "stage": "qa", "reason": str(exc)})

        processed += 1

    write_csv(out_dir / "colors.csv", rows, COLORS_CSV_FIELDS)
    write_csv(out_dir / "skipped.csv", skipped, ["image", "group", "stage", "reason"])
    summary = {"run_id": cfg.run_id, "n_images": total, "processed": processed,
               "skipped": len(skipped), "colour_rows": len(rows)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"\n{processed} processed, {len(skipped)} skipped -> {out_dir}")
    return summary


def run_from_json(run_dir: Path, n_colors: Optional[int], custom_colors: Optional[List[str]],
                  seed: Optional[int]) -> Dict:
    """Stage F's payoff: re-run colour over a finished run without loading the U-Net."""
    run_dir = Path(run_dir)
    saved = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    n_colors = saved["n_colors"] if n_colors is None else n_colors
    seed = saved["seed"] if seed is None else seed
    if custom_colors is None:
        custom_colors = saved.get("custom_colors")
    label = saved.get("label", OPISTHOSOMA_LABEL)

    rows: List[Dict] = []
    skipped: List[Dict] = []
    entries = saved["image_list"]
    for idx, entry in enumerate(entries, start=1):
        img_path = Path(entry["path"])
        group, name = entry["group"], img_path.name
        jp = run_dir / "json" / f"{group}__{img_path.stem}.json"
        log(f"Processing image {idx} of {len(entries)}: {name} (from JSON)")
        if not jp.exists():
            log("  SKIP no JSON artefact for this image")
            skipped.append({"image": name, "group": group, "stage": "from_json",
                            "reason": "no JSON"})
            continue
        try:
            dets = dicts_to_detections(json.loads(jp.read_text(encoding="utf-8")), jp.parent)
            dets = [d for d in dets if d.label == label and d.mask is not None]
            if not dets:
                raise ValueError("no opisthosoma mask in the artefact")
            mask = np.asarray(dets[0].mask).astype(bool)
            image01 = np.asarray(load_image(str(img_path)), dtype=float) / 255.0
            info = extract_colors(image01, mask, n_colors=n_colors,
                                  custom_colors=custom_colors, seed=seed)
        except (ValueError, FileNotFoundError) as exc:
            log(f"  SKIP {exc}")
            skipped.append({"image": name, "group": group, "stage": "from_json",
                            "reason": str(exc)})
            continue
        rows.extend(colour_rows(name, group, "opisthosoma", label, float(dets[0].score), 1,
                                info, int(mask.sum())))

    out_csv = run_dir / "colors_from_json.csv"
    write_csv(out_csv, rows, COLORS_CSV_FIELDS)
    # torch arrives unconditionally via repro_segmentr's module-level import, so it is not
    # evidence of anything; what must stay absent is the segmenter itself.
    loaded = ("argiope.segmentation.unet_backend" in sys.modules
              or "segmentation_models_pytorch" in sys.modules)
    log(f"\nre-analysed {len(entries) - len(skipped)} images, {len(skipped)} skipped -> {out_csv}")
    log(f"segmenter loaded during this run: {loaded}")
    assert not loaded, "--from-json must not load the U-Net"
    return {"rows": len(rows), "skipped": len(skipped), "model_loaded": loaded}


# ======================================================================================
# CLI
# ======================================================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run SegmentR's colour/JSON/QA stages on Argiope's trained U-Net masks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--argiope-root", type=Path, default=HERE.parents[1])
    p.add_argument("--images", type=str, default="data/interim/opistho_seg/images")
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--source", choices=("import", "cli"), default="import",
                   help="in-process UnetSegmenter, or the `argiope describe` CLI contract")
    p.add_argument("--weights", type=str, default="checkpoints/opistho_unet.pt")
    p.add_argument("--encoder", type=str, default="resnet34")
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n-colors", type=int, default=5)
    p.add_argument("--custom-colors", type=str, default=None)
    p.add_argument("--faithful-json", action="store_true")
    p.add_argument("--from-json", type=Path, default=None)
    p.add_argument("--no-qa", action="store_true")
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--out", type=str, default="repro/segmentr/out_unet")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.from_json is not None:
        run_from_json(args.from_json, None, resolve_custom_colors(args.custom_colors), None)
        return 0

    root = args.argiope_root.resolve()
    images_root = Path(args.images)
    if not images_root.is_absolute():
        images_root = root / images_root
    chosen = collect_images(images_root, args.n, args.seed)
    if not chosen:
        log(f"no images found under {images_root}")
        return 1

    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = root / weights

    if args.source == "import":
        if not weights.exists():
            log(f"missing checkpoint: {weights}")
            log("regenerate with: conda activate argiope && argiope train-segmenter")
            return 1
        source = UnetImportSource(weights, args.encoder, args.imgsz, args.threshold, args.device)
    else:
        source = CliSource()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("unet-%Y%m%dT%H%M%SZ")
    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = root / out_root
    out_dir = out_root / run_id

    cfg = AdapterConfig(
        run_id=run_id, argiope_root=str(root), images=str(images_root), n=args.n, seed=args.seed,
        source=args.source, weights=str(weights), encoder=args.encoder, imgsz=args.imgsz,
        threshold=args.threshold, device=args.device, n_colors=args.n_colors,
        custom_colors=resolve_custom_colors(args.custom_colors),
        faithful_json=args.faithful_json, qa=not args.no_qa,
        started_utc=datetime.now(timezone.utc).isoformat(), versions=package_versions(),
        image_list=[{"path": str(p), "group": g} for p, g in chosen],
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(json.dumps(cfg.__dict__, indent=2), encoding="utf-8")
    log(f"run {run_id} -> {out_dir}")
    log(f"{len(chosen)} images, source={args.source}")

    run_batch(cfg, out_dir, source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
