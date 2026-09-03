"""Sanity check: does the U-Net mask give the same colours as the ground-truth mask?

The question the adapter has to answer is not "is the mask good" -- that is the segmenter's
own validation, already done -- but "does a good-enough mask produce the same colour
description as a perfect one". If the palette from the predicted mask matches the palette
from the hand-drawn mask wherever IoU is decent, the colour stage is robust to the
segmenter's residual error and the adapter is safe to use.

Runs over the held-out split only, rebuilt with Argiope's own `split_pairs` (seed 42,
val_frac 0.2) so it is exactly the split the model never trained on.

Colour agreement is reported as CIE76 dE between the dominant (largest) cluster of each
palette, plus a coverage-weighted palette distance: for every GT cluster, the dE to its
nearest predicted cluster, weighted by that cluster's share of the mask.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import adapt_unet as ad  # noqa: E402
from repro_segmentr import extract_colors, load_image, log, write_csv  # noqa: E402


def palette_lab(info) -> tuple[np.ndarray, np.ndarray]:
    """(k, 3) Lab centroids and their coverage fractions, largest first."""
    rows = info["dominant_color_info"]
    lab = np.array([[r["lab_l"], r["lab_a"], r["lab_b"]] for r in rows], dtype=float)
    size = np.array([r["cluster_size"] for r in rows], dtype=float)
    total = size.sum() or 1.0
    order = np.argsort(-size)
    return lab[order], size[order] / total


def palette_distance(gt_info, pred_info) -> tuple[float, float]:
    """(dE between dominant clusters, coverage-weighted nearest-cluster dE)."""
    g_lab, g_cov = palette_lab(gt_info)
    p_lab, _ = palette_lab(pred_info)
    dominant = float(np.linalg.norm(g_lab[0] - p_lab[0]))
    d = np.linalg.norm(g_lab[:, None, :] - p_lab[None, :, :], axis=2)   # CIE76
    weighted = float((d.min(axis=1) * g_cov).sum())
    return dominant, weighted


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--argiope-root", type=Path, default=HERE.parents[2])
    p.add_argument("--dataset", type=str, default="data/interim/opistho_seg")
    p.add_argument("--weights", type=str, default="checkpoints/opistho_unet.pt")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n-colors", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--iou-good", type=float, default=0.7)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    root = a.argiope_root.resolve()
    ds = root / a.dataset
    out_dir = a.out or (root / "repro/segmentr/outputs/sanity-gt-vs-unet")
    out_dir.mkdir(parents=True, exist_ok=True)

    # the model's own split, reused rather than re-implemented
    from argiope.segmentation.train import load_pairs, split_pairs

    _, val = split_pairs(load_pairs(ds), a.val_frac, a.seed)
    log(f"held-out split: {len(val)} pairs (seed {a.seed}, val_frac {a.val_frac})")

    source = ad.UnetImportSource(root / a.weights, "resnet34", 512, 0.5, a.device)
    rows = []
    for i, (img_rel, mask_rel) in enumerate(val, start=1):
        # load_pairs returns paths relative to the Argiope root, not to the dataset dir
        img_path = Path(img_rel) if Path(img_rel).is_absolute() else root / img_rel
        gt_path = Path(mask_rel) if Path(mask_rel).is_absolute() else root / mask_rel
        name = Path(img_rel).name
        log(f"Processing image {i} of {len(val)}: {name}")

        rgb = np.asarray(load_image(str(img_path)), dtype=np.uint8)
        image01 = rgb.astype(float) / 255.0
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            log("  SKIP unreadable GT mask")
            continue
        gt = gt > 127
        pred, score, _ = source.mask_for(rgb, img_path)

        row = {
            "image": name, "gt_px": int(gt.sum()), "pred_px": int(pred.sum()),
            "score": round(float(score), 4), "iou": round(ad.iou(gt, pred), 4),
            "dominant_de": "", "weighted_de": "", "gt_hex": "", "pred_hex": "", "note": "",
        }
        if not pred.any():
            row["note"] = "empty predicted mask"
            log(f"  IoU 0.0000 — empty predicted mask (GT {int(gt.sum())} px)")
        elif not gt.any():
            row["note"] = "empty GT mask"
        else:
            gi = extract_colors(image01, gt, n_colors=a.n_colors, seed=a.seed)
            pi = extract_colors(image01, pred, n_colors=a.n_colors, seed=a.seed)
            dom, wt = palette_distance(gi, pi)
            row["dominant_de"] = round(dom, 2)
            row["weighted_de"] = round(wt, 2)
            row["gt_hex"] = gi["dominant_color_info"][int(np.argmax(
                [c["cluster_size"] for c in gi["dominant_color_info"]]))]["hex_color"]
            row["pred_hex"] = pi["dominant_color_info"][int(np.argmax(
                [c["cluster_size"] for c in pi["dominant_color_info"]]))]["hex_color"]
            log(f"  IoU {row['iou']:.3f}  dominant dE {dom:5.2f}  weighted dE {wt:5.2f}  "
                f"{row['gt_hex']} vs {row['pred_hex']}")
        rows.append(row)

    write_csv(out_dir / "sanity.csv", rows, list(rows[0].keys()))

    def q(v, pc):
        s = sorted(v)
        return s[min(len(s) - 1, int(pc * (len(s) - 1) + 0.5))] if s else float("nan")

    ious = [r["iou"] for r in rows]
    good = [r for r in rows if r["iou"] >= a.iou_good and r["dominant_de"] != ""]
    scored = [r for r in rows if r["dominant_de"] != ""]
    empty = sum(1 for r in rows if r["note"] == "empty predicted mask")

    summary = {
        "n": len(rows),
        "empty_predicted_masks": empty,
        "iou_median": round(q(ious, .5), 4),
        "iou_mean": round(float(np.mean(ious)), 4),
        "iou_ge_good": len(good),
        "iou_good_threshold": a.iou_good,
        "dominant_de_median_all":
            round(q([r["dominant_de"] for r in scored], .5), 2) if scored else None,
        "dominant_de_median_good":
            round(q([r["dominant_de"] for r in good], .5), 2) if good else None,
        "weighted_de_median_good":
            round(q([r["weighted_de"] for r in good], .5), 2) if good else None,
        "good_within_de_5": sum(1 for r in good if r["weighted_de"] <= 5),
        "good_within_de_10": sum(1 for r in good if r["weighted_de"] <= 10),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log("\n" + "=" * 62)
    log(f"held-out n={summary['n']}  IoU median {summary['iou_median']}  "
        f"mean {summary['iou_mean']}  empty masks {empty}")
    log(f"images with IoU >= {a.iou_good}: {summary['iou_ge_good']}")
    log(f"  dominant-cluster dE, median: {summary['dominant_de_median_good']}")
    log(f"  coverage-weighted palette dE, median: {summary['weighted_de_median_good']}")
    log(f"  of those, within dE 5: {summary['good_within_de_5']}  "
        f"within dE 10: {summary['good_within_de_10']}")
    log("(dE 2.3 is the classic just-noticeable difference; dE 10 is a clearly different colour)")
    log(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
