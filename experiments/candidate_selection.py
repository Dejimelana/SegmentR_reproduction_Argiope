"""EXPERIMENT — not part of the reproduction.

The reproduction (repro_segmentr.py, REPORT.md, the two commits) reproduces Boyko (2025)
faithfully and stays untouched. This script asks a question the reproduction deliberately
could not: `refine_masks` collapses SlimSAM's three candidate masks by taking their UNION
and throws away the `iou_scores` the model returns alongside them. If we *select* the
candidate the model rates highest instead of unioning all three, does the abdomen come
back?

Design — a controlled comparison, one variable:

* the boxes are read from the finished run's JSON artefacts, so GroundingDINO is never
  re-run and both arms see byte-identical boxes, labels and scores;
* one SlimSAM forward pass per image supplies both arms, so the weights, the processor
  and the post-processing are shared;
* the ONLY difference is how (n_boxes, 3, H, W) becomes (n_boxes, H, W):
    union  = (mask.float().mean(0) > 0)      <- upstream refine_masks
    argmax = mask[argmax(iou_scores)]        <- the variant under test

It can falsify the report's stated mechanism. If the three candidates turn out to have
near-identical areas, then the union is not what destroys the part mask, the real culprit
is the whole-animal box GroundingDINO hands to SAM, and the report's explanation needs
correcting. That outcome is recorded, not hidden.

No ground truth is involved and nothing here is a segmentation-quality metric; every
number is an area the models themselves produced.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import repro_segmentr as rs  # noqa: E402


def collapse_union(masks_bool: torch.Tensor) -> np.ndarray:
    """Upstream refine_masks, restricted to one box: mean over candidates, threshold > 0."""
    m = masks_bool.float().mean(dim=0)
    return (m > 0).numpy().astype(bool)


def run(run_dir: Path, out_dir: Path, score_threshold: float, limit: int | None) -> None:
    cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForMaskGeneration, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg_id = cfg["segmenter_id"]
    rs.log(f"loading {seg_id} on {device} (detector is NOT loaded: boxes come from the JSON)")
    model = AutoModelForMaskGeneration.from_pretrained(seg_id).to(device)
    proc = AutoProcessor.from_pretrained(seg_id)

    roi_label = rs.normalise_label(cfg["include_labels"][0])
    exc_labels = {rs.normalise_label(x) for x in cfg["exclude_labels"]}
    ref_labels = {rs.normalise_label(x) for x in cfg["reference_prompts"]}

    per_box, per_image = [], []
    entries = cfg["image_list"][: limit or len(cfg["image_list"])]

    for idx, entry in enumerate(entries, start=1):
        img_path = Path(entry["path"])
        species, name = entry["species"], img_path.name
        jp = run_dir / "json" / f"{species}__{img_path.stem}.json"
        rs.log(f"Processing image {idx} of {len(entries)}: {species}/{name}")
        if not jp.exists():
            rs.log("  SKIP no JSON artefact")
            continue

        records = json.loads(jp.read_text(encoding="utf-8"))
        dets = [rs.DetectionResult.from_dict(r) for r in records]
        keep = [d for d in dets if d.score >= score_threshold]
        if not keep:
            rs.log("  SKIP no detection above the score threshold")
            continue

        image = rs.load_image(str(img_path))
        W, H = image.size
        boxes = rs.get_boxes(keep)

        inputs = proc(images=image, input_boxes=boxes, return_tensors="pt")
        inputs = {k: (v.to(torch.float32) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)

        orig = [(int(s[0].item()), int(s[1].item())) for s in inputs["original_sizes"].cpu()]
        resh = [(int(s[0].item()), int(s[1].item())) for s in inputs["reshaped_input_sizes"].cpu()]
        masks = proc.post_process_masks(
            out.pred_masks.cpu(), original_sizes=orig, reshaped_input_sizes=resh
        )[0]                                   # (n_boxes, 3, H, W) bool
        iou = out.iou_scores.cpu().numpy()[0]  # (n_boxes, 3)

        by_label_u, by_label_a = {}, {}
        for i, det in enumerate(keep):
            cand = masks[i]                     # (3, H, W)
            areas = [int(cand[c].sum()) for c in range(cand.shape[0])]
            best = int(np.argmax(iou[i]))
            u = collapse_union(cand)
            a = cand[best].numpy().astype(bool)
            lab = rs.normalise_label(det.label)

            per_box.append({
                "image": name, "species": species, "label": lab,
                "det_score": round(float(det.score), 4), "box_index": i,
                "iou_0": round(float(iou[i][0]), 4), "iou_1": round(float(iou[i][1]), 4),
                "iou_2": round(float(iou[i][2]), 4), "best_candidate": best,
                "area_cand_0": areas[0], "area_cand_1": areas[1], "area_cand_2": areas[2],
                "area_union": int(u.sum()), "area_selected": int(a.sum()),
                "frame_px": W * H,
                "spread": round(max(areas) / max(1, min(areas)), 4),
                "selected_over_union": round(int(a.sum()) / max(1, int(u.sum())), 4),
            })
            for store, m in ((by_label_u, u), (by_label_a, a)):
                store[lab] = store[lab] | m if lab in store else m.copy()

        def summarise(store, arm):
            roi = store.get(roi_label)
            if roi is None:
                return None
            exc = [store[k] for k in exc_labels if k in store]
            ref = [store[k] for k in ref_labels if k in store]
            final = rs.exclude_masks(roi, exc) if exc else roi
            ref_px = int(np.logical_or.reduce(ref).sum()) if ref else 0
            return {
                "image": name, "species": species, "arm": arm,
                "roi_px": int(roi.sum()), "ref_px": ref_px,
                "exclude_px": int(np.logical_or.reduce(exc).sum()) if exc else 0,
                "final_px": int(final.sum()), "frame_px": W * H,
                "granularity": round(int(roi.sum()) / ref_px, 4) if ref_px else "",
                "survival": round(int(final.sum()) / max(1, int(roi.sum())), 6),
            }

        for arm, store in (("union_upstream", by_label_u), ("argmax_iou", by_label_a)):
            row = summarise(store, arm)
            if row:
                per_image.append(row)
        u_row = per_image[-2] if len(per_image) >= 2 and per_image[-1]["arm"] == "argmax_iou" else None
        if u_row:
            rs.log(f"  union  roi={u_row['roi_px']:>8} px  final={u_row['final_px']:>7} px   "
                   f"argmax roi={per_image[-1]['roi_px']:>8} px  final={per_image[-1]['final_px']:>7} px")

    rs.write_csv(out_dir / "per_box.csv", per_box, list(per_box[0].keys()) if per_box else ["image"])
    rs.write_csv(out_dir / "per_image.csv", per_image,
                 list(per_image[0].keys()) if per_image else ["image"])
    (out_dir / "experiment_config.json").write_text(json.dumps({
        "source_run": str(run_dir), "segmenter_id": seg_id, "device": device,
        "score_threshold": score_threshold, "detector_loaded": False,
        "note": "boxes reused from the source run's JSON; only the candidate-collapse rule varies",
    }, indent=2), encoding="utf-8")
    rs.log(f"\n{len(per_box)} boxes, {len(per_image) // 2} images -> {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--score-threshold", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()
    run(a.run_dir.resolve(), a.out.resolve(), a.score_threshold, a.limit)
