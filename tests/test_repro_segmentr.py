"""Pure-function tests for the SegmentR reproduction.

No weights, no GPU, no network: every test here runs on synthetic arrays. Model code
(`detect`, `segment`) is exercised only through `ModelCache`, which is never constructed.
Style follows the parent project's tests/test_segmentation_backend.py, which likewise
tests only the pure part of a model-backed module.

Several tests pin down behaviour that is *wrong* in the original and reproduced anyway
(`refine_masks` taking the union of SlimSAM's three candidates; mean/median colour taken
in RGB; the double-scaled per-pixel hex). They are named so that a future reader cannot
mistake them for endorsements.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import repro_segmentr as rs  # noqa: E402

H = W = 8


def _rect(r0, r1, c0, c1, h=H, w=W):
    m = np.zeros((h, w), np.uint8)
    m[r0:r1, c0:c1] = 1
    return m


def _det(score, label, mask=None, box=(0, 0, 1, 1)):
    return rs.DetectionResult(
        score=score, label=label, box=rs.BoundingBox(*box), mask=mask
    )


# ---------------------------------------------------------------- isolation guarantees

def test_module_does_not_import_argiope_or_a_model():
    """The hard rule: the probe must not lean on the pipeline it will be compared to."""
    assert "argiope" not in sys.modules
    assert "transformers" not in sys.modules


def test_argiope_root_default_is_two_levels_up():
    assert rs.parse_args([]).argiope_root == Path(rs.__file__).resolve().parents[2]


# ------------------------------------------------------- ported dataclasses (utils.py)

def test_bounding_box_xyxy():
    assert rs.BoundingBox(1, 2, 3, 4).xyxy == [1, 2, 3, 4]


def test_detection_result_from_dict():
    det = rs.DetectionResult.from_dict(
        {"score": 0.7, "label": "a spider.", "box": {"xmin": 1, "ymin": 2, "xmax": 3, "ymax": 4}}
    )
    assert det.score == 0.7 and det.label == "a spider." and det.box.xyxy == [1, 2, 3, 4]
    assert det.mask is None


def test_get_boxes_wraps_one_level_deeper():
    """SlimSAM's processor wants [[[x0,y0,x1,y1], ...]] -- upstream's extra nesting."""
    boxes = rs.get_boxes([_det(0.9, "a", box=(1, 2, 3, 4)), _det(0.8, "b", box=(5, 6, 7, 8))])
    assert boxes == [[[1, 2, 3, 4], [5, 6, 7, 8]]]


def test_normalise_label_ignores_case_and_terminating_period():
    assert rs.normalise_label(" The Abdomen of a Spider. ") == "the abdomen of a spider"
    assert rs.normalise_label("a spider") == rs.normalise_label("a spider.")


# --------------------------------------------------------------- refine_masks (utils.py)

def test_refine_masks_takes_the_union_of_slimsams_three_candidates():
    """Reproduced defect, not endorsed.

    SlimSAM emits three candidate masks per box to resolve whole/part/subpart ambiguity.
    Upstream averages over that axis and thresholds at > 0, i.e. takes their union, so
    the step that should pick a part instead returns everything any candidate proposed.
    """
    cand_a = _rect(0, 2, 0, 2)
    cand_b = _rect(5, 8, 5, 8)
    cand_c = np.zeros((H, W), np.uint8)
    stacked = torch.tensor(np.stack([cand_a, cand_b, cand_c])[None], dtype=torch.bool)

    out = rs.refine_masks(stacked, polygon_refinement=False)

    assert len(out) == 1
    np.testing.assert_array_equal(out[0], np.maximum(cand_a, cand_b))
    assert out[0].dtype == np.uint8


def test_refine_masks_polygon_refinement_fills_the_interior():
    ring = _rect(1, 7, 1, 7)
    ring[3:5, 3:5] = 0
    stacked = torch.tensor(ring[None, None], dtype=torch.bool)

    plain = rs.refine_masks(stacked, polygon_refinement=False)[0]
    refined = rs.refine_masks(stacked, polygon_refinement=True)[0]

    assert plain[3, 3] == 0                    # the hole survives without refinement
    assert refined[3, 3] > 0                   # fillPoly of the largest contour closes it


# ---------------------------------------------- stage C: combine / exclude / merge

def test_combine_masks_ors_everything_above_threshold():
    a, b, low = _rect(0, 2, 0, 2), _rect(2, 4, 2, 4), _rect(6, 8, 6, 8)
    out = rs.combine_masks([a, b, low], [0.9, 0.6, 0.2], threshold=0.5)
    np.testing.assert_array_equal(out, (np.maximum(a, b) > 0))
    assert not out[7, 7]                        # the sub-threshold mask contributed nothing


def test_combine_masks_threshold_is_inclusive():
    """Upstream uses scores >= threshold; a score exactly at 0.5 is kept."""
    assert rs.combine_masks([_rect(0, 2, 0, 2)], [0.5], threshold=0.5).any()


def test_combine_masks_raises_when_nothing_survives():
    """Upstream's stop(); the batch driver turns this into a logged, counted skip."""
    with pytest.raises(ValueError, match="No masks meet the score threshold"):
        rs.combine_masks([_rect(0, 2, 0, 2)], [0.4], threshold=0.5)


def test_exclude_masks_subtracts_and_leaves_the_input_alone():
    roi = _rect(0, 8, 0, 8) > 0
    legs = _rect(0, 8, 6, 8)
    out = rs.exclude_masks(roi, [legs])
    assert out[:, :6].all() and not out[:, 6:].any()
    assert roi.all()                            # input not mutated


def test_exclude_masks_with_no_exclusions_is_identity():
    roi = _rect(1, 4, 1, 4) > 0
    np.testing.assert_array_equal(rs.exclude_masks(roi, []), roi)


def test_merge_masks_by_label_ors_same_label_and_drops_low_scores():
    dets = [
        _det(0.8, "the abdomen of a spider.", _rect(0, 2, 0, 2)),
        _det(0.6, "the abdomen of a spider", _rect(2, 4, 2, 4)),
        _det(0.3, "the abdomen of a spider.", _rect(6, 8, 6, 8)),   # below threshold
        _det(0.9, "the legs of a spider.", _rect(4, 6, 4, 6)),
    ]
    merged = rs.merge_masks_by_label(dets, score_threshold=0.5)

    assert set(merged) == {"the abdomen of a spider", "the legs of a spider"}
    abd = merged["the abdomen of a spider"]
    assert abd["n_detections"] == 2
    assert abd["score"] == pytest.approx(0.8)   # max over the merged detections
    assert abd["mask"][0, 0] and abd["mask"][2, 2] and not abd["mask"][7, 7]


def test_merge_masks_by_label_ignores_detections_without_a_mask():
    assert rs.merge_masks_by_label([_det(0.9, "x", None)], score_threshold=0.5) == {}


# ----------------------------------------------------------- stage E: colour in CIELAB

def _two_colour_image(left="#FF0000", right="#0000FF"):
    img = np.zeros((4, 4, 3), float)
    img[:, :2] = rs.hex_to_rgb01(left)
    img[:, 2:] = rs.hex_to_rgb01(right)
    return img


def test_hex_rgb_roundtrip():
    for hex_code in ("#FFD700", "#000000", "#8B5A2B", "#F0F0F0"):
        assert rs.rgb01_to_hex(rs.hex_to_rgb01(hex_code)) == hex_code


def test_rgb01_to_lab_matches_the_textbook_value_for_pure_red():
    lab = rs.rgb01_to_lab(np.array([[1.0, 0.0, 0.0]]))[0]
    np.testing.assert_allclose(lab, [53.2408, 80.0925, 67.2032], atol=1e-3)


def test_extract_palette_two_colours():
    img = _two_colour_image()
    mask = np.ones((4, 4), bool)
    info = rs.extract_colors(img, mask, n_colors=2, seed=42)["dominant_color_info"]

    assert sorted(c["hex_color"] for c in info) == ["#0000FF", "#FF0000"]
    assert sorted(c["cluster_size"] for c in info) == [8, 8]


def test_cluster_sizes_sum_to_the_number_of_masked_pixels():
    img = _two_colour_image()
    mask = np.zeros((4, 4), bool)
    mask[:, :3] = True                                    # 12 pixels
    out = rs.extract_colors(img, mask, n_colors=2, seed=42)
    assert sum(c["cluster_size"] for c in out["dominant_color_info"]) == 12
    assert len(out["masked_pixels"]) == 12


def test_mask_selects_only_the_masked_region():
    img = _two_colour_image()
    mask = np.zeros((4, 4), bool)
    mask[:, :2] = True                                    # the red half only
    info = rs.extract_colors(img, mask, n_colors=1, seed=42)["dominant_color_info"]
    assert info[0]["hex_color"] == "#FF0000"


def test_custom_colours_assign_every_pixel_to_the_nearest_reference_in_lab():
    """The supervised branch: centres are the references, sizes come from assignment."""
    img = _two_colour_image("#FF0000", "#0000FF")
    mask = np.ones((4, 4), bool)
    refs = ["#FF0000", "#0000FF", "#00FF00"]
    info = rs.extract_colors(img, mask, custom_colors=refs, seed=42)["dominant_color_info"]

    assert [c["hex_color"] for c in info] == refs         # centres are the references
    assert [c["cluster_size"] for c in info] == [8, 8, 0]  # green claims nothing


def test_custom_colours_ignore_n_colors():
    img = _two_colour_image()
    info = rs.extract_colors(
        img, np.ones((4, 4), bool), n_colors=5, custom_colors=["#FF0000", "#0000FF"]
    )["dominant_color_info"]
    assert len(info) == 2


def test_mean_and_median_colour_are_taken_in_rgb_not_lab():
    """Reproduced defect, not endorsed.

    Upstream computes colMeans/median on the RGB pixels while everything else is
    perceptual. Half pure red and half pure blue therefore average to a purple that
    appears nowhere in the image and in neither cluster.
    """
    out = rs.extract_colors(_two_colour_image(), np.ones((4, 4), bool), n_colors=2, seed=42)
    assert out["mean_color"] == "#800080"
    assert out["median_color"] == "#800080"
    assert out["mean_color"] not in [c["hex_color"] for c in out["dominant_color_info"]]


def test_upstream_per_pixel_hex_reproduces_the_double_scaling_bug():
    """Reproduced defect, not endorsed: [0,1] pixels divided by 255 a second time."""
    pixels = np.array([[1.0, 1.0, 1.0], [0.5, 0.25, 0.0]])
    assert rs.upstream_pixel_hex(pixels) == ["#010101", "#010000"]


def test_extract_colors_raises_on_an_empty_mask():
    with pytest.raises(ValueError, match="Mask selects no pixels"):
        rs.extract_colors(_two_colour_image(), np.zeros((4, 4), bool))


def test_extract_colors_is_seed_reproducible():
    img = np.random.default_rng(0).random((16, 16, 3))
    mask = np.ones((16, 16), bool)
    a = rs.extract_colors(img, mask, n_colors=4, seed=7)["dominant_color_info"]
    b = rs.extract_colors(img, mask, n_colors=4, seed=7)["dominant_color_info"]
    assert a == b


# ---------------------------------------------- stage C+D+E together (part subtraction)

def _spider_scene():
    """A 'spider': a 4x4 abdomen block with a 4x2 leg strip overlapping its right edge."""
    img = np.zeros((8, 8, 3), float)
    img[2:6, 1:5] = rs.hex_to_rgb01("#FFD700")      # abdomen: yellow
    img[2:6, 4:6] = rs.hex_to_rgb01("#000000")      # legs: black, overlapping column 4
    abdomen = _rect(2, 6, 1, 5)
    legs = _rect(2, 6, 4, 6)
    return img, abdomen, legs


def test_part_subtraction_removes_the_legs_from_the_abdomen_roi():
    img, abdomen, legs = _spider_scene()
    dets = [
        _det(0.7, "the abdomen of a spider.", abdomen),
        _det(0.8, "the legs of a spider.", legs),
    ]
    out = rs.process_masks_and_extract_colors(
        img, dets,
        include_labels=["the abdomen of a spider."],
        exclude_labels=["the legs of a spider."],
        score_threshold=0.5, n_colors=1, seed=42,
    )
    final = out["final_mask"]
    assert final[2:6, 1:4].all()                 # abdomen minus the overlap survives
    assert not final[:, 4:].any()                # every leg pixel is gone
    assert final.sum() == 12
    assert out["color_info"]["dominant_color_info"][0]["hex_color"] == "#FFD700"


def test_without_subtraction_the_leg_colour_contaminates_the_palette():
    img, abdomen, legs = _spider_scene()
    dets = [_det(0.7, "the abdomen of a spider.", abdomen)]
    out = rs.process_masks_and_extract_colors(
        img, dets, include_labels=["the abdomen of a spider."],
        exclude_labels=None, score_threshold=0.5, n_colors=2, seed=42,
    )
    hexes = {c["hex_color"] for c in out["color_info"]["dominant_color_info"]}
    assert hexes == {"#FFD700", "#000000"}


def test_process_masks_scales_a_0_255_image_into_0_1():
    img, abdomen, legs = _spider_scene()
    dets = [_det(0.7, "roi", abdomen), _det(0.8, "legs", legs)]
    out = rs.process_masks_and_extract_colors(
        (img * 255), dets, include_labels=["roi"], exclude_labels=["legs"],
        score_threshold=0.5, n_colors=1, seed=42,
    )
    assert out["image"].max() <= 1.0
    assert out["color_info"]["dominant_color_info"][0]["hex_color"] == "#FFD700"


def test_process_masks_raises_when_subtraction_empties_the_roi():
    _, abdomen, _ = _spider_scene()
    img = np.zeros((8, 8, 3), float)
    dets = [_det(0.7, "roi", abdomen), _det(0.9, "legs", abdomen)]
    with pytest.raises(ValueError, match="empty after part subtraction"):
        rs.process_masks_and_extract_colors(
            img, dets, include_labels=["roi"], exclude_labels=["legs"],
            score_threshold=0.5, seed=42,
        )


def test_process_masks_raises_when_no_detection_clears_the_threshold():
    img, abdomen, _ = _spider_scene()
    with pytest.raises(ValueError, match="No masks meet the score threshold"):
        rs.process_masks_and_extract_colors(
            img, [_det(0.4, "roi", abdomen)], include_labels=["roi"],
            score_threshold=0.5, seed=42,
        )


# --------------------------------------------------- stage F: the JSON artefact

def test_detections_to_dicts_faithful_mode_crops_the_mask_to_the_image():
    det = _det(0.6, "roi", np.ones((10, 12), np.uint8), box=(1, 2, 3, 4))
    rec = rs.detections_to_dicts([det], img_height=8, img_width=6, faithful=True)[0]
    assert rec["label"] == "roi" and rec["score"] == pytest.approx(0.6)
    assert rec["box"] == {"xmin": 1, "ymin": 2, "xmax": 3, "ymax": 4}
    assert np.asarray(rec["mask"]).shape == (8, 6)


def test_detections_to_dicts_keeps_a_null_mask_null():
    rec = rs.detections_to_dicts([_det(0.6, "roi", None)], 8, 6, faithful=True)[0]
    assert rec["mask"] is None


def test_json_roundtrip_faithful_format():
    det = _det(0.6, "roi", _rect(1, 4, 1, 4), box=(1, 2, 3, 4))
    recs = rs.detections_to_dicts([det], H, W, faithful=True)
    back = rs.dicts_to_detections(json.loads(json.dumps(recs)), Path("."))
    np.testing.assert_array_equal(back[0].mask, det.mask)
    assert back[0].box.xyxy == [1, 2, 3, 4]


def test_json_roundtrip_via_mask_pngs(tmp_path):
    det = _det(0.6, "roi", _rect(1, 4, 1, 4), box=(1, 2, 3, 4))
    json_path = tmp_path / "argiope_aurantia__abc.json"
    rs.write_image_json(json_path, [det], H, W, faithful=False)

    recs = json.loads(json_path.read_text())
    assert recs[0]["mask"] == "masks/argiope_aurantia__abc_000.png"
    assert (tmp_path / recs[0]["mask"]).exists()

    back = rs.dicts_to_detections(recs, tmp_path)
    np.testing.assert_array_equal(back[0].mask > 0, det.mask > 0)


def test_dicts_to_detections_reports_a_missing_mask_png(tmp_path):
    recs = [{"score": 0.6, "label": "roi",
             "box": {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1},
             "mask": "masks/nope.png"}]
    with pytest.raises(FileNotFoundError):
        rs.dicts_to_detections(recs, tmp_path)


def test_from_json_reruns_colour_without_loading_a_model(tmp_path):
    """Definition of done #3, proven on synthetic data: no checkpoint is touched."""
    img, abdomen, legs = _spider_scene()
    img_path = tmp_path / "argiope_aurantia" / "abc.png"
    img_path.parent.mkdir(parents=True)
    Image.fromarray((img * 255).astype(np.uint8)).save(img_path)

    run_dir = tmp_path / "run"
    (run_dir / "json").mkdir(parents=True)
    rs.write_image_json(
        run_dir / "json" / "argiope_aurantia__abc.json",
        [_det(0.7, "the abdomen of a spider.", abdomen),
         _det(0.8, "the legs of a spider.", legs)],
        8, 8, faithful=False,
    )
    (run_dir / "run_config.json").write_text(json.dumps({
        "n_colors": 1, "seed": 42, "custom_colors": None, "score_threshold": 0.5,
        "include_labels": ["the abdomen of a spider."],
        "exclude_labels": ["the legs of a spider."],
        "image_list": [{"path": str(img_path), "species": "argiope_aurantia"}],
    }))

    out = rs.run_from_json(run_dir, n_colors=None, custom_colors=None, seed=None)

    assert out["models_loaded"] is False
    assert "transformers" not in sys.modules
    rows = (run_dir / "colors_from_json.csv").read_text().strip().splitlines()
    assert len(rows) == 2                                   # header + one cluster
    assert "#FFD700" in rows[1]                             # legs subtracted, yellow left


def test_from_json_can_re_analyse_with_a_different_palette(tmp_path):
    """The artefact's point: change n_colors/palette and re-run with no model."""
    img, abdomen, _ = _spider_scene()
    img_path = tmp_path / "argiope_aurantia" / "abc.png"
    img_path.parent.mkdir(parents=True)
    Image.fromarray((img * 255).astype(np.uint8)).save(img_path)
    run_dir = tmp_path / "run"
    (run_dir / "json").mkdir(parents=True)
    rs.write_image_json(
        run_dir / "json" / "argiope_aurantia__abc.json",
        [_det(0.7, "roi", abdomen)], 8, 8, faithful=False,
    )
    (run_dir / "run_config.json").write_text(json.dumps({
        "n_colors": 1, "seed": 42, "custom_colors": None, "score_threshold": 0.5,
        "include_labels": ["roi"], "exclude_labels": [],
        "image_list": [{"path": str(img_path), "species": "argiope_aurantia"}],
    }))

    rs.run_from_json(
        run_dir, n_colors=None, custom_colors=list(rs.ARGIOPE_PALETTE.values()), seed=None
    )

    rows = (run_dir / "colors_from_json.csv").read_text().strip().splitlines()
    assert len(rows) == 1 + len(rs.ARGIOPE_PALETTE)


# --------------------------------------------------------- stage G: QA and export

def test_recolour_by_cluster_replaces_masked_pixels_with_their_centroid():
    img = _two_colour_image()
    mask = np.ones((4, 4), bool)
    info = rs.extract_colors(img, mask, n_colors=2, seed=42)
    out = rs.recolour_by_cluster(img, mask, info)
    np.testing.assert_allclose(out[0, 0], [1, 0, 0], atol=1e-3)
    np.testing.assert_allclose(out[0, 3], [0, 0, 1], atol=1e-3)


def test_recolour_leaves_unmasked_pixels_white():
    img = _two_colour_image()
    mask = np.zeros((4, 4), bool)
    mask[:, :2] = True
    out = rs.recolour_by_cluster(img, mask, rs.extract_colors(img, mask, n_colors=1, seed=42))
    np.testing.assert_allclose(out[0, 3], [1, 1, 1])


def test_export_transparent_png_crops_to_the_mask_bounding_box(tmp_path):
    img = np.ones((8, 8, 3), float)
    written = rs.export_transparent_png(
        img, {"roi": _rect(2, 6, 1, 5) > 0}, tmp_path, "pfx", crop=True, remove_overlap=False
    )
    assert len(written) == 1
    arr = np.asarray(Image.open(written[0]))
    assert arr.shape == (4, 4, 4)
    assert (arr[..., 3] == 255).all()


def test_export_transparent_png_remove_overlap_erases_the_other_label(tmp_path):
    img = np.ones((8, 8, 3), float)
    masks = {"roi": _rect(0, 8, 0, 8) > 0, "legs": _rect(0, 8, 6, 8) > 0}
    written = rs.export_transparent_png(
        img, masks, tmp_path, "pfx", crop=False, remove_overlap=True
    )
    by_name = {p.name.split("_")[-1]: p for p in written}
    roi = np.asarray(Image.open(by_name["roi.png"]))
    assert (roi[:, :6, 3] == 255).all() and (roi[:, 6:, 3] == 0).all()


def test_export_transparent_png_skips_a_fully_overlapped_mask(tmp_path):
    img = np.ones((8, 8, 3), float)
    masks = {"roi": _rect(0, 8, 6, 8) > 0, "legs": _rect(0, 8, 0, 8) > 0}
    written = rs.export_transparent_png(
        img, masks, tmp_path, "pfx", crop=False, remove_overlap=True
    )
    assert [p.name for p in written] == ["pfx_001_legs.png"]


# ------------------------------------------------------- stage H: sampling and CSV

def _fake_gbif(tmp_path, counts):
    root = tmp_path / "gbif"
    for species, k in counts.items():
        d = root / species
        d.mkdir(parents=True)
        for i in range(k):
            (d / "img{:03d}.jpg".format(i)).write_bytes(b"")
    return root


def test_sample_images_is_species_stratified_and_seeded(tmp_path):
    root = _fake_gbif(tmp_path, {"argiope_argentata": 50, "argiope_aurantia": 50,
                                 "argiope_bruennichi": 50})
    a, _ = rs.sample_images(root, n=40, seed=42)
    b, _ = rs.sample_images(root, n=40, seed=42)
    c, _ = rs.sample_images(root, n=40, seed=7)

    assert len(a) == 40
    assert a == b                                            # same seed, same sample
    assert a != c                                            # different seed, different sample
    per_species = {}
    for _, sp in a:
        per_species[sp] = per_species.get(sp, 0) + 1
    assert sorted(per_species.values()) == [13, 13, 14]      # 40 split three ways


def test_sample_images_logs_when_a_species_is_short(tmp_path):
    root = _fake_gbif(tmp_path, {"argiope_argentata": 2, "argiope_aurantia": 50,
                                 "argiope_bruennichi": 50})
    chosen, notes = rs.sample_images(root, n=30, seed=42)
    assert len(chosen) == 22                                 # 2 + 10 + 10, never silent
    assert any("only 2 available" in n for n in notes)


def test_sample_images_reports_an_empty_root(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        rs.sample_images(tmp_path / "empty", n=4, seed=42)


def test_images_from_manifest_resolves_paths_and_logs_missing(tmp_path):
    real = tmp_path / "data" / "raw" / "gbif" / "argiope_aurantia" / "a.jpg"
    real.parent.mkdir(parents=True)
    real.write_bytes(b"")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"source_image": "data\\raw\\gbif\\argiope_aurantia\\a.jpg"}) + "\n"
        + json.dumps({"source_image": "data\\raw\\gbif\\argiope_aurantia\\gone.jpg"}) + "\n"
    )
    chosen, notes = rs.images_from_manifest(manifest, tmp_path, n=None)
    assert [p.name for p, _ in chosen] == ["a.jpg"]
    assert chosen[0][1] == "argiope_aurantia"
    assert any("missing on disk" in n for n in notes)


def test_colour_rows_carry_every_required_column():
    img = _two_colour_image()
    info = rs.extract_colors(img, np.ones((4, 4), bool), n_colors=2, seed=42)
    rows = rs.colour_rows("a.jpg", "argiope_aurantia", "final_roi", "roi", 0.7, 1, info, 16)
    assert len(rows) == 2
    for row in rows:
        assert set(row) == set(rs.COLORS_CSV_FIELDS)
    assert sum(r["cluster_size"] for r in rows) == 16
    assert rows[0]["mean_color"].startswith("#")


def test_resolve_custom_colors():
    assert rs.resolve_custom_colors(None) is None
    assert rs.resolve_custom_colors("argiope") == list(rs.ARGIOPE_PALETTE.values())
    assert rs.resolve_custom_colors("#FF0000, #00FF00") == ["#FF0000", "#00FF00"]


def test_the_reference_palette_is_the_four_argiope_colours():
    assert set(rs.ARGIOPE_PALETTE) == {"yellow", "black", "silver", "brown"}
