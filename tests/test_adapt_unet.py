"""Tests for the U-Net -> SegmentR adapter.

Everything here runs on synthetic arrays except the final end-to-end smoke, which needs the
trained checkpoint and is skipped when it is absent, so `pytest repro/segmentr` stays green
on a machine with no weights and no GPU.

The adapter's contract with the reproduction is also pinned here: importing it must not drag
in `argiope`, and `--from-json` must re-run colour with the segmenter never loaded.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import adapt_unet as ad  # noqa: E402
import repro_segmentr as rs  # noqa: E402

CKPT = ROOT.parents[1] / "checkpoints" / "opistho_unet.pt"
DATASET = ROOT.parents[1] / "data" / "interim" / "opistho_seg" / "images"
needs_weights = pytest.mark.skipif(
    not CKPT.exists() or not DATASET.exists(),
    reason="trained checkpoint / dataset not present (regenerate with argiope train-segmenter)",
)


def _rect_mask(h=20, w=30, r0=4, r1=12, c0=6, c1=18):
    m = np.zeros((h, w), bool)
    m[r0:r1, c0:c1] = True
    return m


# ------------------------------------------------------------------ isolation contract

def test_importing_the_adapter_does_not_import_argiope():
    """`argiope` must be lazy: --from-json and the tests stay free of torch model code."""
    code = (
        "import sys; sys.path.insert(0, r'%s'); import adapt_unet; "
        "print('argiope' in sys.modules)" % str(ROOT)
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


def test_adapter_does_not_use_the_stage_c_merge():
    """combine_masks ORs same-label detections and destroys per-specimen identity."""
    src = (ROOT / "adapt_unet.py").read_text(encoding="utf-8")
    body = src.split('"""', 2)[-1]           # ignore the module docstring, which explains why
    assert "combine_masks" not in body
    assert "merge_masks_by_label" not in body
    assert "process_masks_and_extract_colors" not in body


# ---------------------------------------------------------------------- mask -> detection

def test_mask_bounds_is_inclusive():
    assert ad.mask_bounds(_rect_mask()) == (6, 4, 17, 11)


def test_mask_bounds_raises_on_an_empty_mask():
    with pytest.raises(ValueError, match="mask is empty"):
        ad.mask_bounds(np.zeros((8, 8), bool))


def test_mask_bounds_spans_disjoint_blobs():
    m = np.zeros((20, 20), bool)
    m[2:4, 2:4] = True
    m[15:18, 16:19] = True
    assert ad.mask_bounds(m) == (2, 2, 18, 17)


def test_mask_to_detection_shape_matches_what_segmentr_expects():
    m = _rect_mask()
    det = ad.mask_to_detection(m, score=0.87)
    assert isinstance(det, rs.DetectionResult)
    assert det.label == "opisthosoma"
    assert det.score == pytest.approx(0.87)
    assert det.box.xyxy == [6, 4, 17, 11]
    assert det.mask.dtype == np.uint8
    np.testing.assert_array_equal(det.mask > 0, m)


def test_mask_to_detection_accepts_a_uint8_mask():
    det = ad.mask_to_detection(_rect_mask().astype(np.uint8) * 255, score=1.0)
    assert det.box.xyxy == [6, 4, 17, 11]
    assert det.mask.max() == 1


def test_mask_to_detection_rejects_an_empty_mask():
    with pytest.raises(ValueError, match="mask is empty"):
        ad.mask_to_detection(np.zeros((8, 8), bool))


def test_iou():
    a = _rect_mask()
    assert ad.iou(a, a) == pytest.approx(1.0)
    assert ad.iou(a, np.zeros_like(a)) == 0.0
    b = _rect_mask(r0=4, r1=12, c0=12, c1=24)          # half-overlapping
    assert 0.0 < ad.iou(a, b) < 1.0


# ------------------------------------------------------------------ colour on a known mask

def _two_colour_image(h=20, w=30):
    img = np.zeros((h, w, 3), float)
    img[:, : w // 2] = rs.hex_to_rgb01("#F2C554")      # abdomen yellow
    img[:, w // 2 :] = rs.hex_to_rgb01("#231F23")      # abdomen black
    return img


def test_extract_colors_over_the_adapter_mask_returns_the_masked_colours_only():
    img = _two_colour_image()
    mask = np.zeros((20, 30), bool)
    mask[4:12, 6:14] = True                            # entirely inside the yellow half
    info = rs.extract_colors(img, mask, n_colors=1, seed=42)
    assert info["dominant_color_info"][0]["hex_color"] == "#F2C554"
    assert info["dominant_color_info"][0]["cluster_size"] == 64


def test_extract_colors_recovers_both_bands_when_the_mask_spans_them():
    img = _two_colour_image()
    mask = np.zeros((20, 30), bool)
    mask[4:12, 11:19] = True                           # straddles the boundary at x=15
    info = rs.extract_colors(img, mask, n_colors=2, seed=42)
    assert sorted(c["hex_color"] for c in info["dominant_color_info"]) == ["#231F23", "#F2C554"]
    assert sum(c["cluster_size"] for c in info["dominant_color_info"]) == 64


def test_colour_rows_carry_the_adapter_label():
    img = _two_colour_image()
    mask = np.ones((20, 30), bool)
    info = rs.extract_colors(img, mask, n_colors=2, seed=42)
    rows = rs.colour_rows("a.jpg", "images", "opisthosoma", "opisthosoma", 0.9, 1, info, 600)
    assert len(rows) == 2
    for r in rows:
        assert set(r) == set(rs.COLORS_CSV_FIELDS)
        assert r["role"] == "opisthosoma" and r["label"] == "opisthosoma"


# --------------------------------------------------------------- stage F round trip

def test_json_round_trip_preserves_the_adapter_mask(tmp_path):
    m = _rect_mask()
    det = ad.mask_to_detection(m, score=0.66)
    jp = tmp_path / "images__abc.json"
    rs.write_image_json(jp, [det], 20, 30, faithful=False)

    recs = json.loads(jp.read_text())
    assert recs[0]["label"] == "opisthosoma"
    assert recs[0]["mask"] == "masks/images__abc_000.png"

    back = rs.dicts_to_detections(recs, tmp_path)
    np.testing.assert_array_equal(np.asarray(back[0].mask) > 0, m)
    assert back[0].box.xyxy == det.box.xyxy


def _fake_run(tmp_path, hexes=("#F2C554",)):
    """A finished adapter run: one image, one mask PNG, one run_config."""
    img = _two_colour_image()
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img_path = img_dir / "abc.png"                     # PNG: JPEG would shift the colours
    Image.fromarray((img * 255).astype(np.uint8)).save(img_path)

    run_dir = tmp_path / "run"
    (run_dir / "json").mkdir(parents=True)
    mask = np.zeros((20, 30), bool)
    mask[4:12, 6:14] = True
    rs.write_image_json(run_dir / "json" / "images__abc.json",
                        [ad.mask_to_detection(mask, 0.9)], 20, 30, faithful=False)
    (run_dir / "run_config.json").write_text(json.dumps({
        "n_colors": 1, "seed": 42, "custom_colors": None, "label": "opisthosoma",
        "image_list": [{"path": str(img_path), "group": "images"}],
    }))
    return run_dir


def test_from_json_reruns_colour_without_loading_the_unet(tmp_path):
    run_dir = _fake_run(tmp_path)
    out = ad.run_from_json(run_dir, n_colors=None, custom_colors=None, seed=None)
    assert out["model_loaded"] is False
    assert "argiope.segmentation.unet_backend" not in sys.modules
    rows = (run_dir / "colors_from_json.csv").read_text().strip().splitlines()
    assert len(rows) == 2                              # header + one cluster
    assert "#F2C554" in rows[1]


def test_from_json_can_re_analyse_with_a_reference_palette(tmp_path):
    run_dir = _fake_run(tmp_path)
    ad.run_from_json(run_dir, n_colors=None,
                     custom_colors=list(rs.ARGIOPE_PALETTE.values()), seed=None)
    rows = (run_dir / "colors_from_json.csv").read_text().strip().splitlines()
    assert len(rows) == 1 + len(rs.ARGIOPE_PALETTE)


# ------------------------------------------------------------------------ image collection

def test_collect_images_recurses_and_labels_by_parent_directory(tmp_path):
    for sub, names in (("a", ["1.jpg", "2.png"]), ("b", ["3.jpeg"])):
        d = tmp_path / sub
        d.mkdir()
        for n in names:
            (d / n).write_bytes(b"")
    (tmp_path / "a" / "notes.txt").write_bytes(b"")
    got = ad.collect_images(tmp_path)
    assert [p.name for p, _ in got] == ["1.jpg", "2.png", "3.jpeg"]
    assert {g for _, g in got} == {"a", "b"}


def test_collect_images_sampling_is_seeded(tmp_path):
    for i in range(12):
        (tmp_path / f"img{i:02d}.jpg").write_bytes(b"")
    a = ad.collect_images(tmp_path, n=5, seed=42)
    b = ad.collect_images(tmp_path, n=5, seed=42)
    c = ad.collect_images(tmp_path, n=5, seed=7)
    assert len(a) == 5 and a == b and a != c


def test_collect_images_accepts_a_single_file(tmp_path):
    f = tmp_path / "one.jpg"
    f.write_bytes(b"")
    assert ad.collect_images(f) == [(f, tmp_path.name)]


# ------------------------------------------------------------------------ end to end

@needs_weights
def test_end_to_end_smoke_over_real_images(tmp_path):
    """Definition of done: the adapter runs the real U-Net and emits SegmentR's artefacts."""
    rc = ad.main([
        "--images", str(DATASET), "--n", "2", "--seed", "42",
        "--out", str(tmp_path), "--run-id", "smoke",
    ])
    assert rc == 0
    run = tmp_path / "smoke"
    for name in ("colors.csv", "skipped.csv", "run_config.json", "summary.json"):
        assert (run / name).exists(), name

    summary = json.loads((run / "summary.json").read_text())
    assert summary["processed"] == 2
    assert summary["skipped"] == 0
    assert summary["colour_rows"] == 2 * 5                 # 2 images x n_colors

    assert len(list((run / "json").glob("*.json"))) == 2
    assert len(list((run / "json" / "masks").glob("*.png"))) == 2
    assert len(list((run / "qa").glob("*_qa.png"))) == 2
    assert len(list((run / "cutouts").glob("*.png"))) == 2

    header, *rows = (run / "colors.csv").read_text(encoding="utf-8").strip().splitlines()
    assert header.split(",") == rs.COLORS_CSV_FIELDS
    assert all(r.split(",")[3] == "opisthosoma" for r in rows)


@needs_weights
def test_from_json_reproduces_the_live_run_without_the_model(tmp_path):
    """Stage F over a real run: identical colour rows, segmenter never loaded."""
    assert ad.main(["--images", str(DATASET), "--n", "2", "--seed", "42",
                    "--out", str(tmp_path), "--run-id", "reload", "--no-qa"]) == 0
    run = tmp_path / "reload"
    live = (run / "colors.csv").read_text(encoding="utf-8").strip().splitlines()

    code = (
        "import sys; sys.path.insert(0, r'%s'); import adapt_unet as ad; "
        "out = ad.run_from_json(r'%s', None, None, None); "
        "print('LOADED', out['model_loaded'])" % (str(ROOT), str(run))
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "LOADED False" in proc.stdout

    reloaded = (run / "colors_from_json.csv").read_text(encoding="utf-8").strip().splitlines()
    assert reloaded == live
