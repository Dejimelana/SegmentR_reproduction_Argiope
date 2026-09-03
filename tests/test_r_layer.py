"""The R layer, exercised through Rscript.

The R wrapper's whole job is to shell out to `argiope describe` and turn the JSON contract
into data frames, so the only test worth writing runs actual R. Skipped when Rscript,
jsonlite, the argiope executable or the checkpoint is missing, which keeps
`pytest repro/segmentr` green on a machine that has none of them.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARGIOPE_ROOT = ROOT.parents[1]
R_SRC = ROOT / "R" / "argiope.R"
CKPT = ARGIOPE_ROOT / "checkpoints" / "opistho_unet.pt"
DATASET = ARGIOPE_ROOT / "data" / "interim" / "opistho_seg" / "images"


def _rscript():
    exe = shutil.which("Rscript")
    if exe:
        return exe
    for c in Path("C:/Program Files/R").glob("R-*/bin/Rscript.exe"):
        return str(c)
    return None


def _argiope_exe():
    exe = shutil.which("argiope")
    if exe:
        return exe
    cand = Path(sys.executable).parent / "Scripts" / "argiope.exe"
    return str(cand) if cand.exists() else None


RSCRIPT = _rscript()
ARGIOPE = _argiope_exe()


def _has_jsonlite():
    if not RSCRIPT:
        return False
    out = subprocess.run(
        [RSCRIPT, "-e", 'cat(requireNamespace("jsonlite", quietly=TRUE))'],
        capture_output=True, text=True,
    )
    return out.stdout.strip().endswith("TRUE")


needs_r = pytest.mark.skipif(
    not RSCRIPT or not ARGIOPE or not CKPT.exists() or not DATASET.exists()
    or not _has_jsonlite(),
    reason="needs Rscript + jsonlite + the argiope executable + the trained checkpoint",
)


def _run_r(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([RSCRIPT, script, *args], capture_output=True, text=True,
                          cwd=str(ARGIOPE_ROOT))


def test_r_source_parses_without_the_executable():
    """Sourcing the wrapper must not require Python, R packages or a model."""
    if not RSCRIPT:
        pytest.skip("Rscript not installed")
    script = (f'source("{R_SRC.as_posix()}"); '
              'cat(exists("argiope_describe"), exists("argiope_palette"))')
    out = subprocess.run([RSCRIPT, "-e", script], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "TRUE TRUE" in out.stdout


def test_argiope_executable_error_is_actionable(tmp_path):
    """The usual failure is R launched outside the conda env; say so, don't just fail."""
    if not RSCRIPT:
        pytest.skip("Rscript not installed")
    script = tmp_path / "s.R"
    script.write_text(
        f'source("{R_SRC.as_posix()}")\n'
        'options(argiope.executable = NULL)\n'
        'Sys.setenv(ARGIOPE_EXE = "")\n'
        'old <- Sys.getenv("PATH"); Sys.setenv(PATH = "")\n'
        'r <- tryCatch(argiope_executable(), error = function(e) conditionMessage(e))\n'
        'Sys.setenv(PATH = old)\n'
        'cat(r)\n',
        encoding="utf-8",
    )
    out = subprocess.run([RSCRIPT, str(script)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "argiope.executable" in out.stdout


@needs_r
def test_r_describe_returns_a_palette_table(tmp_path):
    """End to end: R -> argiope describe -> JSON -> data.frame."""
    images = sorted(DATASET.glob("*.jpg"))
    assert images, "no images in the dataset"

    script = tmp_path / "describe.R"
    script.write_text(
        f'source("{R_SRC.as_posix()}")\n'
        f'options(argiope.executable = "{Path(ARGIOPE).as_posix()}")\n'
        'args <- commandArgs(trailingOnly = TRUE)\n'
        'ok <- FALSE\n'
        'for (img in args) {\n'
        '  d <- argiope_describe(img)\n'
        '  pal <- argiope_palette(d)\n'
        '  o <- argiope_opisthosoma(d)\n'
        '  stopifnot(is.data.frame(pal), is.data.frame(o), nrow(o) == 1L)\n'
        '  stopifnot(identical(colnames(pal),\n'
        '            c("hex","name","coverage","ci_low","ci_high","delta_e")))\n'
        '  if (nrow(pal) > 0L) {\n'
        '    stopifnot(all(grepl("^#[0-9A-Fa-f]{6}$", pal$hex)))\n'
        '    stopifnot(all(pal$coverage >= 0 & pal$coverage <= 1))\n'
        '    stopifnot(file.exists(d$mask_path))\n'
        '    ok <- TRUE\n'
        '  }\n'
        '}\n'
        'cat(if (ok) "PALETTE_OK" else "NO_PALETTE")\n',
        encoding="utf-8",
    )
    # a few images: the U-Net legitimately returns an empty mask on some, and an empty
    # palette is a valid outcome rather than a failure
    out = _run_r(str(script), *[str(p) for p in images[:4]])
    assert out.returncode == 0, out.stdout + out.stderr
    assert "PALETTE_OK" in out.stdout, out.stdout


@needs_r
def test_r_batch_collects_failures_instead_of_aborting(tmp_path):
    """A broken path must produce an `error` row, not abandon the batch."""
    images = sorted(DATASET.glob("*.jpg"))[:2]
    script = tmp_path / "batch.R"
    script.write_text(
        f'source("{R_SRC.as_posix()}")\n'
        f'options(argiope.executable = "{Path(ARGIOPE).as_posix()}")\n'
        'args <- commandArgs(trailingOnly = TRUE)\n'
        'df <- argiope_describe_batch(c(args, "does_not_exist.jpg"), verbose = FALSE)\n'
        'stopifnot(is.data.frame(df))\n'
        'stopifnot("error" %in% colnames(df))\n'
        'stopifnot(sum(!is.na(df$error)) == 1L)\n'
        'cat("BATCH_OK", nrow(df))\n',
        encoding="utf-8",
    )
    out = _run_r(str(script), *[str(p) for p in images])
    assert out.returncode == 0, out.stdout + out.stderr
    assert "BATCH_OK" in out.stdout, out.stdout
