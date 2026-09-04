"""Standalone copy of Argiope's trained U-Net segmenter.

This is a VERBATIM copy of the class from the Argiope project:

    src/argiope/segmentation/unet_backend.py   (UnetSegmenter, _mask_bbox, _largest_component)
    src/argiope/segmentation/base.py           (MaskResult)

It exists so a self-contained bundle can run without the `argiope` package installed.
`adapt_unet.py` imports the real one first and only falls back to this. Being a copy, it can
drift: if the backend changes upstream, re-copy it rather than editing here.

Requires: torch, segmentation-models-pytorch, opencv-python, numpy, scipy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass
class MaskResult:
    mask: np.ndarray                  # bool/uint8 HxW, True = opisthosoma
    score: float                      # model confidence in [0, 1]
    bbox: tuple[int, int, int, int]   # x, y, w, h of the mask in image coords
    backend: str                      # provenance


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return (0, 0, 0, 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def _largest_component(mask: np.ndarray, fill_holes: bool = True) -> np.ndarray:
    """Keep only the largest connected component (+ fill its holes)."""
    import cv2

    m = np.squeeze(np.asarray(mask)).astype(np.uint8)
    if m.sum() == 0:
        return m.astype(bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 2:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (labels == largest).astype(np.uint8)
    if fill_holes:
        from scipy import ndimage

        m = ndimage.binary_fill_holes(m).astype(np.uint8)
    return m.astype(bool)


class UnetSegmenter:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.weights = cfg.get("unet_weights", "checkpoints/opistho_unet.pt")
        self.encoder = cfg.get("unet_encoder", "resnet34")
        self.imgsz = int(cfg.get("unet_imgsz", 512))
        self.threshold = float(cfg.get("unet_threshold", 0.5))
        self.tta = bool(cfg.get("unet_tta", True))
        self.postprocess = bool(cfg.get("unet_postprocess", True))
        self.device = cfg.get("device", "cuda")
        self._model = None

    def load(self) -> "UnetSegmenter":
        import segmentation_models_pytorch as smp
        import torch

        ckpt = torch.load(self.weights, map_location=self.device, weights_only=False)
        self.encoder = ckpt.get("encoder", self.encoder)
        self.imgsz = int(ckpt.get("imgsz", self.imgsz))
        model = smp.Unet(encoder_name=self.encoder, encoder_weights=None, in_channels=3, classes=1)
        model.load_state_dict(ckpt["state_dict"])
        self._model = model.eval().to(self.device)
        return self

    def _infer_prob(self, image: np.ndarray) -> np.ndarray:
        import cv2
        import torch

        h, w = image.shape[:2]
        arr = (cv2.resize(image, (self.imgsz, self.imgsz)) / 255.0 - _MEAN) / _STD
        arr = np.ascontiguousarray(arr.transpose(2, 0, 1))
        x = torch.from_numpy(arr)[None].float().to(self.device)
        with torch.no_grad():
            prob = torch.sigmoid(self._model(x))[0, 0].float().cpu().numpy()
        return cv2.resize(prob, (w, h))

    def segment(self, image: np.ndarray, prompt: dict[str, Any] | None = None) -> MaskResult:
        """Segment the opisthosoma from an RGB uint8 image. ``prompt`` is ignored."""
        if self._model is None:
            self.load()
        prob = self._infer_prob(image)
        if self.tta:
            hf = self._infer_prob(np.ascontiguousarray(image[:, ::-1]))[:, ::-1]
            vf = self._infer_prob(np.ascontiguousarray(image[::-1]))[::-1]
            prob = (prob + hf + vf) / 3.0
        mask = prob >= self.threshold
        if self.postprocess:
            mask = _largest_component(mask)
        score = float(prob[mask].mean()) if mask.any() else 0.0
        return MaskResult(mask, score, _mask_bbox(mask), f"unet:{Path(self.weights).name}")
