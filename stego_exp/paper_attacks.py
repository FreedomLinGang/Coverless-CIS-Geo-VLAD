"""Lin & Liang TMM 2026 Table IX attack suite (exact parameter names/values).

Geometry:
  Center cropping 5/10/20%; Edge cropping 5/10/20%;
  Rotation 10/30/50°; Translation (36,20)/(40,25)/(80,50);
  Scaling ×3 (resize up then back).

Non-geometry:
  Gauss noise μ=0,σ=0.001; Salt&Pepper density=0.001; Speckle σ=0.01;
  Median/Mean/Gauss filter 3×3; JPEG Q=90; HistEq; Gamma 0.8.
"""
from __future__ import annotations

import io
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps


def _to_float01(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def _from_float01(arr: np.ndarray) -> Image.Image:
    a = np.clip(arr, 0.0, 1.0)
    return Image.fromarray((a * 255.0 + 0.5).astype(np.uint8))


def center_crop(img: Image.Image, ratio: float) -> Image.Image:
    """Keep central (1-ratio) region, resize back to original size."""
    w, h = img.size
    keep_w = max(1, int(round(w * (1.0 - ratio))))
    keep_h = max(1, int(round(h * (1.0 - ratio))))
    left = (w - keep_w) // 2
    top = (h - keep_h) // 2
    cropped = img.crop((left, top, left + keep_w, top + keep_h))
    return cropped.resize((w, h), Image.BILINEAR)


def edge_crop(img: Image.Image, ratio: float) -> Image.Image:
    """Remove `ratio` from EACH edge (keep (1-2r)), resize back.

    Distinct from center-crop in Table IX (edge is more aggressive).
    """
    w, h = img.size
    dx = int(round(w * ratio))
    dy = int(round(h * ratio))
    if 2 * dx >= w or 2 * dy >= h:
        dx, dy = max(0, w // 4), max(0, h // 4)
    cropped = img.crop((dx, dy, w - dx, h - dy))
    return cropped.resize((w, h), Image.BILINEAR)


def rotate(img: Image.Image, angle: float) -> Image.Image:
    return img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))


def translate_px(img: Image.Image, dx: int, dy: int) -> Image.Image:
    arr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = arr.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    out = cv2.warpAffine(arr, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


def scale_times(img: Image.Image, factor: float = 3.0) -> Image.Image:
    """Scaling ratio = 3 times: enlarge by factor then restore original size."""
    w, h = img.size
    big = img.resize((max(1, int(w * factor)), max(1, int(h * factor))), Image.BILINEAR)
    return big.resize((w, h), Image.BILINEAR)


def gauss_noise(img: Image.Image, sigma: float = 0.001) -> Image.Image:
    arr = _to_float01(img)
    arr = arr + np.random.normal(0.0, sigma, arr.shape).astype(np.float32)
    return _from_float01(arr)


def salt_pepper(img: Image.Image, density: float = 0.001) -> Image.Image:
    """Table IX lists μ=0,σ=0.001 for S&P → treat σ as noise density."""
    arr = np.asarray(img.convert("RGB")).copy()
    h, w = arr.shape[:2]
    mask = np.random.random((h, w))
    arr[mask < density / 2.0] = 0
    arr[mask > 1.0 - density / 2.0] = 255
    return Image.fromarray(arr)


def speckle_noise(img: Image.Image, sigma: float = 0.01) -> Image.Image:
    arr = _to_float01(img)
    noise = np.random.normal(0.0, sigma, arr.shape).astype(np.float32)
    return _from_float01(arr + arr * noise)


def median_filter(img: Image.Image, ksize: int = 3) -> Image.Image:
    return img.filter(ImageFilter.MedianFilter(size=ksize))


def mean_filter(img: Image.Image, ksize: int = 3) -> Image.Image:
    arr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    out = cv2.blur(arr, (ksize, ksize))
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


def gauss_filter(img: Image.Image, ksize: int = 3) -> Image.Image:
    arr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    out = cv2.GaussianBlur(arr, (ksize, ksize), 0)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


def jpeg_compress(img: Image.Image, quality: int = 90) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def hist_eq(img: Image.Image) -> Image.Image:
    return ImageOps.equalize(img.convert("RGB"))


def gamma_correct(img: Image.Image, gamma: float = 0.8) -> Image.Image:
    arr = _to_float01(img)
    return _from_float01(np.power(arr, gamma))


# Ordered exactly like Table IX rows (geometry then non-geometry).
PAPER_TABLE_IX_ATTACKS: list[tuple[str, Callable[[Image.Image], Image.Image]]] = [
    ("center_crop_5", lambda im: center_crop(im, 0.05)),
    ("center_crop_10", lambda im: center_crop(im, 0.10)),
    ("center_crop_20", lambda im: center_crop(im, 0.20)),
    ("edge_crop_5", lambda im: edge_crop(im, 0.05)),
    ("edge_crop_10", lambda im: edge_crop(im, 0.10)),
    ("edge_crop_20", lambda im: edge_crop(im, 0.20)),
    ("rotate_10", lambda im: rotate(im, 10.0)),
    ("rotate_30", lambda im: rotate(im, 30.0)),
    ("rotate_50", lambda im: rotate(im, 50.0)),
    ("translate_36_20", lambda im: translate_px(im, 36, 20)),
    ("translate_40_25", lambda im: translate_px(im, 40, 25)),
    ("translate_80_50", lambda im: translate_px(im, 80, 50)),
    ("scaling_x3", lambda im: scale_times(im, 3.0)),
    ("gauss_noise", lambda im: gauss_noise(im, 0.001)),
    ("salt_pepper", lambda im: salt_pepper(im, 0.001)),
    ("speckle_noise", lambda im: speckle_noise(im, 0.01)),
    ("median_filter", lambda im: median_filter(im, 3)),
    ("mean_filter", lambda im: mean_filter(im, 3)),
    ("gauss_filter", lambda im: gauss_filter(im, 3)),
    ("jpeg_90", lambda im: jpeg_compress(im, 90)),
    ("hist_eq", hist_eq),
    ("gamma_0.8", lambda im: gamma_correct(im, 0.8)),
]

PAPER_ATTACK_NAMES = [n for n, _ in PAPER_TABLE_IX_ATTACKS]
_ATTACK_FN = {n: fn for n, fn in PAPER_TABLE_IX_ATTACKS}


def apply_paper_attack(img: Image.Image, name: str) -> Image.Image:
    if name not in _ATTACK_FN:
        raise ValueError(f"Unknown paper attack: {name}")
    return _ATTACK_FN[name](img)


ATTACK_DISPLAY = {
    "center_crop_5": "Center Cropping 5%",
    "center_crop_10": "Center Cropping 10%",
    "center_crop_20": "Center Cropping 20%",
    "edge_crop_5": "Edge Cropping 5%",
    "edge_crop_10": "Edge Cropping 10%",
    "edge_crop_20": "Edge Cropping 20%",
    "rotate_10": "Rotation 10°",
    "rotate_30": "Rotation 30°",
    "rotate_50": "Rotation 50°",
    "translate_36_20": "Translation (36, 20)",
    "translate_40_25": "Translation (40, 25)",
    "translate_80_50": "Translation (80, 50)",
    "scaling_x3": "Scaling ×3",
    "gauss_noise": "Gauss Noise (σ=0.001)",
    "salt_pepper": "Salt & Pepper (d=0.001)",
    "speckle_noise": "Speckle Noise (σ=0.01)",
    "median_filter": "Median Filter 3×3",
    "mean_filter": "Mean Filter 3×3",
    "gauss_filter": "Gauss Filter 3×3",
    "jpeg_90": "JPEG Compression Q=90",
    "hist_eq": "Histogram Equalization",
    "gamma_0.8": "Gamma Correction 0.8",
}
