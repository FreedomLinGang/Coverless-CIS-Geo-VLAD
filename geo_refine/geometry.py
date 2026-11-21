"""Geometric fine-grained features from axis-aligned detection boxes."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class GeoFeatures:
    diag_angle_deg: float  # BL→TR vs bottom edge, in (0, 90]
    spatial_angle_deg: float  # image-center→box-center vs +x, in [0, 360)
    dist_ratio: float  # |center_box - center_img| / (0.5 * image_diagonal)
    aspect: float  # h/w
    area_ratio: float


def box_geometry(box: list[float], img_w: float, img_h: float) -> GeoFeatures:
    x1, y1, x2, y2 = box
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    # 1) diagonal BL→TR vs bottom horizontal: angle = atan(h/w) ∈ (0, 90°]
    diag_angle = math.degrees(math.atan2(h, w))
    # 2) box center vs image center
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    icx, icy = img_w * 0.5, img_h * 0.5
    ang = math.degrees(math.atan2(-(cy - icy), cx - icx))  # image y down → flip for math
    # atan2 returns [-180,180]; map to [0,360)
    if ang < 0:
        ang += 360.0
    dist = math.hypot(cx - icx, cy - icy)
    half_diag = 0.5 * math.hypot(img_w, img_h)
    dist_ratio = dist / max(1e-6, half_diag)
    return GeoFeatures(
        diag_angle_deg=diag_angle,
        spatial_angle_deg=ang,
        dist_ratio=dist_ratio,
        aspect=h / w,
        area_ratio=(w * h) / max(1.0, img_w * img_h),
    )


def equal_bin(value: float, lo: float, hi: float, n_bins: int) -> int:
    if n_bins <= 1:
        return 0
    v = min(max(value, lo), hi - 1e-9)
    width = (hi - lo) / n_bins
    return int((v - lo) / width)


def diag_bin(angle_deg: float, n: int = 4) -> int:
    return equal_bin(angle_deg, 0.0, 90.0, n)


def spatial_bin(angle_deg: float, n: int = 8) -> int:
    return equal_bin(angle_deg, 0.0, 360.0, n)


def dist_bin(ratio: float, edges: list[float]) -> int:
    """edges ascending, e.g. [0.25, 0.5, 0.75] → 4 bins."""
    for i, e in enumerate(edges):
        if ratio < e:
            return i
    return len(edges)


def subclass_id(
    label: str,
    geo: GeoFeatures,
    diag_n: int,
    spatial_n: int,
    dist_edges: list[float],
    use_diag: bool = True,
    use_spatial: bool = True,
    use_dist: bool = True,
) -> str:
    parts = [label]
    if use_diag:
        parts.append(f"D{diag_bin(geo.diag_angle_deg, diag_n)}")
    if use_spatial:
        parts.append(f"S{spatial_bin(geo.spatial_angle_deg, spatial_n)}")
    if use_dist:
        parts.append(f"R{dist_bin(geo.dist_ratio, dist_edges)}")
    return "|".join(parts)


def detection_record(
    label: str,
    score: float,
    box: list[float],
    img_w: float,
    img_h: float,
) -> dict[str, Any]:
    geo = box_geometry(box, img_w, img_h)
    return {
        "label": label,
        "score": float(score),
        "box": [float(x) for x in box],
        "diag_angle": geo.diag_angle_deg,
        "spatial_angle": geo.spatial_angle_deg,
        "dist_ratio": geo.dist_ratio,
        "aspect": geo.aspect,
        "area_ratio": geo.area_ratio,
        "img_w": img_w,
        "img_h": img_h,
    }
