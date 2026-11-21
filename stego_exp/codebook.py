"""Object sequence codebook with long-tail balancing and optional attributes."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
from PIL import Image


COLOR_BINS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
PERSON_SHAPES = ["tall", "square", "wide"]


def _quantize_color(img: Image.Image, box: list[float]) -> str:
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.width, x2), min(img.height, y2)
    if x2 <= x1 or y2 <= y1:
        return "C0"
    crop = img.crop((x1, y1, x2, y2)).resize((16, 16))
    arr = np.asarray(crop).astype(np.float32)
    mean = arr.reshape(-1, 3).mean(axis=0)
    # 8 bins by brightness + rg
    brightness = mean.mean()
    rg = mean[0] - mean[1]
    b_idx = 0 if brightness < 85 else (1 if brightness < 170 else 2)
    c_idx = 0 if rg < -20 else (1 if rg < 20 else 2)
    idx = min(7, b_idx * 3 + min(2, c_idx))
    return COLOR_BINS[idx]


def _person_shape(aspect: float) -> str:
    if aspect >= 1.4:
        return "tall"
    if aspect <= 0.75:
        return "wide"
    return "square"


def semantic_atom(obj: dict[str, Any], img: Image.Image | None, use_attributes: bool) -> str:
    # Prefer precomputed geometric / fine atom if present
    if obj.get("atom"):
        return str(obj["atom"])
    label = obj["label"]
    if not use_attributes:
        return label
    color = _quantize_color(img, obj["box"]) if img is not None else "C0"
    if label == "person":
        return f"{label}|{_person_shape(obj.get('aspect', 1.0))}|{color}"
    return f"{label}|{color}"


def build_codebook(
    detections: dict[str, list[dict]],
    segment_bits: int,
    use_attributes: bool,
    image_loader=None,
) -> dict[str, str]:
    """Map semantic atom -> binary code of length segment_bits.

    Long-tail balancing: group atoms so that image coverage per code is similar
    (paper groups rare labels together).
    """
    n_codes = 2**segment_bits
    # Count images containing each atom
    atom_images: dict[str, set[str]] = defaultdict(set)
    for path, objs in detections.items():
        img = image_loader(path) if (use_attributes and image_loader) else None
        atoms = set()
        for o in objs:
            atoms.add(semantic_atom(o, img, use_attributes))
        for a in atoms:
            atom_images[a].add(path)

    # Sort atoms by coverage ascending (rare first), then pack into groups
    atoms_sorted = sorted(atom_images.keys(), key=lambda a: (len(atom_images[a]), a))
    groups: list[list[str]] = [[] for _ in range(n_codes)]
    group_cover = [0] * n_codes

    for atom in atoms_sorted:
        # Assign to currently least-covered group
        g = int(np.argmin(group_cover))
        groups[g].append(atom)
        group_cover[g] += len(atom_images[atom])

    codebook = {}
    for g, atoms in enumerate(groups):
        code = format(g, f"0{segment_bits}b")
        for atom in atoms:
            codebook[atom] = code
    return codebook


def select_objects_proposed(
    objs: list[dict],
    area_ratio_thresh: float,
    max_objects: int,
) -> list[dict]:
    chosen = [o for o in objs if o["area_ratio"] >= area_ratio_thresh]
    if not chosen:
        chosen = objs[:1] if objs else []
    return chosen[:max_objects]
