"""Map detections to geometric subclass atoms using saved schemes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from geo_refine.geometry import GeoFeatures, dist_bin, diag_bin, spatial_bin, subclass_id


def load_schemes(json_path: Path, dataset_key: str) -> dict[str, dict]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    block = data[dataset_key]
    return block["schemes"]


def atom_for_det(det: dict[str, Any], scheme: dict[str, Any] | None) -> str:
    """Return semantic atom for one detection given class scheme (or coarse label)."""
    lab = det["label"]
    if scheme is None:
        return lab
    geo = GeoFeatures(
        diag_angle_deg=float(det.get("diag_angle", 45.0)),
        spatial_angle_deg=float(det.get("spatial_angle", 0.0)),
        dist_ratio=float(det.get("dist_ratio", 0.0)),
        aspect=float(det.get("aspect", 1.0)),
        area_ratio=float(det.get("area_ratio", 0.0)),
    )
    sid = subclass_id(
        lab,
        geo,
        scheme["diag_n"],
        scheme["spatial_n"],
        scheme.get("dist_edges") or [],
        scheme["use_diag"],
        scheme["use_spatial"],
        scheme["use_dist"],
    )
    # Map sparse raw ids into OTHER if that was used in scheme counts
    counts = scheme.get("subclass_counts") or {}
    if sid not in counts and f"{lab}|OTHER" in counts:
        # any missing fine bin goes to OTHER bucket for coding stability
        return f"{lab}|OTHER"
    return sid


def attach_atoms(
    detections: dict[str, list[dict]],
    schemes: dict[str, dict] | None,
    use_geo: bool,
) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for path, dets in detections.items():
        rows = []
        for d in dets:
            dd = dict(d)
            sch = schemes.get(d["label"]) if (use_geo and schemes) else None
            dd["atom"] = atom_for_det(d, sch)
            rows.append(dd)
        out[path] = rows
    return out
