"""Adaptive geometric subclass scheme with experimental balance search.

Dimensions (from axis-aligned boxes):
  1) diag angle: atan(h/w) ∈ (0,90°]  — ref 4×22.5°
  2) spatial angle: img-center→box-center ∈ [0,360) — ref 8×45°
  3) dist_ratio: |c_box-c_img| / (0.5*img_diag) — quantile bins

For each class, grid-search (diag_n, spatial_n, dist_n) under a count budget,
score by usable density + balance (low CV), then merge residual into |OTHER.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .geometry import GeoFeatures, subclass_id


@dataclass
class ClassScheme:
    label: str
    count: int
    share: float
    tier: str
    diag_n: int
    spatial_n: int
    dist_n: int
    dist_edges: list[float]
    use_diag: bool
    use_spatial: bool
    use_dist: bool
    budget: int
    n_subclasses_raw: int
    n_subclasses_merged: int
    subclass_counts: dict[str, int]
    sparse_raw: int
    balance_cv: float
    score: float


def quantile_edges(values: list[float], n_bins: int) -> list[float]:
    if n_bins <= 1 or len(values) < max(4, n_bins):
        return []
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = [float(x) for x in np.quantile(values, qs)]
    out: list[float] = []
    last = -1.0
    for e in edges:
        if e <= last:
            e = last + 1e-4
        out.append(e)
        last = e
    return out


def _cv(counts: list[int]) -> float:
    if len(counts) <= 1:
        return 0.0
    arr = np.asarray(counts, dtype=np.float64)
    m = float(arr.mean())
    if m <= 1e-9:
        return 0.0
    return float(arr.std() / m)


def merge_sparse(counter: Counter, min_count: int) -> Counter:
    by_label: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for k, v in counter.items():
        lab = k.split("|", 1)[0]
        by_label[lab].append((k, v))
    out: Counter = Counter()
    for lab, items in by_label.items():
        keep, other = [], 0
        for k, v in items:
            if v >= min_count:
                keep.append((k, v))
            else:
                other += v
        for k, v in keep:
            out[k] += v
        if other:
            if not keep:
                items_sorted = sorted(items, key=lambda x: -x[1])
                rem = 0
                for i, (k, v) in enumerate(items_sorted):
                    if i < max(1, len(items_sorted) // 2) and v >= max(5, min_count // 3):
                        out[k] += v
                    else:
                        rem += v
                if rem:
                    out[f"{lab}|OTHER"] += rem
            else:
                out[f"{lab}|OTHER"] += other
    return out


def _assign_counts(
    dets: list[dict],
    lab: str,
    diag_n: int,
    spatial_n: int,
    dist_edges: list[float],
    use_diag: bool,
    use_spatial: bool,
    use_dist: bool,
) -> Counter:
    raw = Counter()
    for d in dets:
        if d["label"] != lab:
            continue
        geo = GeoFeatures(
            diag_angle_deg=d["diag_angle"],
            spatial_angle_deg=d["spatial_angle"],
            dist_ratio=d["dist_ratio"],
            aspect=d["aspect"],
            area_ratio=d["area_ratio"],
        )
        sid = subclass_id(
            lab, geo, diag_n, spatial_n, dist_edges, use_diag, use_spatial, use_dist
        )
        raw[sid] += 1
    return raw


def score_partition(raw: Counter, min_count: int) -> dict[str, float]:
    """Higher is better: many usable bins, low CV among usable, little mass in OTHER."""
    merged = merge_sparse(raw, min_count)
    usable = [(k, v) for k, v in merged.items() if not k.endswith("|OTHER") and v >= min_count]
    other_n = sum(v for k, v in merged.items() if k.endswith("|OTHER"))
    total = sum(merged.values()) or 1
    n_u = len(usable)
    cv = _cv([v for _, v in usable]) if usable else 1.0
    other_frac = other_n / total
    # favor usable subclasses; hard-penalize OTHER dumps and imbalance
    score = n_u * (1.0 / (1.0 + cv)) * max(0.05, 1.0 - other_frac) ** 1.5
    return {
        "score": float(score),
        "n_usable": float(n_u),
        "cv": float(cv),
        "other_frac": float(other_frac),
        "n_merged": float(len(merged)),
    }


def search_class_params(
    dets: list[dict],
    lab: str,
    count: int,
    dist_values: list[float],
    min_count: int,
) -> dict[str, Any]:
    """Grid-search geometric splits for one class."""
    budget = max(1, count // min_count)
    if budget <= 1 or count < min_count:
        return {
            "tier": "none",
            "budget": 1,
            "diag_n": 1,
            "spatial_n": 1,
            "dist_n": 1,
            "dist_edges": [],
            "use_diag": False,
            "use_spatial": False,
            "use_dist": False,
            "score": 0.0,
            "balance_cv": 0.0,
        }

    # Reference-inspired candidates; product of bins should not far exceed budget
    diag_opts = [1, 2, 4]
    spatial_opts = [1, 2, 4, 8]
    dist_opts = [1, 2, 4]

    best: dict[str, Any] | None = None
    best_score = -1.0

    for dn in diag_opts:
        for sn in spatial_opts:
            for rn in dist_opts:
                use_d, use_s, use_r = dn > 1, sn > 1, rn > 1
                if dn == 1:
                    use_d = False
                if sn == 1:
                    use_s = False
                if rn == 1:
                    use_r = False
                if not (use_d or use_s or use_r):
                    continue
                eff_d = dn if use_d else 1
                eff_s = sn if use_s else 1
                eff_r = rn if use_r else 1
                product = eff_d * eff_s * eff_r
                # keep expected count/bin >= min_count (strict)
                if product > budget:
                    continue
                if product < 2 and budget >= 2:
                    continue
                # full 4×8×4 rarely balances; avoid when all three dims on
                if use_d and use_s and use_r and product > 32:
                    continue
                edges = quantile_edges(dist_values, rn) if use_r else []
                raw = _assign_counts(dets, lab, eff_d, eff_s, edges, use_d, use_s, use_r)
                met = score_partition(raw, min_count)
                # prefer multi-dim, mild preference for ref-like spatial=4 and dist=2
                dim_bonus = 0.08 * (int(use_d) + int(use_s) + int(use_r))
                ref_prox = 0.0
                if use_s and eff_s == 4:
                    ref_prox += 0.03
                if use_d and eff_d == 2:
                    ref_prox += 0.02
                if use_r and eff_r == 2:
                    ref_prox += 0.02
                # strongly prefer low OTHER and low CV
                s = (
                    met["score"]
                    + dim_bonus
                    + ref_prox
                    - 0.5 * met["other_frac"]
                    - 0.15 * met["cv"]
                )
                if s > best_score:
                    best_score = s
                    tier = (
                        "full"
                        if use_d and use_s and use_r
                        else "high"
                        if use_d and use_s
                        else "medium"
                        if use_s or use_d
                        else "coarse"
                    )
                    best = {
                        "tier": tier,
                        "budget": budget,
                        "diag_n": eff_d,
                        "spatial_n": eff_s,
                        "dist_n": eff_r,
                        "dist_edges": edges,
                        "use_diag": use_d,
                        "use_spatial": use_s,
                        "use_dist": use_r,
                        "score": float(met["score"]),
                        "balance_cv": float(met["cv"]),
                        "n_usable": int(met["n_usable"]),
                        "other_frac": float(met["other_frac"]),
                    }

    if best is None:
        # fallback: spatial only
        sn = min(4, budget)
        return {
            "tier": "coarse",
            "budget": budget,
            "diag_n": 1,
            "spatial_n": sn,
            "dist_n": 1,
            "dist_edges": [],
            "use_diag": False,
            "use_spatial": True,
            "use_dist": False,
            "score": 0.0,
            "balance_cv": 0.0,
        }
    return best


def choose_min_count(total: int, n_classes: int) -> int:
    """Scale min_count with dataset size so rare classes aren't over-split."""
    # target: ~min 25–80 detections per subclass
    base = max(20, min(80, total // max(200, n_classes * 40)))
    return int(base)


def build_schemes(
    detections: list[dict[str, Any]],
    min_count: int | None = None,
) -> dict[str, Any]:
    total = len(detections)
    by_label = Counter(d["label"] for d in detections)
    if min_count is None:
        min_count = choose_min_count(total, len(by_label))

    dist_by_label: dict[str, list[float]] = defaultdict(list)
    diag_by_label: dict[str, list[float]] = defaultdict(list)
    spat_by_label: dict[str, list[float]] = defaultdict(list)
    for d in detections:
        dist_by_label[d["label"]].append(d["dist_ratio"])
        diag_by_label[d["label"]].append(d["diag_angle"])
        spat_by_label[d["label"]].append(d["spatial_angle"])

    schemes: dict[str, ClassScheme] = {}
    all_raw = Counter()
    all_merged = Counter()

    for lab, cnt in sorted(by_label.items(), key=lambda x: -x[1]):
        share = cnt / max(1, total)
        params = search_class_params(detections, lab, cnt, dist_by_label[lab], min_count)
        raw_c = _assign_counts(
            detections,
            lab,
            params["diag_n"],
            params["spatial_n"],
            params["dist_edges"],
            params["use_diag"],
            params["use_spatial"],
            params["use_dist"],
        )
        merged_c = merge_sparse(raw_c, min_count)
        all_raw.update(raw_c)
        all_merged.update(merged_c)
        schemes[lab] = ClassScheme(
            label=lab,
            count=cnt,
            share=share,
            tier=params["tier"],
            diag_n=params["diag_n"],
            spatial_n=params["spatial_n"],
            dist_n=params.get("dist_n", 1),
            dist_edges=params["dist_edges"],
            use_diag=params["use_diag"],
            use_spatial=params["use_spatial"],
            use_dist=params["use_dist"],
            budget=params["budget"],
            n_subclasses_raw=len(raw_c),
            n_subclasses_merged=len(merged_c),
            subclass_counts=dict(sorted(merged_c.items(), key=lambda x: -x[1])),
            sparse_raw=sum(1 for v in raw_c.values() if v < min_count),
            balance_cv=float(params.get("balance_cv", 0.0)),
            score=float(params.get("score", 0.0)),
        )

    # reference uniform 4×8×4 for comparison
    ref_raw = Counter()
    for d in detections:
        geo = GeoFeatures(
            d["diag_angle"], d["spatial_angle"], d["dist_ratio"], d["aspect"], d["area_ratio"]
        )
        sid = subclass_id(d["label"], geo, 4, 8, [0.25, 0.5, 0.75], True, True, True)
        ref_raw[sid] += 1
    ref_merged = merge_sparse(ref_raw, min_count)
    sparse_ref = sum(1 for v in ref_raw.values() if v < min_count)

    # marginal histograms (global) for report
    def hist_edges(vals: list[float], edges: list[float], lo: float, hi: float) -> dict[str, int]:
        bins = [0] * (len(edges) + 1)
        for v in vals:
            bins[sum(1 for e in edges if v >= e)] += 1
        labels = []
        prev = lo
        for i, e in enumerate(edges + [hi]):
            labels.append(f"[{prev:.2f},{e:.2f})")
            prev = e
        return {labels[i]: bins[i] for i in range(len(bins))}

    all_diag = [d["diag_angle"] for d in detections]
    all_spat = [d["spatial_angle"] for d in detections]
    all_dist = [d["dist_ratio"] for d in detections]

    return {
        "total_detections": total,
        "n_classes": len(by_label),
        "class_counts": dict(by_label),
        "schemes": {k: asdict(v) for k, v in schemes.items()},
        "adaptive_subclass_counts": dict(all_merged),
        "reference_uniform": {
            "n_subclasses_raw": len(ref_raw),
            "n_subclasses_merged": len(ref_merged),
            "n_sparse_raw": sparse_ref,
            "sparse_ratio": sparse_ref / max(1, len(ref_raw)),
        },
        "adaptive_summary": {
            "n_subclasses_raw": len(all_raw),
            "n_subclasses_merged": len(all_merged),
            "n_sparse_raw": sum(1 for v in all_raw.values() if v < min_count),
            "tier_counts": dict(Counter(s.tier for s in schemes.values())),
            "mean_balance_cv": float(np.mean([s.balance_cv for s in schemes.values()]))
            if schemes
            else 0.0,
        },
        "marginals": {
            "diag_4x22.5": hist_edges(all_diag, [22.5, 45.0, 67.5], 0.0, 90.0),
            "spatial_8x45": hist_edges(all_spat, [45 * i for i in range(1, 8)], 0.0, 360.0),
            "dist_quartile": hist_edges(
                all_dist, [float(x) for x in np.quantile(all_dist, [0.25, 0.5, 0.75])], 0.0, 1.5
            ),
        },
        "min_count": min_count,
    }
