# -*- coding: utf-8 -*-
"""Build adaptive geometric subclass schemes on full detections."""
from __future__ import annotations

import json
from pathlib import Path

from full_exp.config import RESULTS
from full_exp.prepare_detections import prepare_all
from geo_refine.adaptive_scheme import build_schemes


def flat_dets(preds: dict[str, list]) -> list[dict]:
    out = []
    for v in preds.values():
        out.extend(v)
    return out


def write_md(all_sch: dict, path: Path):
    lines = [
        "# Full-dataset Geometric Subclass Schemes (YOLOv8m @0.5)",
        "",
        "Adaptive per-class (D,S,R) under min_count budget; sparse → OTHER.",
        "",
    ]
    for name, sch in all_sch.items():
        summ = sch["adaptive_summary"]
        lines.append(f"## {name}")
        lines.append("")
        lines.append(
            f"- detections: **{sch['total_detections']}**, classes: **{sch['n_classes']}**, "
            f"min_count={sch['min_count']}"
        )
        lines.append(
            f"- adaptive: raw={summ['n_subclasses_raw']} → merged=**{summ['n_subclasses_merged']}**, "
            f"tiers={summ['tier_counts']}, mean_CV={summ['mean_balance_cv']:.3f}"
        )
        lines.append("")
        lines.append("| Class | count | tier | D×S×R | #sub(merged) | CV |")
        lines.append("|---|---:|---|---|---:|---:|")
        for lab, s in sorted(sch["schemes"].items(), key=lambda x: -x[1]["count"]):
            lines.append(
                f"| {lab} | {s['count']} | {s['tier']} | "
                f"{s['diag_n']}×{s['spatial_n']}×{s['dist_n']} | "
                f"{s['n_subclasses_merged']} | {s['balance_cv']:.2f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_all_schemes(det_paths: dict[str, Path] | None = None) -> Path:
    if det_paths is None:
        det_paths = prepare_all()
    RESULTS.mkdir(parents=True, exist_ok=True)
    all_sch: dict = {}
    for name, cache in det_paths.items():
        print(f"\n===== schemes {name} =====")
        preds = json.loads(cache.read_text(encoding="utf-8"))
        flat = flat_dets(preds)
        print(f"images={len(preds)} dets={len(flat)}")
        sch = build_schemes(flat)
        all_sch[name] = sch
        summ = sch["adaptive_summary"]
        print(
            f"  atoms merged={summ['n_subclasses_merged']}  "
            f"mean_CV={summ['mean_balance_cv']:.3f}"
        )

    out_json = RESULTS / "geo_subclass_full.json"
    out_md = RESULTS / "geo_subclass_full.md"
    out_json.write_text(json.dumps(all_sch, indent=2, ensure_ascii=False), encoding="utf-8")
    write_md(all_sch, out_md)
    print("Wrote", out_json)
    return out_json


if __name__ == "__main__":
    build_all_schemes()
