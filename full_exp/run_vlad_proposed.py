# -*- coding: utf-8 -*-
"""Full-dataset proposed method with VLAD-SIFT recovery (no cluster ID in aux).

Official entry for capacity + Table IX robustness. Reuses VLAD banks from
``full_exp_vlad_sift`` and writes results under ``full_exp/results/``.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from full_exp.build_schemes import build_all_schemes
from full_exp.config import CACHE, LOGS, RESULTS, SEED
from full_exp.prepare_detections import prepare_all
from full_exp_vlad_sift.config import DATASETS
from full_exp_vlad_sift.run_vlad_sift import run_all, run_one_dataset, write_report


def run_full(datasets=None, skip_capacity: bool = False):
    """Prepare detections/schemes if needed, then run VLAD-SIFT Table IX."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    print("=== 1/3 prepare detections ===")
    det_paths = prepare_all()

    print("=== 2/3 build geometric schemes ===")
    scheme_json = RESULTS / "geo_subclass_full.json"
    if not scheme_json.exists():
        scheme_json = build_all_schemes(det_paths)
    else:
        print(f"reuse {scheme_json}")

    print("=== 3/3 VLAD-SIFT capacity + robustness ===")
    datasets = list(datasets or DATASETS)
    partial = RESULTS / "full_vlad_proposed_results.partial.json"
    all_res = {}
    if partial.exists():
        all_res = json.loads(partial.read_text(encoding="utf-8"))
        print(f"resume partial datasets: {list(all_res)}")

    for name in datasets:
        if name in all_res and "error" not in all_res[name].get("robustness", {}):
            print(f"skip done {name}")
            continue
        res = run_one_dataset(name, skip_capacity=skip_capacity)
        all_res[name] = res
        partial.write_text(json.dumps(all_res, indent=2, ensure_ascii=False), encoding="utf-8")

    out_json = RESULTS / "full_vlad_proposed_results.json"
    out_md = RESULTS / "full_vlad_proposed_results.md"
    out_json.write_text(json.dumps(all_res, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(all_res, out_md)

    vlad_json = Path(__file__).resolve().parents[1] / "full_exp_vlad_sift" / "results" / "vlad_sift_results.json"
    vlad_md = vlad_json.with_suffix(".md")
    if vlad_json.exists():
        shutil.copy2(vlad_json, out_json.with_name("vlad_sift_results.json"))
    if vlad_md.exists():
        shutil.copy2(vlad_md, out_md.with_name("vlad_sift_results.md"))

    print("Wrote", out_md)
    print("seed", SEED)
    return all_res


if __name__ == "__main__":
    import sys

    ds = [a for a in sys.argv[1:] if not a.startswith("-")]
    cap = "--capacity" in sys.argv
    if ds:
        run_full(ds, skip_capacity=not cap)
    else:
        run_all(ds or None)
