"""Reprocess cached detections with fixed adaptive scheme + VOC Faster R-CNN mAP."""
from __future__ import annotations

import json
from pathlib import Path

from geo_refine.adaptive_scheme import build_schemes
from geo_refine.config import CACHE, RESULTS, RunConfig
from geo_refine.run_study import (
    compare_on_voc,
    flatten,
    quality_proxy,
    run_detect_cache,
    write_report,
)
from geo_refine.voc_eval import eval_voc_map, list_voc_images


def reprocess_cache(cache_path: Path, min_count: int) -> dict:
    preds = json.loads(cache_path.read_text(encoding="utf-8"))
    flat = flatten(preds)
    scheme = build_schemes(flat, min_count=min_count)
    return {"proxy": quality_proxy(preds), "scheme": scheme, "n_images": len(preds)}


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    cfg = RunConfig(max_images_voc=500, max_images_coco=500, detectors=("fasterrcnn",), conf_thresh=0.5)

    # 1) Finish VOC Faster R-CNN (weights fixed)
    print("==== VOC Faster R-CNN ====")
    paths = list_voc_images(500, 42)
    cache = CACHE / f"voc_fasterrcnn_n{len(paths)}_c{cfg.conf_thresh}.json"
    preds = run_detect_cache("fasterrcnn", paths, 0.5, "cuda", True, cache)
    metrics = eval_voc_map(preds, paths, 0.5)
    print(f"fasterrcnn VOC mAP50={metrics['mAP50']:.4f} n_det={sum(len(v) for v in preds.values())}")

    # 2) Load existing JSON results and patch
    voc_path = RESULTS / "voc_detector_geo.json"
    coco_path = RESULTS / "coco_detector_geo.json"
    voc_res = json.loads(voc_path.read_text(encoding="utf-8"))
    coco_res = json.loads(coco_path.read_text(encoding="utf-8"))

    voc_res["detectors"]["fasterrcnn"] = {
        "mAP50": metrics["mAP50"],
        "per_class_ap": metrics["per_class_ap"],
        "proxy": quality_proxy(preds),
        "scheme": build_schemes(flatten(preds), min_count=25),
        "seconds": None,
    }
    # reprocess YOLO schemes with new adaptive logic
    for name in ("yolov8s", "yolov8m"):
        c = CACHE / f"voc_{name}_n500_c0.5.json"
        if c.exists() and name in voc_res["detectors"] and "mAP50" in voc_res["detectors"][name]:
            block = reprocess_cache(c, min_count=25)
            voc_res["detectors"][name]["scheme"] = block["scheme"]
            voc_res["detectors"][name]["proxy"] = block["proxy"]

    ok = {k: v for k, v in voc_res["detectors"].items() if "mAP50" in v}
    best = max(ok.items(), key=lambda kv: kv[1]["mAP50"])
    voc_res["best_detector"] = best[0]
    voc_res["best_mAP50"] = best[1]["mAP50"]

    for name in ("fasterrcnn", "yolov8s", "yolov8m"):
        c = CACHE / f"coco_{name}_n500_c0.5.json"
        if c.exists() and name in coco_res.get("detectors", {}):
            block = reprocess_cache(c, min_count=35)
            coco_res["detectors"][name]["scheme"] = block["scheme"]
            coco_res["detectors"][name]["proxy"] = block["proxy"]

    okc = {
        k: v
        for k, v in coco_res["detectors"].items()
        if "proxy" in v and v["proxy"].get("n_det", 0) > 0
    }
    if okc:
        import numpy as np

        bestc = max(
            okc.items(),
            key=lambda kv: kv[1]["proxy"]["mean_conf"] * np.log1p(kv[1]["proxy"]["n_det"]),
        )
        coco_res["best_detector_proxy"] = bestc[0]

    voc_path.write_text(json.dumps(voc_res, indent=2, ensure_ascii=False), encoding="utf-8")
    coco_path.write_text(json.dumps(coco_res, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(voc_res, coco_res, RESULTS / "recommended_schemes.md")
    print("Updated", RESULTS / "recommended_schemes.md")
    print("Best VOC:", voc_res["best_detector"], voc_res["best_mAP50"])


if __name__ == "__main__":
    main()
