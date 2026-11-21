"""CLI: python -m geo_refine.run_main"""
from __future__ import annotations

import argparse

from geo_refine.config import RunConfig
from geo_refine.run_study import main as run_default
from geo_refine import run_study


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--max-voc", type=int, default=600)
    p.add_argument("--max-coco", type=int, default=600)
    p.add_argument(
        "--detectors",
        default="fasterrcnn,yolov8s,yolov8m",
        help="comma-separated: fasterrcnn,yolov8n,yolov8s,yolov8m",
    )
    args = p.parse_args()
    cfg = RunConfig(
        device=args.device,
        conf_thresh=args.conf,
        max_images_voc=args.max_voc,
        max_images_coco=args.max_coco,
        detectors=tuple(x.strip() for x in args.detectors.split(",") if x.strip()),
    )
    # monkey-patch module-level run via local override
    import geo_refine.run_study as rs

    def _main():
        from geo_refine.run_study import compare_on_voc, compare_on_coco, write_report
        from geo_refine.config import RESULTS, CACHE
        import json

        RESULTS.mkdir(parents=True, exist_ok=True)
        CACHE.mkdir(parents=True, exist_ok=True)
        print("==== VOC ====")
        voc_res = compare_on_voc(cfg)
        (RESULTS / "voc_detector_geo.json").write_text(
            json.dumps(voc_res, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("==== COCO-RGB ====")
        coco_res = compare_on_coco(cfg)
        (RESULTS / "coco_detector_geo.json").write_text(
            json.dumps(coco_res, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        report = RESULTS / "recommended_schemes.md"
        write_report(voc_res, coco_res, report)
        print(f"Report: {report}")

    _main()


if __name__ == "__main__":
    main()
