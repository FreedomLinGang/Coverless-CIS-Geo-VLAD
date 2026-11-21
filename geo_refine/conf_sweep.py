"""Sweep confidence thresholds on cached full-VOC detections (no re-inference).

Uses voc_full_*_n*_c0.5.json caches; filters score >= thresh for 0.5..0.9.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from geo_refine.config import CACHE, RESULTS
from geo_refine.full_compare import gt_counts, matched_tp_counts
from geo_refine.voc_eval import eval_voc_map, list_voc_images


DETECTORS = ["fasterrcnn", "yolov8m", "yolov8x"]
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


def filter_preds(preds: dict, conf: float) -> dict:
    return {
        k: [d for d in v if float(d["score"]) >= conf]
        for k, v in preds.items()
    }


def eval_at_conf(name: str, preds0: dict, paths: list[Path], conf: float, gt: Counter) -> dict:
    preds = filter_preds(preds0, conf)
    metrics = eval_voc_map(preds, paths, iou_thresh=0.5)
    det_c = Counter(d["label"] for v in preds.values() for d in v)
    tp, fp, fn = matched_tp_counts(preds, paths, 0.5)
    total_gt = int(sum(gt.values()))
    total_det = int(sum(det_c.values()))
    total_tp = int(sum(tp.values()))
    total_fp = int(sum(fp.values()))
    total_fn = int(sum(fn.values()))
    return {
        "detector": name,
        "conf": conf,
        "mAP50": metrics["mAP50"],
        "total_gt": total_gt,
        "total_detected": total_det,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "det_over_gt": total_det / max(1, total_gt),
        "recall": total_tp / max(1, total_gt),
        "precision": total_tp / max(1, total_det),
    }


def write_md(rows: list[dict], path: Path):
    lines = [
        "# VOC2012 Full: Confidence Threshold Sweep",
        "",
        "Source: cached detections at conf≥0.5, then re-filtered (equivalent to raising threshold).",
        "Match: same class + IoU≥0.5. GT objs = 15686 (all annotated images).",
        "",
        "## Summary",
        "",
        "| Detector | Conf | mAP@0.5 | GT | Det | TP | FP | FN | Det/GT | Recall | Precision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['detector']} | {r['conf']:.1f} | {r['mAP50']:.4f} | {r['total_gt']} | "
            f"{r['total_detected']} | {r['total_tp']} | {r['total_fp']} | {r['total_fn']} | "
            f"{r['det_over_gt']:.3f} | {r['recall']:.3f} | {r['precision']:.3f} |"
        )
    lines.append("")
    lines.append("## By detector")
    lines.append("")
    for name in DETECTORS:
        lines.append(f"### `{name}`")
        lines.append("")
        lines.append("| Conf | mAP@0.5 | Det | TP | FP | FN | Recall | Precision |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            if r["detector"] != name:
                continue
            lines.append(
                f"| {r['conf']:.1f} | {r['mAP50']:.4f} | {r['total_detected']} | "
                f"{r['total_tp']} | {r['total_fp']} | {r['total_fn']} | "
                f"{r['recall']:.3f} | {r['precision']:.3f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    paths = list_voc_images(max_images=0, seed=42)
    gt = gt_counts(paths)
    print(f"VOC images={len(paths)} GT objs={sum(gt.values())}")

    rows: list[dict] = []
    for name in DETECTORS:
        cache = CACHE / f"voc_full_{name}_n{len(paths)}_c0.5.json"
        if not cache.exists():
            # try glob any n
            cands = sorted(CACHE.glob(f"voc_full_{name}_n*_c0.5.json"))
            if not cands:
                print(f"[skip] no cache for {name}")
                continue
            cache = cands[-1]
        print(f"Load {cache.name} ...")
        preds0 = json.loads(cache.read_text(encoding="utf-8"))
        for conf in THRESHOLDS:
            r = eval_at_conf(name, preds0, paths, conf, gt)
            rows.append(r)
            print(
                f"  conf={conf:.1f} mAP={r['mAP50']:.4f} det={r['total_detected']} "
                f"TP={r['total_tp']} FP={r['total_fp']} FN={r['total_fn']} "
                f"P={r['precision']:.3f} R={r['recall']:.3f}"
            )

    out_json = RESULTS / "voc_conf_sweep.json"
    out_md = RESULTS / "voc_conf_sweep.md"
    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_md(rows, out_md)
    print("Wrote", out_md)


if __name__ == "__main__":
    main()
