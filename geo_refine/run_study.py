"""Run detector comparison + geometric fine-grain statistics."""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .adaptive_scheme import build_schemes
from .config import CACHE, COCO_TRAIN, RESULTS, RunConfig
from .detectors import build_detector
from .voc_eval import eval_voc_map, list_coco_images, list_voc_images


def run_detect_cache(
    detector_name: str,
    image_paths: list[Path],
    conf: float,
    device: str,
    voc_mode: bool,
    cache_path: Path,
) -> dict[str, list[dict]]:
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    det = build_detector(detector_name, device=device)
    out: dict[str, list[dict]] = {}
    for p in tqdm(image_paths, desc=f"detect[{detector_name}]"):
        try:
            out[str(p)] = det.detect(p, conf=conf, voc_mode=voc_mode)
        except Exception as e:
            print(f"[warn] {p}: {e}")
            out[str(p)] = []
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(out), encoding="utf-8")
    return out


def flatten(preds: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for v in preds.values():
        rows.extend(v)
    return rows


def quality_proxy(preds: dict[str, list[dict]]) -> dict:
    flat = flatten(preds)
    if not flat:
        return {"n_det": 0, "mean_conf": 0.0, "per_image_mean": 0.0}
    confs = [d["score"] for d in flat]
    per_img = [len(v) for v in preds.values()]
    return {
        "n_det": len(flat),
        "mean_conf": float(np.mean(confs)),
        "per_image_mean": float(np.mean(per_img)),
        "class_counts": dict(Counter(d["label"] for d in flat)),
    }


def compare_on_voc(cfg: RunConfig) -> dict:
    paths = list_voc_images(cfg.max_images_voc, cfg.seed)
    print(f"VOC images: {len(paths)}")
    results = {"n_images": len(paths), "detectors": {}}
    for name in cfg.detectors:
        t0 = time.time()
        cache = CACHE / f"voc_{name}_n{len(paths)}_c{cfg.conf_thresh}.json"
        try:
            preds = run_detect_cache(name, paths, cfg.conf_thresh, cfg.device, True, cache)
        except Exception as e:
            results["detectors"][name] = {"error": str(e)}
            print(f"[fail] {name}: {e}")
            continue
        metrics = eval_voc_map(preds, paths, iou_thresh=cfg.iou_eval)
        proxy = quality_proxy(preds)
        flat = flatten(preds)
        scheme = build_schemes(flat, min_count=30)
        results["detectors"][name] = {
            "seconds": time.time() - t0,
            "mAP50": metrics["mAP50"],
            "per_class_ap": metrics["per_class_ap"],
            "proxy": proxy,
            "scheme": scheme,
        }
        print(f"{name}: mAP50={metrics['mAP50']:.4f} n_det={proxy['n_det']}")
    # pick best by mAP
    ok = {k: v for k, v in results["detectors"].items() if "mAP50" in v}
    if ok:
        best = max(ok.items(), key=lambda kv: kv[1]["mAP50"])
        results["best_detector"] = best[0]
        results["best_mAP50"] = best[1]["mAP50"]
    return results


def compare_on_coco(cfg: RunConfig) -> dict:
    folder = COCO_TRAIN if COCO_TRAIN.exists() else COCO_TRAIN
    paths = list_coco_images(folder, cfg.max_images_coco, cfg.seed)
    print(f"COCO-RGB images: {len(paths)} from {folder}")
    results = {"n_images": len(paths), "folder": str(folder), "detectors": {}}
    for name in cfg.detectors:
        t0 = time.time()
        cache = CACHE / f"coco_{name}_n{len(paths)}_c{cfg.conf_thresh}.json"
        try:
            preds = run_detect_cache(name, paths, cfg.conf_thresh, cfg.device, False, cache)
        except Exception as e:
            results["detectors"][name] = {"error": str(e)}
            print(f"[fail] {name}: {e}")
            continue
        proxy = quality_proxy(preds)
        flat = flatten(preds)
        # COCO denser → slightly higher min_count
        scheme = build_schemes(flat, min_count=40)
        results["detectors"][name] = {
            "seconds": time.time() - t0,
            "proxy": proxy,
            "scheme": scheme,
        }
        print(f"{name}: n_det={proxy['n_det']} mean_conf={proxy['mean_conf']:.3f}")
    # best by mean_conf * log(n_det) heuristic when no GT
    ok = {k: v for k, v in results["detectors"].items() if "proxy" in v and v["proxy"]["n_det"] > 0}
    if ok:
        best = max(
            ok.items(),
            key=lambda kv: kv[1]["proxy"]["mean_conf"] * np.log1p(kv[1]["proxy"]["n_det"]),
        )
        results["best_detector_proxy"] = best[0]
    return results


def write_report(voc_res: dict, coco_res: dict, path: Path):
    lines = []
    lines.append("# Geometric Fine-Grain Detection Study\n")
    lines.append("## 1. Detector comparison\n")
    lines.append("### VOC2012 (with GT → mAP@0.5)\n")
    lines.append("| Detector | mAP@0.5 | #Detections | mean conf | time(s) |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, block in voc_res.get("detectors", {}).items():
        if "error" in block or "mAP50" not in block:
            lines.append(f"| {name} | ERROR | - | - | - |")
            continue
        sec = block.get("seconds")
        sec_s = f"{sec:.1f}" if isinstance(sec, (int, float)) else "-"
        lines.append(
            f"| {name} | {block['mAP50']:.4f} | {block['proxy']['n_det']} | "
            f"{block['proxy']['mean_conf']:.3f} | {sec_s} |"
        )
    lines.append(f"\n**Best on VOC:** `{voc_res.get('best_detector')}` "
                 f"(mAP={voc_res.get('best_mAP50', 0):.4f})\n")

    lines.append("### COCO-RGB train subset (no GT → proxy)\n")
    lines.append("| Detector | #Detections | mean conf | det/image | time(s) |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, block in coco_res.get("detectors", {}).items():
        if "error" in block:
            lines.append(f"| {name} | ERROR | - | - | - |")
            continue
        p = block["proxy"]
        lines.append(
            f"| {name} | {p['n_det']} | {p['mean_conf']:.3f} | {p['per_image_mean']:.2f} | {block['seconds']:.1f} |"
        )
    lines.append(f"\n**Best proxy on COCO-RGB:** `{coco_res.get('best_detector_proxy')}`\n")

    # recommended schemes from best detectors
    lines.append("## 2. Recommended fine-grain schemes\n")
    best_v = voc_res.get("best_detector")
    if best_v and best_v in voc_res["detectors"]:
        sch = voc_res["detectors"][best_v]["scheme"]
        lines.append(f"### VOC2012 (detector=`{best_v}`)\n")
        lines.append(
            f"- total dets: {sch['total_detections']}, classes: {sch['n_classes']}\n"
            f"- uniform 4×8×4 raw subclasses: {sch['reference_uniform']['n_subclasses_raw']} "
            f"(sparse<{30}: {sch['reference_uniform']['n_sparse_raw']}, "
            f"ratio={sch['reference_uniform']['sparse_ratio']:.2f})\n"
            f"- adaptive merged subclasses: {sch['adaptive_summary']['n_subclasses_merged']} "
            f"(tiers={sch['adaptive_summary']['tier_counts']})\n"
        )
        lines.append("| Class | count | share | tier | budget | diag_n | spatial_n | dist | #sub(raw→merged) |")
        lines.append("|---|---:|---:|---|---:|---:|---:|---|---:|")
        for lab, s in sorted(sch["schemes"].items(), key=lambda kv: -kv[1]["count"]):
            dist_flag = "Y" if s["use_dist"] else "N"
            budget = s.get("budget", "-")
            lines.append(
                f"| {lab} | {s['count']} | {s['share']:.3f} | {s['tier']} | {budget} | "
                f"{s['diag_n']} | {s['spatial_n']} | {dist_flag} | "
                f"{s['n_subclasses_raw']}→{s['n_subclasses_merged']} |"
            )
            # show top subclasses for frequent classes
            if s["tier"] in ("full", "high", "medium") and s.get("subclass_counts"):
                top = list(s["subclass_counts"].items())[:8]
                tip = ", ".join(f"{a}:{b}" for a, b in top)
                lines.append(f"| ↳ | | | | | | | | {tip} |")

    best_c = coco_res.get("best_detector_proxy")
    if best_c and best_c in coco_res["detectors"]:
        sch = coco_res["detectors"][best_c]["scheme"]
        lines.append(f"\n### COCO-RGB (detector=`{best_c}`)\n")
        lines.append(
            f"- total dets: {sch['total_detections']}, classes: {sch['n_classes']}\n"
            f"- uniform 4×8×4 raw subclasses: {sch['reference_uniform']['n_subclasses_raw']} "
            f"(sparse ratio={sch['reference_uniform']['sparse_ratio']:.2f})\n"
            f"- adaptive merged subclasses: {sch['adaptive_summary']['n_subclasses_merged']} "
            f"(tiers={sch['adaptive_summary']['tier_counts']})\n"
        )
        lines.append("| Class | count | share | tier | diag_n | spatial_n | dist | #sub(merged) |")
        lines.append("|---|---:|---:|---|---:|---:|---|---:|")
        # top 25 classes
        items = sorted(sch["schemes"].items(), key=lambda kv: -kv[1]["count"])[:25]
        for lab, s in items:
            dist_flag = "Y" if s["use_dist"] else "N"
            lines.append(
                f"| {lab} | {s['count']} | {s['share']:.3f} | {s['tier']} | "
                f"{s['diag_n']} | {s['spatial_n']} | {dist_flag} | {s['n_subclasses_merged']} |"
            )
        lines.append("\n*(table shows top-25 classes by count)*\n")

    lines.append("## 3. Design notes\n")
    lines.append(
        "- **Diag angle**: BL→TR vs bottom edge = atan(h/w) ∈ (0,90°]; ref 4 bins ×22.5°.\n"
        "- **Spatial angle**: image-center→box-center vs +x ∈ [0,360); ref 8 bins ×45°.\n"
        "- **Distance ratio**: |c_box−c_img| / (0.5·image_diagonal); quantile edges per class.\n"
        "- **Adaptive tiers**: full/medium/coarse/none by class frequency to avoid sparsity.\n"
        "- Sparse subclasses merged into `{class}|OTHER`.\n"
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    cfg = RunConfig()
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


if __name__ == "__main__":
    main()
