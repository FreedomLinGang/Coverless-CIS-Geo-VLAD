"""YOLOv8m@0.5 geometric subclass study on VOC2007 / VOC2012 / COCO train+val."""
from __future__ import annotations

import json
import time
from pathlib import Path

from tqdm import tqdm

from geo_refine.adaptive_scheme import build_schemes
from geo_refine.config import (
    CACHE,
    COCO_TRAIN,
    COCO_VAL,
    RESULTS,
    VOC2007_TEST,
    VOC2007_TRAIN,
    VOC2012_ROOT,
)
from geo_refine.detectors import build_detector


def list_images(folder: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)


def load_or_detect(
    cache: Path,
    paths: list[Path],
    device: str = "cuda",
    conf: float = 0.5,
    voc_mode: bool = True,
    folder: Path | None = None,
) -> list[dict]:
    if cache.exists():
        print(f"load cache {cache.name}")
        preds = json.loads(cache.read_text(encoding="utf-8"))
    else:
        print(f"detect {len(paths)} images → {cache.name}")
        det = build_detector("yolov8m", device=device)
        t0 = time.time()
        if folder is not None and hasattr(det, "detect_folder"):
            # batched folder inference (much faster on COCO)
            preds = det.detect_folder(folder, conf=conf, voc_mode=voc_mode, batch=16)
            # ensure all requested paths present
            for p in paths:
                preds.setdefault(str(p.resolve()), preds.get(str(p), []))
                if str(p) not in preds and str(p.resolve()) in preds:
                    preds[str(p)] = preds[str(p.resolve())]
        else:
            preds = {}
            for p in tqdm(paths, desc="yolov8m"):
                try:
                    preds[str(p)] = det.detect(p, conf=conf, voc_mode=voc_mode)
                except Exception as e:
                    print(f"[warn] {p}: {e}")
                    preds[str(p)] = []
        CACHE.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(preds), encoding="utf-8")
        print(f"detect seconds={time.time() - t0:.1f}")
    flat: list[dict] = []
    for v in preds.values():
        flat.extend(v)
    return flat


def merge_caches(caches: list[Path]) -> list[dict]:
    flat: list[dict] = []
    for c in caches:
        if not c.exists():
            raise FileNotFoundError(c)
        preds = json.loads(c.read_text(encoding="utf-8"))
        for v in preds.values():
            flat.extend(v)
        print(f"  + {c.name}: images={len(preds)} dets={sum(len(v) for v in preds.values())}")
    return flat


def write_dataset_md(name: str, sch: dict, lines: list[str]):
    ref = sch["reference_uniform"]
    ada = sch["adaptive_summary"]
    lines.append(f"## {name}")
    lines.append("")
    lines.append(
        f"- detections: **{sch['total_detections']}**, classes: **{sch['n_classes']}**, "
        f"min_count={sch['min_count']}"
    )
    lines.append(
        f"- uniform 4×8×4: raw={ref['n_subclasses_raw']}, sparse={ref['n_sparse_raw']} "
        f"({ref['sparse_ratio']:.0%}), merged={ref['n_subclasses_merged']}"
    )
    lines.append(
        f"- **adaptive**: raw={ada['n_subclasses_raw']} → merged=**{ada['n_subclasses_merged']}**, "
        f"tiers={ada['tier_counts']}, mean_CV={ada['mean_balance_cv']:.3f}"
    )
    lines.append("")
    lines.append("### Global marginals (reference bins)")
    lines.append("")
    for dim, hist in sch["marginals"].items():
        lines.append(f"**{dim}**: " + ", ".join(f"{k}:{v}" for k, v in hist.items()))
    lines.append("")
    lines.append(
        "| Class | count | share | tier | D×S×R | #sub(raw→merged) | CV | top subclasses |"
    )
    lines.append("|---|---:|---:|---|---|---:|---:|---|")
    for lab, s in sch["schemes"].items():
        d = s["diag_n"] if s["use_diag"] else 1
        sp = s["spatial_n"] if s["use_spatial"] else 1
        r = s["dist_n"] if s["use_dist"] else 1
        top = list(s["subclass_counts"].items())[:5]
        top_s = ", ".join(f"{k.split('|',1)[-1] if '|' in k else k}:{v}" for k, v in top)
        lines.append(
            f"| {lab} | {s['count']} | {s['share']:.3f} | {s['tier']} | "
            f"{d}×{sp}×{r} | {s['n_subclasses_raw']}→{s['n_subclasses_merged']} | "
            f"{s['balance_cv']:.2f} | {top_s} |"
        )
    lines.append("")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    conf = 0.5

    # --- VOC2007 train+test (already cached as combined) ---
    print("==== VOC2007 ====")
    c07 = CACHE / "voc2007_yolov8m_n9963_c0.5.json"
    if not c07.exists():
        paths = list_images(VOC2007_TRAIN / "JPEGImages") + list_images(VOC2007_TEST / "JPEGImages")
        load_or_detect(c07, paths, device, conf, voc_mode=True)
    dets07 = merge_caches([c07])
    sch07 = build_schemes(dets07)

    # --- VOC2012 full (trainval) ---
    print("==== VOC2012 ====")
    c12 = CACHE / "voc2012_yolov8m_n17125_c0.5.json"
    if not c12.exists():
        paths = list_images(VOC2012_ROOT / "JPEGImages")
        load_or_detect(c12, paths, device, conf, voc_mode=True)
    dets12 = merge_caches([c12])
    sch12 = build_schemes(dets12)

    # early save VOC so long COCO run still leaves usable results
    partial = {
        "detector": "yolov8m",
        "conf": conf,
        "VOC2007": sch07,
        "VOC2012": sch12,
    }
    (RESULTS / "geo_subclass_yolov8m_voc_partial.json").write_text(
        json.dumps(partial, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Saved VOC partial schemes")

    # --- COCO train + val ---
    print("==== COCO2017 train+val ====")
    c_val = CACHE / "coco_val_yolov8m_n5000_c0.5.json"
    if not c_val.exists():
        load_or_detect(c_val, list_images(COCO_VAL), device, conf, voc_mode=False)

    train_paths = list_images(COCO_TRAIN)
    c_train = CACHE / f"coco_train_yolov8m_n{len(train_paths)}_c0.5.json"
    if not c_train.exists():
        print(f"COCO train images: {len(train_paths)} (batched folder predict)")
        load_or_detect(
            c_train, train_paths, device, conf, voc_mode=False, folder=COCO_TRAIN
        )
    else:
        print(f"load cache {c_train.name}")
    dets_coco = merge_caches([c_train, c_val])
    sch_coco = build_schemes(dets_coco)

    out = {
        "detector": "yolov8m",
        "conf": conf,
        "VOC2007": sch07,
        "VOC2012": sch12,
        "COCO2017_trainval": sch_coco,
    }
    out_json = RESULTS / "geo_subclass_yolov8m.json"
    out_md = RESULTS / "geo_subclass_yolov8m.md"
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Geometric Subclass Schemes (YOLOv8m, conf=0.5)",
        "",
        "## Method",
        "",
        "1. **Diag angle** `atan(h/w)` ∈ (0,90°]: shape of box (ref 4×22.5°).",
        "2. **Spatial angle** image-center→box-center ∈ [0,360): location (ref 8×45°).",
        "3. **Dist ratio** `|c_box−c_img|/(0.5·diag)`: near/far (quantile edges, not fixed).",
        "",
        "Per class we **grid-search** (D,S,R) under budget≈count/min_count, maximize "
        "`n_usable × 1/(1+CV) × (1−0.7·OTHER_frac)`, then merge sparse bins into `|OTHER`.",
        "Uniform 4×8×4 is reported as a sparse reference scheme.",
        "",
        "Datasets use **train+val/test together** (VOC07 train+test, VOC2012 all, COCO train2017+val2017).",
        "",
    ]
    write_dataset_md("VOC2007 (train+test)", sch07, lines)
    write_dataset_md("VOC2012 (full)", sch12, lines)
    write_dataset_md("COCO2017 (train+val)", sch_coco, lines)

    lines += [
        "## Recommended default for stego codebook",
        "",
        "- Prefer **adaptive** schemes above, not fixed 4×8×4.",
        "- Frequent classes (`person`, `car`, …): keep multi-dim split (often 2–4 × 4 × 2).",
        "- Rare classes: fewer bins or no split (`tier=none`).",
        "- Distance uses **class-wise quantiles** (more balanced than fixed 0.25/0.5/0.75).",
        "- Spatial often more informative than diag for natural photos; diag helps for tall/wide objects.",
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", out_md)
    print(
        "summary:",
        f"VOC07 merged={sch07['adaptive_summary']['n_subclasses_merged']}",
        f"VOC12 merged={sch12['adaptive_summary']['n_subclasses_merged']}",
        f"COCO merged={sch_coco['adaptive_summary']['n_subclasses_merged']}",
    )


if __name__ == "__main__":
    main()
