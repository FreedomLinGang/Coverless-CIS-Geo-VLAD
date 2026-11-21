"""Load VOC Pascal XML ground-truth boxes as detections (no network weights)."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def parse_voc_xml(xml_path: Path) -> list[dict[str, Any]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    w = float(size.findtext("width"))
    h = float(size.findtext("height"))
    objs = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        bnd = obj.find("bndbox")
        x1 = float(bnd.findtext("xmin"))
        y1 = float(bnd.findtext("ymin"))
        x2 = float(bnd.findtext("xmax"))
        y2 = float(bnd.findtext("ymax"))
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        objs.append(
            {
                "label": name,
                "score": 1.0,
                "box": [x1, y1, x2, y2],
                "area": area,
                "area_ratio": area / float(max(1.0, w * h)),
                "cx": (x1 + x2) * 0.5,
                "cy": (y1 + y2) * 0.5,
                "aspect": (y2 - y1) / max(1e-6, x2 - x1),
            }
        )
    objs.sort(key=lambda o: (-o["area"], o["cy"], o["cx"], o["label"]))
    return objs


def build_voc_gt_detections(jpeg_dir: Path, ann_dir: Path, image_paths: list[Path]) -> dict[str, list[dict]]:
    records = {}
    for p in image_paths:
        xml = ann_dir / f"{p.stem}.xml"
        if not xml.exists():
            records[str(p)] = []
            continue
        records[str(p)] = parse_voc_xml(xml)
    return records
