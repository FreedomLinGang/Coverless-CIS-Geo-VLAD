# Geometric Fine-Grain Detection Study

Compare Faster R-CNN / YOLOv8 on VOC2012 & coco_rgb, then build adaptive geometric subclasses.

## Run

```bash
conda activate torchgpu
cd our_code
python -m geo_refine.run_main --max-voc 600 --max-coco 600 --detectors fasterrcnn,yolov8s,yolov8m
```

## Outputs

- `geo_refine/results/voc_detector_geo.json`
- `geo_refine/results/coco_detector_geo.json`
- `geo_refine/results/recommended_schemes.md`
