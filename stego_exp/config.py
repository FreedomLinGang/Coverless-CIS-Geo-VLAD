"""Experiment configuration aligned with Lin & Liang TMM 2026 protocol."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache"
RESULT_DIR = ROOT / "results"

# Local datasets (same families as the paper; use available local mirrors)
DATASETS = {
    "voc2012": Path(r"D:\datasets\VOC2012\JPEGImages"),
    "coco_val2017": Path(r"D:\datasets\voc\val2017"),
    "coco_train_rgb": Path(r"D:\datasets\coco_rgb\train"),
}

COCO_CLASS_NAMES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


@dataclass
class ExpConfig:
    dataset: str = "coco_val2017"
    max_images: int = 2000
    conf_thresh: float = 0.5
    # Area ratio threshold (box_area / image_area)
    area_ratio_thresh: float = 0.02
    max_objects_per_image: int = 12
    segment_bits: int = 6
    n_clusters: int = 64
    device: str = "cuda"
    seed: int = 42
    # Capacity test message lengths (bytes), paper protocol
    message_lengths: tuple[int, ...] = (10, 100, 1000)
    capacity_trials: int = 5
    # Robustness
    robustness_messages: int = 20
    message_bytes_for_robust: int = 50
    attack_repeats: int = 1
    use_attributes: bool = True  # fine-grained coding on same images
    retrieve_topk: int = 20


DEFAULT_CFG = ExpConfig()
