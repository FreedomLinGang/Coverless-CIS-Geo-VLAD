"""Entry point: python -m stego_exp.run_main"""
from __future__ import annotations

import argparse

from stego_exp.config import ExpConfig
from stego_exp.run_experiment import run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="voc2012", choices=["voc2012", "coco_val2017", "coco_train_rgb"])
    p.add_argument("--max-images", type=int, default=600)
    p.add_argument("--segment-bits", type=int, default=4)
    p.add_argument("--capacity-trials", type=int, default=3)
    p.add_argument("--robust-messages", type=int, default=5)
    p.add_argument("--msg-bytes", type=int, default=20)
    p.add_argument("--area-ratio", type=float, default=0.015)
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-attributes", action="store_true")
    args = p.parse_args()

    cfg = ExpConfig(
        dataset=args.dataset,
        max_images=args.max_images,
        segment_bits=args.segment_bits,
        capacity_trials=args.capacity_trials,
        robustness_messages=args.robust_messages,
        message_bytes_for_robust=args.msg_bytes,
        area_ratio_thresh=args.area_ratio,
        device=args.device,
        use_attributes=not args.no_attributes,
    )
    run(cfg)


if __name__ == "__main__":
    main()
