# Coverless Image Steganography: Geometric Subclass Mapping + Cluster-Free Retrieval

Official code for the IEEE Transactions on Multimedia (TMM) manuscript:

> *Coverless Image Steganography Based on Geometric Subclass Mapping and Cluster-Free Retrieval*

This repository reproduces the **proposed pipeline** (adaptive geometric subclasses, least-load packing, VLAD-64 shortlist + SIFT–RANSAC verification) and the main **Cluster-SIFT** ablation used in the paper.

## What is included

| Path | Role in the paper |
|---|---|
| `stego_exp/` | Core library: attacks, codebook, geometry atoms, VLAD/SIFT retrieval, hide/recover |
| `geo_refine/` | Detector wrappers, adaptive geometric subclass schemes |
| `full_exp/` | Full-dataset proposed runs (capacity + 22-attack robustness) |
| `full_exp_vlad_sift/` | VLAD bank build / VLAD–SIFT recovery |
| `full_exp_cluster_sift/` | Cluster-SIFT ablation (Table VII / A5-style controls) |
| `_smoke_vlad_pipeline.py` | Quick sanity check |
| `_ablation_run.py`, `_bench_timing_voc2012.py`, `_make_fig_*.py` | Paper tables/figures helpers |

**Not included (by design):** VOC/COCO images, YOLO/Faster R-CNN weight binaries, large VLAD caches, raw result dumps, and third-party PDFs. Download public datasets yourself; weights are fetched by Ultralytics/torchvision or placed under a local path (see below).

## Environment

- Python 3.10+ recommended
- CUDA GPU recommended for YOLOv8m detection (CPU works but is slow)

```bash
conda create -n cis_geo_vlad python=3.10 -y
conda activate cis_geo_vlad
pip install -r requirements.txt
```

## Datasets

Set absolute paths in `full_exp/config.py` (and sibling configs) to:

- PASCAL VOC 2007 / 2012
- MS COCO 2017 `train2017` + `val2017`

The paper indexes the **full JPEG folder** for VOC 2012 and COCO 2017 train+val (not the Lin–Liang VOC2012 trainval subset / COCO 2014).

## Reproduce (high level)

From the repository root:

```bash
# 1) Smoke test (small)
python _smoke_vlad_pipeline.py

# 2) Proposed method, full protocol (long)
python -m full_exp.run_vlad_proposed

# 3) Cluster-SIFT ablation
python -m full_exp_cluster_sift.run_cluster_sift
```

Exact seeds used in the paper:

- Capacity / parameter study: `42`
- Robustness / ablation: `49`

Default knobs match the manuscript: YOLOv8m, confidence `0.5`, `M=8`, area gate `1.5%`, VLAD-64, `k_pool=40`, segment length `n=6`, Lin–Liang 22-attack set.

## Repository

https://github.com/FreedomLinGang/Coverless-CIS-Geo-VLAD

## Citation

If you use this code, please cite the TMM paper (update volume/pages after publication).

## License

Released for academic research. See `LICENSE` (MIT unless otherwise noted by co-authors).

## Contact

Corresponding authors as listed in the manuscript (Chunqiang Yu, Xianquan Zhang).
