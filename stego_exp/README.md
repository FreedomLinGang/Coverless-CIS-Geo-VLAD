# Coverless Stego Experiments (Proposed Method)

本文方法在 VOC/COCO 图库上的容量与鲁棒实验。正式全量结果见 `full_exp/`。

## 环境

```bash
conda activate torchgpu
cd our_code
```

## 小规模快速实验（VOC2012 GT 框）

```bash
python -m stego_exp.run_main --dataset voc2012 --max-images 600 --segment-bits 4 --capacity-trials 3 --robust-messages 5 --msg-bytes 20
```

结果输出：

- `stego_exp/results/proposed_voc2012_n600.md`
- `stego_exp/results/proposed_voc2012_n600.json`

## 方法要点

| 模块 | 设置 |
|---|---|
| 目标选择 | 面积比阈值自适应（默认 ≥1.5%–2%） |
| 语义码本 | 粗类 + 几何子类（全量实验）/ 可选颜色形态属性 |
| 原图检索 | **VLAD-64 全库粗检索 top-40 + SIFT-J 精排**（`CoverlessSystem(vlad_retriever=...)`） |

接收端 `recover_with_recall()` 在注入 `vlad_retriever` 时**忽略** `StegoRecord.cluster`。

## 正式全量实验

见 `full_exp/run_vlad_proposed.py` 或 `full_exp_vlad_sift/run_vlad_sift.py`（VOC2007/2012 + COCO2017 train+val）。

## COCO 说明

`--dataset coco_val2017` 需要检测权重。若 SSL 下载失败，请手动将权重放到：

`C:\Users\lg\.cache\torch\hub\checkpoints\fasterrcnn_resnet50_fpn_coco-258fb6c6.pth`
