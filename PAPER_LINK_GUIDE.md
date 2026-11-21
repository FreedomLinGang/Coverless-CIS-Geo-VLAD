# 论文 / 补充材料中如何写代码链接（TMM）

## 政策要点（实务）

- TMM **鼓励**公开代码与可复现材料，通常**不强制**；公开有助于可信度与外审印象。
- 若审稿为双盲：初稿阶段**不要**放带作者姓名的公开 GitHub；用  
  - “Code will be released upon acceptance”，或  
  - 匿名仓库 / Anonymous GitHub / Zenodo（去掉姓名与单位）。
- 录用后 camera-ready：换成正式公开 URL。

## 推荐写法（省版面、可信）

### 正文 Experiment Setup（一句即可）

> Implementation details and scripts that reproduce the reported capacity and robustness protocols are available at [URL] (also summarized in the Supplementary Material).

### 补充材料 Implementation Notes（推荐主放这里）

增加一条 bullet：

> **Code.** A reference implementation of the proposed pipeline and the Cluster-SIFT ablation is released at `https://github.com/<org>/<repo>` (MIT). Public VOC/COCO images and detector weights are not redistributed; paths and seeds match the main paper (`42` / `49`).

## 不建议

- 在摘要里放链接（占字且初审不看重）。
- 初稿就放含真名的仓库（若双盲）。
- 把整盘 `paper_5_our_code`（含 cache、PDF、权重）直接上传。
