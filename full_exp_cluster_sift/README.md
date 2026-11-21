# full_exp_cluster_sift
#
# 有簇 ID 基线（SIFT-mean k-means → aux cluster ID → 簇内 SIFT-J）。
# **正式接收端方案已改为无簇 VLAD-SIFT**，见 full_exp_vlad_sift/ 与 stego_exp/vlad_retrieval.py。
#
# K: VOC=256, COCO=1024（config 中 N_CLUSTERS 用于建库）
#
# Results: results/cluster_sift_results.md , full_exp_vlad_sift/results/COMPARE.md
