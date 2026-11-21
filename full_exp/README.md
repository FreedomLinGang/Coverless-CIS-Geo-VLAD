# Full-dataset proposed-method experiments
#
# **Recommended (VLAD-SIFT, no cluster ID in aux):**
#   D:\Anaconda3\envs\torchgpu\python.exe -m full_exp.run_vlad_proposed
#
# Legacy hybrid retrieval (ORB+SSIM, uses cluster for recovery):
#   D:\Anaconda3\envs\torchgpu\python.exe -m full_exp.run_proposed
#
# Datasets (train+val/test combined):
#   VOC2007, VOC2012, COCO2017(train+val)
#
# Stages:
#   1) prepare detections (remap/reuse caches, detect missing)
#   2) adaptive geometric subclass schemes
#   3) VLAD bank + capacity (10B/100B/1KB/2KB) + Table IX robustness
