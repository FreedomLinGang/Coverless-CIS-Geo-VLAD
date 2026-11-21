"""Offline global descriptors (no pretrained weight download required)."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans


def pil_gray(img: Image.Image, size=256) -> np.ndarray:
    return cv2.cvtColor(np.asarray(img.convert("RGB").resize((size, size))), cv2.COLOR_RGB2GRAY)


def embed_image(img: Image.Image) -> np.ndarray:
    """Compact handcrafted descriptor: color hist + edge hist + ORB desc mean."""
    rgb = np.asarray(img.convert("RGB").resize((256, 256)))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    ang = cv2.phase(gx, gy, angleInDegrees=True)
    edge_hist = cv2.calcHist([ang.astype(np.float32)], [0], (mag > 30).astype(np.uint8), [16], [0, 360]).flatten()
    orb = cv2.ORB_create(nfeatures=200)
    _, desc = orb.detectAndCompute(gray, None)
    if desc is None:
        orb_feat = np.zeros(32, dtype=np.float32)
    else:
        orb_feat = desc.astype(np.float32).mean(axis=0)
    feat = np.concatenate(
        [
            hist_h / (hist_h.sum() + 1e-6),
            hist_s / (hist_s.sum() + 1e-6),
            hist_v / (hist_v.sum() + 1e-6),
            edge_hist / (edge_hist.sum() + 1e-6),
            orb_feat / (np.linalg.norm(orb_feat) + 1e-6),
        ]
    ).astype(np.float32)
    feat = feat / (np.linalg.norm(feat) + 1e-8)
    return feat


class FeatureBank:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.paths: list[str] = []
        self.feats: np.ndarray | None = None
        self.clusters: dict[str, int] = {}

    def embed(self, img: Image.Image) -> np.ndarray:
        return embed_image(img)

    def build(self, path_to_img: dict[str, Image.Image], n_clusters: int = 64):
        self.paths = list(path_to_img.keys())
        feats = [self.embed(path_to_img[p]) for p in self.paths]
        self.feats = np.stack(feats, axis=0)
        k = min(n_clusters, max(2, len(self.paths) // 4))
        km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=256, n_init=3)
        labels = km.fit_predict(self.feats)
        self.clusters = {p: int(labels[i]) for i, p in enumerate(self.paths)}
        self.km = km

    def cluster_of(self, path: str) -> int:
        return self.clusters[path]

    def candidates_by_cluster(self, cluster_id: int) -> list[str]:
        return [p for p, c in self.clusters.items() if c == cluster_id]

    def recall_by_feature(self, query_img: Image.Image, topk: int = 20) -> list[str]:
        q = self.embed(query_img)
        sims = self.feats @ q
        idx = np.argsort(-sims)[:topk]
        return [self.paths[i] for i in idx]


def ssim_score(a: Image.Image, b: Image.Image) -> float:
    from skimage.metrics import structural_similarity as ssim

    aa = np.asarray(a.convert("L").resize((256, 256)))
    bb = np.asarray(b.convert("L").resize((256, 256)))
    return float(ssim(aa, bb, data_range=255))


def orb_inlier_ratio(a: Image.Image, b: Image.Image) -> float:
    ia = cv2.cvtColor(np.asarray(a.convert("RGB").resize((512, 512))), cv2.COLOR_RGB2GRAY)
    ib = cv2.cvtColor(np.asarray(b.convert("RGB").resize((512, 512))), cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=800)
    ka, da = orb.detectAndCompute(ia, None)
    kb, db = orb.detectAndCompute(ib, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(da, db, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 8:
        return len(good) / 50.0
    src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if mask is None:
        return 0.0
    return float(mask.sum()) / float(len(mask))


def retrieve_ssim(query: Image.Image, candidates, path_to_img: dict[str, Image.Image], max_cand: int = 40) -> str:
    cands = list(candidates)
    if len(cands) > max_cand:
        # keep a random subset for speed; include more by taking head then sample
        cands = cands[:max_cand]
    best_p, best_s = None, -1.0
    for p in cands:
        s = ssim_score(query, path_to_img[p])
        if s > best_s:
            best_s, best_p = s, p
    return best_p


def ahash(img: Image.Image, hash_size: int = 8) -> np.ndarray:
    g = np.asarray(img.convert("L").resize((hash_size, hash_size)), dtype=np.float32)
    return (g > g.mean()).astype(np.uint8).flatten()


def ahash_sim(a: Image.Image, b: Image.Image) -> float:
    ha, hb = ahash(a), ahash(b)
    return float((ha == hb).mean())


def retrieve_fast_hybrid(
    query: Image.Image,
    bank: FeatureBank,
    path_to_img,
    cluster_hint: int | None = None,
    topk: int = 30,
) -> str:
    """Fast retrieval: feature recall + aHash + light ORB on shortlist."""
    recalled = bank.recall_by_feature(query, topk=topk)
    if cluster_hint is not None:
        cluster_set = set(bank.candidates_by_cluster(cluster_hint))
        # prefer same-cluster recalls, then others
        mixed = [p for p in recalled if p in cluster_set] + [p for p in recalled if p not in cluster_set]
        # also add a few from cluster by aHash prefilter if needed
        if len([p for p in mixed if p in cluster_set]) < 5:
            for p in list(cluster_set)[:80]:
                if p not in mixed:
                    mixed.append(p)
                if len(mixed) >= topk:
                    break
        seen, candidates = set(), []
        for p in mixed:
            if p not in seen:
                seen.add(p)
                candidates.append(p)
            if len(candidates) >= topk:
                break
    else:
        candidates = recalled[:topk]

    # rank by aHash then refine top-8 with ORB
    scored = [(ahash_sim(query, path_to_img[p]), p) for p in candidates]
    scored.sort(reverse=True)
    short = [p for _, p in scored[:8]]
    best_p, best_s = scored[0][1], -1.0
    qf = bank.embed(query)
    path_index = {p: i for i, p in enumerate(bank.paths)}
    for p in short:
        orb = orb_inlier_ratio(query, path_to_img[p])
        pf = bank.feats[path_index[p]]
        sim = float(qf @ pf)
        ah = ahash_sim(query, path_to_img[p])
        score = 0.50 * orb + 0.25 * max(0.0, sim) + 0.25 * ah
        if score > best_s:
            best_s, best_p = score, p
    return best_p


def retrieve_proposed(
    query: Image.Image,
    bank: FeatureBank,
    path_to_img: dict[str, Image.Image],
    cluster_hint: int | None = None,
    topk: int = 20,
) -> str:
    recalled = bank.recall_by_feature(query, topk=topk)
    if cluster_hint is not None:
        cluster_set = set(bank.candidates_by_cluster(cluster_hint))
        mixed = [p for p in recalled if p in cluster_set] + [p for p in recalled if p not in cluster_set]
        seen, candidates = set(), []
        for p in mixed:
            if p not in seen:
                seen.add(p)
                candidates.append(p)
            if len(candidates) >= topk:
                break
    else:
        candidates = recalled

    qf = bank.embed(query)
    path_index = {p: i for i, p in enumerate(bank.paths)}
    best_p, best_s = None, -1.0
    for p in candidates:
        orb = orb_inlier_ratio(query, path_to_img[p])
        pf = bank.feats[path_index[p]]
        sim = float(qf @ pf)
        # Hybrid: geometry (ORB) + appearance (feat/SSIM). SSIM helps hist-eq etc.
        s = ssim_score(query, path_to_img[p])
        score = 0.55 * orb + 0.20 * max(0.0, sim) + 0.25 * max(0.0, s)
        if score > best_s:
            best_s, best_p = score, p
    return best_p
