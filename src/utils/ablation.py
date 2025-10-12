import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score

def _as_label_combinations(Y: np.ndarray) -> np.ndarray:
    """Multi-hot (B, L) → 各行をユニーク化して組合せID (0..K-1) に変換。"""
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2D (B, L). Got {Y.shape}")
    Yb = (Y > 0).astype(np.uint8)
    keys = np.array([row.tobytes() for row in Yb], dtype=object)
    _, inv = np.unique(keys, return_inverse=True)
    return inv

def _keep_top_fraction_classes(labels: np.ndarray, fraction: float) -> np.ndarray:
    """出現頻度上位 fraction の組合せタイプだけを残す行マスクを返す。"""
    if not (0 < fraction <= 1.0):
        raise ValueError("keep_fraction must be in (0, 1].")
    vals, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)  # 頻度降順
    vals_sorted = vals[order]
    k = max(1, int(np.ceil(len(vals_sorted) * fraction)))
    keep_classes = set(vals_sorted[:k].tolist())
    return np.isin(labels, list(keep_classes))

def evaluate_embedding(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    keep_fraction: float = 0.5,
):
    """
    表現評価（Contrastive表現 × 線形評価の前段の隠れ表現を想定）
    - ラベル組合せをクラスとして扱い、
    - 頻出上位 keep_fraction の組合せタイプのみで
    - Silhouette（↑良）と Davies–Bouldin（↓良）を計算（どちらもEuclidean）。
    戻り値: (silhouette, dbi)
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (B, D). Got {X.shape}")
    if Y.ndim != 2 or len(Y) != len(X):
        raise ValueError(f"Y must be (B, L) and match X. Got X={X.shape}, Y={Y.shape}")

    labels = _as_label_combinations(Y)
    mask = _keep_top_fraction_classes(labels, keep_fraction)
    X_sel, labels_sel = X[mask], labels[mask]

    if len(np.unique(labels_sel)) < 2:
        raise ValueError("Need at least 2 label-combination classes after filtering.")

    # Euclidean固定
    sil = float(silhouette_score(X_sel, labels_sel))          # metric='euclidean' が既定
    dbi = float(davies_bouldin_score(X_sel, labels_sel))      # Euclideanベース

    return sil, dbi
