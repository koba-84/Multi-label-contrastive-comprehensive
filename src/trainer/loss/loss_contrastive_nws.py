import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

# Epsilon to avoid divided by 0
DEVICE = 'cuda'
EPS = 1e-8


def compute_or(query_tensor: Tensor, key_tensor: Tensor) -> Tensor:
    """Compute or function

    :param input_tensor: Labels instance
    :type input_tensor: Tensor
    :return: Tensir which is the or representation
    :rtype: B x B
    """
    or_build = query_tensor.unsqueeze(1) + key_tensor
    or_build[or_build==2] = 1
    or_build = or_build.sum(dim=2)
    return or_build


def log_softmax_temp(matrix: Tensor, temp: float, mask: Tensor) -> Tensor:
    """Log Softmax with temperature (and mask to remove element on denominator)

    :param matrix: Matrix where we want to apply our log softmax
    :type matrix: Tensor
    :param temp: Temperature value
    :type temp: float
    :param mask: Remove Mask parameters from denominator
    :type mask: Tensor
    :return: Matrix with softmax apply and remove mask
    :rtype: Tensor
    """
    # Check that the temperature is higher than 0
    assert temp > 0
    # Divide by temperature
    matrix_temp = torch.div(matrix, temp)
    # Get the max for "softmax stability"
    max_columns = torch.max(matrix_temp, dim=1, keepdim=True)[0]
    # Detach max to cut the gradient and remove max for stability
    logits = matrix_temp - max_columns.detach()
    # Apply exp on logits
    exp_logits = torch.exp(logits)
    # Apply log softmax and remove mask + add numerator
    log_prob = logits - torch.log((exp_logits * mask).sum(dim=1, keepdim=True) + EPS)
    # Return log_prob
    return log_prob


def compute_loss_contrastive(
    output_features: Tensor,
    labels_query: Tensor,
    prototype_features: Tensor,
    alpha: float,
    beta: float,
    temp: float,
    agg: str,
    sim_matrix: Tensor,
    key_features: Tensor = None,
    key_labels: Tensor = None,
    queue_features: Tensor = None,
    queue_labels: Tensor = None,
) -> Tensor:
    """Compute NWS loss with DCL (denominator=negatives only) in MoCo style.

    - Reference set excludes in-batch; uses key, queue, and prototypes only.
    - Denominator mask = negatives only, weighted by beta (key/queue) or 1 (proto) and a_ir.
    - Numerator weights follow MSC-style w, but only for positives (via pos_mask).
    """
    total = labels_query.size(0)
    nb_labels = prototype_features.size(0)
    dtype = output_features.dtype

    parts_features = []
    parts_weights = []
    pos_masks = []
    neg_masks = []
    section_lengths = []
    
    # Initialize normalize to zero (will be accumulated from key/queue sections)
    normalize = torch.zeros(labels_query.size(0), nb_labels, 
                           dtype=labels_query.dtype, device=labels_query.device)

    # Prepare key section
    mask_and_key = None
    if key_features is not None and key_labels is not None and key_features.numel() > 0:
        key_features = key_features.to(dtype)
        key_labels = key_labels.to(labels_query.dtype)
        # MSC-style mask_and/mask_or for numerator weights
        mask_and_key = torch.einsum('ac,kc->akc', labels_query, key_labels)
        mask_or_key = compute_or(query_tensor=labels_query, key_tensor=key_labels)
        mask_or_key = mask_or_key.unsqueeze(2).clamp_min(1.0)
        mask_and_key = mask_and_key * (1 / mask_or_key) * alpha
        # Normalize denom per class
        normalize = normalize + mask_and_key.sum(dim=1)  # B x L
        # Queue will add later if exists; proto term added after sections collected

        similarity_query_key = output_features @ key_features.T  # B x K
        parts_features.append(similarity_query_key)
        section_lengths.append(key_features.size(0))

        # Pos/Neg masks for DCL
        pos_mask_key = (labels_query @ key_labels.T > 0).to(similarity_query_key.dtype)  # B x K
        neg_mask_key = (1.0 - pos_mask_key)
        pos_masks.append(pos_mask_key)
        neg_masks.append(neg_mask_key)
    else:
        key_features = None
        key_labels = None

    # Prepare queue section
    mask_and_queue = None
    if queue_features is not None and queue_labels is not None and queue_features.numel() > 0:
        queue_features = queue_features.to(dtype)
        queue_labels = queue_labels.to(labels_query.dtype)
        mask_and_queue = torch.einsum('ac,qc->aqc', labels_query, queue_labels)
        mask_or_queue = compute_or(query_tensor=labels_query, key_tensor=queue_labels)
        mask_or_queue = mask_or_queue.unsqueeze(2).clamp_min(1.0)
        mask_and_queue = mask_and_queue * (1 / mask_or_queue) * alpha

        normalize = normalize + mask_and_queue.sum(dim=1)

        similarity_query_queue = output_features @ queue_features.T  # B x Q
        parts_features.append(similarity_query_queue)
        section_lengths.append(queue_features.size(0))

        pos_mask_queue = (labels_query @ queue_labels.T > 0).to(similarity_query_queue.dtype)
        neg_mask_queue = (1.0 - pos_mask_queue)
        pos_masks.append(pos_mask_queue)
        neg_masks.append(neg_mask_queue)
    else:
        queue_features = None
        queue_labels = None
        mask_and_queue = None

    # Prototype section
    similarity_query_proto = output_features @ prototype_features.T  # B x L
    parts_features.append(similarity_query_proto)
    section_lengths.append(nb_labels)
    pos_mask_proto = labels_query.to(similarity_query_proto.dtype)  # B x L
    neg_mask_proto = (1.0 - pos_mask_proto)
    pos_masks.append(pos_mask_proto)
    neg_masks.append(neg_mask_proto)

    # Complete normalize with prototype term (MSC-style)
    normalize = normalize + (- alpha / labels_query.sum(dim=1, keepdim=True) + 1) * labels_query

    # Finalize numerator weights per section (sum over label dim then divide by normalize per class)
    denom = normalize.unsqueeze(1) + EPS  # B x 1 x L
    if mask_and_key is not None:
        w_features_key = (mask_and_key / denom).sum(dim=2)  # B x K
        parts_weights.append(w_features_key)
    if mask_and_queue is not None:
        w_features_queue = (mask_and_queue / denom).sum(dim=2)  # B x Q
        parts_weights.append(w_features_queue)
    # Prototype weights
    w_features_proto = labels_query / (normalize + EPS)  # B x L
    parts_weights.append(w_features_proto)

    # Concat features, weights, pos/neg masks
    final_features = torch.cat(parts_features, dim=1)
    w = torch.cat(parts_weights, dim=1)
    pos_mask_concat = torch.cat(pos_masks, dim=1)
    neg_mask_concat = torch.cat(neg_masks, dim=1)

    total_refs = sum(section_lengths)
    # Section coefficients beta: key/queue=beta, proto=1
    normalize_mask = output_features.new_full((total_refs,), beta)
    normalize_mask[total_refs - nb_labels:] = 1.0

    # Compute sim_agg for a_ir per section
    S = sim_matrix  # L x L, on same device as features via caller
    # mean agg or max agg
    def mean_agg(A: Tensor, B: Tensor) -> Tensor:
        # A: BxL, B: RxL -> returns BxR
        return (A @ S @ B.T) / (A.sum(1, keepdim=True) @ B.sum(1, keepdim=True).T + EPS)

    a_ir_list = []
    # key
    if key_labels is not None:
        if agg == 'mean':
            sim_agg_key = mean_agg(labels_query, key_labels)
        elif agg == 'max':
            # BxKxLxL mask and max
            mask = labels_query.bool().unsqueeze(1).unsqueeze(3) & key_labels.bool().unsqueeze(0).unsqueeze(2)
            sim_all = S.unsqueeze(0).unsqueeze(0).expand(labels_query.size(0), key_labels.size(0), -1, -1)
            sim_agg_key = sim_all.masked_fill(~mask, float('-inf')).amax(dim=(2, 3))
        else:
            raise ValueError('agg must be "mean" or "max"')
        a_key = 1.0 - sim_agg_key
        a_ir_list.append(a_key)
    # queue
    if queue_labels is not None:
        if agg == 'mean':
            sim_agg_queue = mean_agg(labels_query, queue_labels)
        elif agg == 'max':
            mask = labels_query.bool().unsqueeze(1).unsqueeze(3) & queue_labels.bool().unsqueeze(0).unsqueeze(2)
            sim_all = S.unsqueeze(0).unsqueeze(0).expand(labels_query.size(0), queue_labels.size(0), -1, -1)
            sim_agg_queue = sim_all.masked_fill(~mask, float('-inf')).amax(dim=(2, 3))
        else:
            raise ValueError('agg must be "mean" or "max"')
        a_queue = 1.0 - sim_agg_queue
        a_ir_list.append(a_queue)
    # proto: a_ir=1
    a_proto = torch.ones_like(similarity_query_proto)
    a_ir_list.append(a_proto)

    a_ir_concat = torch.cat(a_ir_list, dim=1)

    # Denominator mask: negatives only, apply beta section coeff and a_ir
    mask_diagonal = neg_mask_concat * normalize_mask.unsqueeze(0) * a_ir_concat

    # Numerator weights: only positives
    w_numerator = w * pos_mask_concat

    # Compute masked log-softmax with temperature
    log_softmax = log_softmax_temp(matrix=final_features, temp=temp, mask=mask_diagonal)
    return - (log_softmax * w_numerator).sum(dim=1)


def constrative(output_features: Tensor,
                labels_query: Tensor,
                prototype_features: Tensor,
                alpha: float,
                beta: float,
                temp: float,
                agg: str,
                sim_matrix: Tensor,
                key_features: Tensor = None,
                key_labels: Tensor = None,
                queue_features: Tensor = None,
                queue_labels: Tensor = None) -> Tensor:
    """_summary_

    :param features_labels: Features of the CLS representation shape BxF
    :type features_labels: Tensor
    :param labels_features: Label inside the batch of shape BxL
    :type labels_features: Tensor
    :param features_prototype: Features of prototype of shape LxF
    :type features_prototype: Tensor
    :param labels_prototype: Label of prototype
    :type labels_prototype: Tensor
    :param alpha: Parameter to controle the long tailed distribution
    :type alpha: float
    :param temp: Temperature for Contrastive Learning
    :type temp: float
    :return: Loss for contrastive Learning
    :rtype: Tensor
    """
    # Normalize by the number of labels for each instance in the first time
    loss_contrastive = compute_loss_contrastive(output_features=output_features,
                                                labels_query=labels_query,
                                                prototype_features=prototype_features,
                                                alpha=alpha,
                                                beta=beta,
                                                temp=temp,
                                                agg=agg,
                                                sim_matrix=sim_matrix,
                                                key_features=key_features,
                                                key_labels=key_labels,
                                                queue_features=queue_features,
                                                queue_labels=queue_labels) / (labels_query.sum(dim=1) + EPS)
    return loss_contrastive.mean()


class LossContrastiveNWS(nn.Module):
    """Contrastive Loss for Multi-Label (NWS with DCL, MoCo準拠)

    API: LossContrastiveNWS(alpha, beta, temp, agg, sim)
    - agg: 'mean' or 'max' (必須)
    - sim: LxL label-pair similarity (np.ndarray or torch.Tensor)
    """
    def __init__(self,
                 alpha: float,
                 beta: float,
                 sim,
                 agg: str='mean',
                 temp: float=0.07) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temp = temp
        if agg not in ('mean', 'max'):
            raise ValueError('agg must be "mean" or "max"')
        self.agg = agg
        # to float32 tensor; device is set in forward
        self.sim = torch.as_tensor(sim, dtype=torch.float32)

    def forward(self,
                output_query: Tensor,
                labels_query: Tensor,
                prototype: Tensor,
                queue_feats: Tensor=None,
                queue_labels: Tensor=None,
                key_feats: Tensor=None,
                key_labels: Tensor=None):
        # Apply projection on label features
        normalize_features_query = F.normalize(output_query, dim=-1, p=2)
        normalize_prototype = F.normalize(prototype, dim=-1, p=2)
        key_features = None
        key_labels_tensor = None
        if key_feats is not None and key_labels is not None and key_feats.numel() > 0:
            key_features = F.normalize(key_feats.detach().to(normalize_features_query.dtype), dim=-1, p=2)
            key_labels_tensor = key_labels.detach().to(labels_query.dtype)

        queue_features = None
        queue_labels_tensor = None
        if queue_feats is not None and queue_labels is not None and queue_feats.numel() > 0:
            queue_features = F.normalize(queue_feats.detach().to(normalize_features_query.dtype), dim=-1, p=2)
            queue_labels_tensor = queue_labels.detach().to(labels_query.dtype)
        # 評価時などでモーメンタムエンコーダやキューが利用できない場合、自身のバッチを参照に使う
        if key_features is None and queue_features is None:
            key_features = normalize_features_query.detach()
            key_labels_tensor = labels_query.detach()
        # Move sim to proper device
        sim_matrix = self.sim.to(normalize_features_query.device)
        # Apply Contrastive function
        return constrative(output_features=normalize_features_query,
                           labels_query=labels_query,
                           prototype_features=normalize_prototype,
                           alpha=self.alpha,
                           beta=self.beta,
                           temp=self.temp,
                           agg=self.agg,
                           sim_matrix=sim_matrix,
                           key_features=key_features,
                           key_labels=key_labels_tensor,
                           queue_features=queue_features,
                           queue_labels=queue_labels_tensor)


def compute_label_pair_similarity(Y, method: str):
    """Compute LxL label-pair similarity matrix from N x L multi-hot labels.

    method: 'npmi' or 'jaccard' (必須)
    Returns np.ndarray[float32] of shape LxL with diagonal 1.0.
    """
    if isinstance(Y, torch.Tensor):
        Y_np = Y.detach().cpu().numpy()
    else:
        Y_np = np.asarray(Y)
    assert Y_np.ndim == 2
    N, L = Y_np.shape
    if method == 'jaccard':
        intersection = (Y_np.T @ Y_np).astype(np.float32)  # LxL
        sum_i = Y_np.sum(axis=0, keepdims=False).astype(np.float32)  # L
        union = sum_i[:, None] + sum_i[None, :] - intersection
        # boundary cases not expected per spec; avoid divide-by-zero handling
        S = intersection / union
        np.fill_diagonal(S, 1.0)
        return S.astype(np.float32)
    elif method == 'npmi':
        counts = Y_np.sum(axis=0).astype(np.float64)  # L
        p_i = counts / float(N)
        counts_ij = (Y_np.T @ Y_np).astype(np.float64)  # LxL
        p_ij = counts_ij / float(N)
        # PMI = log(p_ij / (p_i p_j)), NPMI = PMI / (-log p_ij)
        with np.errstate(divide='ignore', invalid='ignore'):
            denom = (p_i[:, None] * p_i[None, :])
            PMI = np.log((p_ij + EPS) / (denom + EPS))
            NPMI = PMI / (-np.log(p_ij + EPS))
        # map [-1,1] -> [0,1]
        S = np.maximum(NPMI, 0.0)
        np.fill_diagonal(S, 0.0)
        return S.astype(np.float32)
    elif method == 'ppmi':
        # PPMI + logistic compression
        counts = Y_np.sum(axis=0).astype(np.float64)            # (L,)
        p_i = counts / float(N)
        counts_ij = (Y_np.T @ Y_np).astype(np.float64)          # (L,L)
        p_ij = counts_ij / float(N)

        with np.errstate(divide='ignore', invalid='ignore'):
            denom = (p_i[:, None] * p_i[None, :])
            PMI = np.log((p_ij + EPS) / (denom + EPS))          # (L,L)

        # 統計量はオフダイアゴナル & 有限値から算出
        offdiag = ~np.eye(PMI.shape[0], dtype=bool)
        valid = np.isfinite(PMI) & offdiag
        if np.any(valid):
            mu = PMI[valid].mean()
            sd = PMI[valid].std(ddof=0)
        else:
            mu, sd = 0.0, 1.0  # フォールバック

        # ロジスティック圧縮（標準化後にσ）
        Z = (PMI - mu) / (sd + EPS)
        S = 1.0 / (1.0 + np.exp(-Z))

        # PPMI化：負のPMIは寄与させない（0）
        S[PMI <= 0] = 0.0

        # 対角は類似度1.0に
        np.fill_diagonal(S, 1.0)
        return S.astype(np.float32)
    elif method == 'pmi_ratio':
        # Power-law-like similarity via cooccurrence ratio r = p_ij / (p_i p_j)
        counts = Y_np.sum(axis=0).astype(np.float64)      # (L,)
        Nf = float(N)
        p_i = counts / Nf
        counts_ij = (Y_np.T @ Y_np).astype(np.float64)    # (L,L)

        # --- Dirichlet/Laplace平滑で連続化（λは小さめでOK） ---
        lam = 1.0
        p_ij = (counts_ij + lam) / (Nf + lam * 4.0)       # 2x2の4セル平滑の素朴版
        denom = ( (counts + 2*lam) / (Nf + 2*lam) )
        denom = denom[:, None] * denom[None, :]           # p(i)p(j) に相当

        r = (p_ij) / (denom + EPS)                        # 比率 r = e^{PMI}

        # --- 冪乗圧縮 + ロジスティックで(0,1)へ ---
        alpha = 0.25                                      # ←尾を重くしたいなら 0.25〜0.5
        r_alpha = np.power(r, alpha)

        # 温度Tで滑らかさ調整（T>1でなだらかに）
        T = 1.5
        S = r_alpha / (1.0 + r_alpha)**(1.0/T)

        np.fill_diagonal(S, 1.0)
        return S.astype(np.float32)

    else:
        raise ValueError("method must be 'npmi' or 'jaccard'") # noqa: E402
