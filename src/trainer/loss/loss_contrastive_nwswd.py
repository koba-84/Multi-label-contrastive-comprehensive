import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from .loss_contrastive_nws import compute_or, log_softmax_temp, EPS


def compute_loss_contrastive_nwswd(
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
    """Compute NWS loss without Decoupled Contrastive Learning (分母に正例を含める版)."""
    total = labels_query.size(0)
    nb_labels = prototype_features.size(0)
    dtype = output_features.dtype

    parts_features = []
    parts_weights = []
    pos_masks = []
    neg_masks = []
    section_lengths = []

    normalize = torch.zeros(labels_query.size(0),
                            nb_labels,
                            dtype=labels_query.dtype,
                            device=labels_query.device)

    # key branch
    mask_and_key = None
    if key_features is not None and key_labels is not None and key_features.numel() > 0:
        key_features = key_features.to(dtype)
        key_labels = key_labels.to(labels_query.dtype)
        mask_and_key = torch.einsum('ac,kc->akc', labels_query, key_labels)
        mask_or_key = compute_or(query_tensor=labels_query, key_tensor=key_labels)
        mask_or_key = mask_or_key.unsqueeze(2).clamp_min(1.0)
        mask_and_key = mask_and_key * (1 / mask_or_key) * alpha

        normalize = normalize + mask_and_key.sum(dim=1)

        similarity_query_key = output_features @ key_features.T
        parts_features.append(similarity_query_key)
        section_lengths.append(key_features.size(0))

        pos_mask_key = (labels_query @ key_labels.T > 0).to(similarity_query_key.dtype)
        neg_mask_key = (1.0 - pos_mask_key)
        pos_masks.append(pos_mask_key)
        neg_masks.append(neg_mask_key)
    else:
        key_features = None
        key_labels = None

    # queue branch
    mask_and_queue = None
    if queue_features is not None and queue_labels is not None and queue_features.numel() > 0:
        queue_features = queue_features.to(dtype)
        queue_labels = queue_labels.to(labels_query.dtype)
        mask_and_queue = torch.einsum('ac,qc->aqc', labels_query, queue_labels)
        mask_or_queue = compute_or(query_tensor=labels_query, key_tensor=queue_labels)
        mask_or_queue = mask_or_queue.unsqueeze(2).clamp_min(1.0)
        mask_and_queue = mask_and_queue * (1 / mask_or_queue) * alpha

        normalize = normalize + mask_and_queue.sum(dim=1)

        similarity_query_queue = output_features @ queue_features.T
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

    # prototype branch
    similarity_query_proto = output_features @ prototype_features.T
    parts_features.append(similarity_query_proto)
    section_lengths.append(nb_labels)
    pos_mask_proto = labels_query.to(similarity_query_proto.dtype)
    neg_mask_proto = (1.0 - pos_mask_proto)
    pos_masks.append(pos_mask_proto)
    neg_masks.append(neg_mask_proto)

    normalize = normalize + (- alpha / labels_query.sum(dim=1, keepdim=True) + 1) * labels_query

    denom = normalize.unsqueeze(1) + EPS
    if mask_and_key is not None:
        w_features_key = (mask_and_key / denom).sum(dim=2)
        parts_weights.append(w_features_key)
    if mask_and_queue is not None:
        w_features_queue = (mask_and_queue / denom).sum(dim=2)
        parts_weights.append(w_features_queue)
    w_features_proto = labels_query / (normalize + EPS)
    parts_weights.append(w_features_proto)

    final_features = torch.cat(parts_features, dim=1)
    w = torch.cat(parts_weights, dim=1)
    pos_mask_concat = torch.cat(pos_masks, dim=1)
    neg_mask_concat = torch.cat(neg_masks, dim=1)

    total_refs = sum(section_lengths)
    normalize_mask = output_features.new_full((total_refs,), beta)
    normalize_mask[total_refs - nb_labels:] = 1.0

    S = sim_matrix

    def mean_agg(A: Tensor, B: Tensor) -> Tensor:
        return (A @ S @ B.T) / (A.sum(1, keepdim=True) @ B.sum(1, keepdim=True).T + EPS)

    a_ir_list = []
    if key_labels is not None:
        if agg == 'mean':
            sim_agg_key = mean_agg(labels_query, key_labels)
        elif agg == 'max':
            mask = labels_query.bool().unsqueeze(1).unsqueeze(3) & key_labels.bool().unsqueeze(0).unsqueeze(2)
            sim_all = S.unsqueeze(0).unsqueeze(0).expand(labels_query.size(0), key_labels.size(0), -1, -1)
            sim_agg_key = sim_all.masked_fill(~mask, float('-inf')).amax(dim=(2, 3))
        else:
            raise ValueError('agg must be "mean" or "max"')
        a_key = 1.0 - sim_agg_key
        a_ir_list.append(a_key)
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
    a_proto = torch.ones_like(similarity_query_proto)
    a_ir_list.append(a_proto)

    a_ir_concat = torch.cat(a_ir_list, dim=1)

    # 分母に正例を含める（正例の重みは1）
    neg_component = neg_mask_concat * normalize_mask.unsqueeze(0) * a_ir_concat
    pos_component = pos_mask_concat
    mask_diagonal = neg_component + pos_component

    w_numerator = w * pos_mask_concat

    log_softmax = log_softmax_temp(matrix=final_features, temp=temp, mask=mask_diagonal)
    return - (log_softmax * w_numerator).sum(dim=1)


def constrative_nwswd(output_features: Tensor,
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
    """NWSWDコントラスト損失"""
    loss_contrastive = compute_loss_contrastive_nwswd(output_features=output_features,
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


class LossContrastiveNWSWD(nn.Module):
    """NWS（Without Decouple）損失"""
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
        self.sim = torch.as_tensor(sim, dtype=torch.float32)

    def forward(self,
                output_query: Tensor,
                labels_query: Tensor,
                prototype: Tensor,
                queue_feats: Tensor=None,
                queue_labels: Tensor=None,
                key_feats: Tensor=None,
                key_labels: Tensor=None):
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

        if key_features is None and queue_features is None:
            key_features = normalize_features_query.detach()
            key_labels_tensor = labels_query.detach()

        sim_matrix = self.sim.to(normalize_features_query.device)

        return constrative_nwswd(output_features=normalize_features_query,
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
