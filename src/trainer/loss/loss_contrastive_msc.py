import torch
from torch import nn
from torch import Tensor
# Epsilon to avoid divided by 0
DEVICE = 'cuda'
EPS = 1e-8
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_


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
    log_prob = logits - torch.log((exp_logits * mask).sum(dim=1, keepdim=True))
    # Return log_prob
    return log_prob


def compute_loss_contrastive(output_features: Tensor,
                            labels_query: Tensor,
                            prototype_features: Tensor,
                            alpha: float,
                            beta: float,
                            temp: float,
                            key_features: Tensor = None,
                            key_labels: Tensor = None,
                            queue_features: Tensor = None,
                            queue_labels: Tensor = None) -> Tensor:
    total = labels_query.size(0)
    nb_labels = prototype_features.size(0)
    dtype = output_features.dtype

    mask_and_features = torch.einsum('ac, bc -> abc', labels_query, labels_query)
    mask_or_features = compute_or(query_tensor=labels_query, key_tensor=labels_query)
    mask_or_features = mask_or_features.unsqueeze(2).clamp_min(1.0)
    mask_and_features = mask_and_features * (1/mask_or_features) * alpha

    mask_and_key = None
    similarity_query_key = None
    key_total = 0
    if key_features is not None and key_labels is not None and key_features.numel() > 0:
        key_features = key_features.to(dtype)
        key_labels = key_labels.to(labels_query.dtype)
        mask_and_key = torch.einsum('ac,kc->akc', labels_query, key_labels)
        mask_or_key = compute_or(query_tensor=labels_query, key_tensor=key_labels)
        mask_or_key = mask_or_key.unsqueeze(2).clamp_min(1.0)
        mask_and_key = mask_and_key * (1/mask_or_key) * alpha
        similarity_query_key = output_features @ key_features.T
        key_total = key_features.size(0)

    mask_and_queue = None
    similarity_query_queue = None
    queue_total = 0
    if queue_features is not None and queue_labels is not None and queue_features.numel() > 0:
        queue_features = queue_features.to(dtype)
        queue_labels = queue_labels.to(labels_query.dtype)
        mask_and_queue = torch.einsum('ac,qc->aqc', labels_query, queue_labels)
        mask_or_queue = compute_or(query_tensor=labels_query, key_tensor=queue_labels)
        mask_or_queue = mask_or_queue.unsqueeze(2).clamp_min(1.0)
        mask_and_queue = mask_and_queue * (1/mask_or_queue) * alpha
        similarity_query_queue = output_features @ queue_features.T
        queue_total = queue_features.size(0)

    normalize = mask_and_features.sum(dim=1)
    if mask_and_key is not None:
        normalize = normalize + mask_and_key.sum(dim=1)
    if mask_and_queue is not None:
        normalize = normalize + mask_and_queue.sum(dim=1)
    normalize = normalize + (- alpha / labels_query.sum(dim=1, keepdim=True) + 1) * labels_query

    denom = normalize.unsqueeze(1) + EPS
    w_features_features = (mask_and_features/denom).sum(dim=2)

    parts_features = [output_features @ output_features.T]
    parts_weights = [w_features_features]
    section_lengths = [total]

    if mask_and_key is not None:
        w_features_key = (mask_and_key/denom).sum(dim=2)
        parts_features.append(similarity_query_key)
        parts_weights.append(w_features_key)
        section_lengths.append(key_total)

    if mask_and_queue is not None:
        w_features_queue = (mask_and_queue/denom).sum(dim=2)
        parts_features.append(similarity_query_queue)
        parts_weights.append(w_features_queue)
        section_lengths.append(queue_total)

    w_features_proto = labels_query /(normalize + EPS)
    parts_features.append(output_features @ prototype_features.T)
    parts_weights.append(w_features_proto)
    section_lengths.append(nb_labels)

    total_refs = sum(section_lengths)
    mask_diagonal = output_features.new_ones((total, total_refs))
    if total > 0:
        indices = torch.arange(total, device=mask_diagonal.device)
        mask_diagonal[indices, indices] = 0

    w = torch.cat(parts_weights, dim=1) * mask_diagonal
    final_features = torch.cat(parts_features, dim=1)

    normalize_mask = output_features.new_full((total_refs,), beta)
    normalize_mask[total_refs - nb_labels:] = 1

    log_softmax = log_softmax_temp(matrix=final_features,
                                   temp=temp,
                                   mask=mask_diagonal * normalize_mask.unsqueeze(0))
    return - (log_softmax * w).sum(dim=1)


def constrative(output_features: Tensor,
                labels_query: Tensor,
                prototype_features: Tensor,
                alpha: float,
                beta: float,
                temp: float,
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
                                                key_features=key_features,
                                                key_labels=key_labels,
                                                queue_features=queue_features,
                                                queue_labels=queue_labels)/(labels_query.sum(dim=1) + EPS)
    return loss_contrastive.mean()


class LossContrastiveMSC(nn.Module):
    """ Contrastive Loss for Multi-Label

    :param nn: 
    :type nn: _type_
    """
    def __init__(self,
                 alpha: float,
                 beta: float,
                 temp: float=0.07) -> None:
        """Classical init of nn.Module

        :param config: configuration file
        :type config: Dict
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temp = temp

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
        # Apply Contrastive function
        return constrative(output_features=normalize_features_query,
                           labels_query=labels_query,
                           prototype_features=normalize_prototype,
                           alpha=self.alpha,
                           beta=self.beta,
                           temp=self.temp,
                           key_features=key_features,
                           key_labels=key_labels_tensor,
                           queue_features=queue_features,
                           queue_labels=queue_labels_tensor)
