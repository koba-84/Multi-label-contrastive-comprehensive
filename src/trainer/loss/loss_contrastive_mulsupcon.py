import torch
from torch import nn
from torch import Tensor
# Epsilon to avoid divided by 0
DEVICE = 'cuda'
EPS = 1e-8
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_



def generate_output_MulSupCon(batch_labels, ref_labels, scores):
    """
    MulSupCon

    Parameters:
        batch_labels: B x C tensor, labels of the anchor
        ref_labels: Q x C tensor, labels of samples from queue
        scores: B x Q tensor, cosine similarity between the anchor and samples from queue
    """
    device = scores.device
    indices = torch.nonzero(batch_labels == 1, as_tuple=False)

    if indices.numel() == 0:
        empty_scores = scores.new_zeros((0, scores.shape[1]))
        empty_masks = torch.zeros((0, ref_labels.shape[0]), dtype=torch.long, device=device)
        empty_weights = torch.zeros((0,), device=device)
        return empty_scores, [empty_masks, empty_weights]

    selected_scores = scores[indices[:, 0]]
    labels = torch.zeros((indices.shape[0], batch_labels.shape[1]),
                         device=device,
                         dtype=batch_labels.dtype)
    labels[torch.arange(indices.shape[0], device=device), indices[:, 1]] = 1
    masks = (labels @ ref_labels.T) > 0

    weights_per_sample = torch.full((selected_scores.shape[0],),
                                    1.0 / max(selected_scores.shape[0], 1),
                                    device=device,
                                    dtype=torch.float32)
    return selected_scores, [masks.to(torch.long), weights_per_sample]


class WeightedSupCon(nn.Module):
    def __init__(self, temperature=0.1):
        super(WeightedSupCon, self).__init__()
        self.temperature = temperature

    def forward(self, score, ref):
        mask, weight = ref
        if score.numel() == 0 or mask.numel() == 0:
            return score.new_tensor(0.0)
        mask = mask.to(score.dtype)
        num_pos = mask.sum(1).clamp_min(EPS)
        log_prob = F.log_softmax(score / self.temperature, dim=1)
        loss = - (log_prob * mask).sum(1) / num_pos
        return (loss * weight).sum()


class LossContrastiveMulSupCon(nn.Module):
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
        self.loss = WeightedSupCon(temperature=temp)

    def forward(self,
                output_query: Tensor,
                labels_query: Tensor,
                prototype: Tensor=None,
                queue_feats: Tensor=None,
                queue_labels: Tensor=None,
                key_feats: Tensor=None,
                key_labels: Tensor=None) -> Tensor:
        """Compute the mulsupcon loss function

        :param output_query: features of the output
        :type output_query: Tensor
        :param labels_query: label of the query
        :type labels_query: Tensor
        :param prototype: Prototype but useless in this case, defaults to None
        :type prototype: Tensor, optional
        :return: loss funtion error
        :rtype: Tensor
        """
        # Apply projection on label features
        normalize_features_query = F.normalize(output_query, dim=-1, p=2)

        references_features = []
        references_labels = []

        if key_feats is not None and key_labels is not None:
            references_features.append(F.normalize(key_feats, dim=-1, p=2))
            references_labels.append(key_labels)

        if queue_feats is not None and queue_labels is not None and queue_feats.numel() > 0:
            references_features.append(F.normalize(queue_feats, dim=-1, p=2))
            references_labels.append(queue_labels)

        if not references_features:
            references_features.append(normalize_features_query)
            references_labels.append(labels_query)

        references_features = torch.cat(references_features, dim=0)
        references_labels = torch.cat(references_labels, dim=0)

        # Apply Contrastive function
        score, ref = generate_output_MulSupCon(labels_query,
                                               references_labels,
                                               normalize_features_query @ references_features.T)
        return self.loss(score=score, ref=ref)

