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
    # on récupère les indices ou les labels sont identiques
    indices = torch.where(batch_labels == 1)
    # On garde que les interactions positives, NB_interaction, nb_labels
    scores = scores[indices[0]]
    # on crée un vecteur labels qui contient que les interactions positvies
    labels = torch.zeros(len(scores), batch_labels.shape[1], device=scores.device)
    labels[range(len(labels)), indices[1]] = 1
    # NB interaction, nb_labels
    masks = (labels @ ref_labels.T).to(torch.bool)
    ####### OURS ######
    mask = torch.ones(masks.shape[0], masks.shape[1], dtype=torch.bool)
    mask[range(scores.size(0)), indices[0].reshape(1, -1)] = False
    masks = masks[mask].reshape(mask.size(0), -1)
    scores = scores[mask].reshape(mask.size(0), -1)
    
    ###################
    
    n_score_per_sample = batch_labels.sum(dim=1).to(torch.int16).tolist() #normalize ok
    weights_per_sample = [1 / len(scores) for n in n_score_per_sample for _ in range(n)]
    weights_per_sample = torch.tensor(
        weights_per_sample,
        device=scores.device,
        dtype=torch.float32
    )
    return scores, [masks.to(torch.long), weights_per_sample]


class WeightedSupCon(nn.Module):
    def __init__(self, temperature=0.1):
        super(WeightedSupCon, self).__init__()
        self.temperature = temperature

    def forward(self, score, ref):
        mask, weight = ref
        num_pos = mask.sum(1) + EPS
        loss = - (torch.log(
            (F.softmax(score / self.temperature, dim=1))) * mask).sum(1) / num_pos
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
                prototype: Tensor=None) -> Tensor:
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
        # Apply Contrastive function
        score, ref = generate_output_MulSupCon(labels_query, labels_query, normalize_features_query @ normalize_features_query.T)
        return self.loss(score=score, ref=ref)

