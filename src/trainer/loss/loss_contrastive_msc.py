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
                            temp:float) -> Tensor:
    total = labels_query.size(0)
    # Save Number of Labels
    nb_labels = prototype_features.size(0)
    # Mask to remove interaction between same instance
    mask_diagonal = torch.ones(total*(total + nb_labels), device='cuda', dtype=torch.uint8)
    mask_diagonal[0::(total + nb_labels)+1] = 0
    mask_diagonal = mask_diagonal.reshape(total, (total + nb_labels))
    # Compute similarities between queryxquery
    similarity_query_query = output_features @ output_features.T
    # Comptue similarities between queryxprototypes
    similarity_query_prototype = output_features @ prototype_features.T
    similarity_features = similarity_query_query
    # print(labels_query.shape, label_query_key.shape)
    mask_and_features = torch.einsum('ac, bc -> abc', labels_query, labels_query)
    mask_or_features = compute_or(query_tensor=labels_query, key_tensor=labels_query)
    mask_and_features = mask_and_features * (1/mask_or_features.unsqueeze(2)) * alpha
    # print(mask_and_features, mask_and_features.shape, label_query_key.shape)
    normalize = mask_and_features.sum(dim=1) + (- alpha / labels_query.sum(dim=1, keepdim=True) + 1) * labels_query
    w_features_features = (mask_and_features/(normalize.unsqueeze(1) + EPS)).sum(dim=2)
    w_features_proto = labels_query /(normalize + EPS)
    # Compute final w, we have to remove the diagonal
    w = torch.cat((w_features_features, w_features_proto), dim=1) * mask_diagonal
    # print(w.sum(dim=1))
    # Concatenate features and proto features
    final_features = torch.cat((similarity_features, similarity_query_prototype), dim=1)
    # Normalize denominator
    normalize_mask = torch.ones(total + nb_labels, device='cuda') * beta
    normalize_mask[-nb_labels:] = 1
    # Create a diagonal mask to remove identical element
    log_softmax = log_softmax_temp(matrix=final_features,
                                   temp=temp,
                                   mask=mask_diagonal * normalize_mask.unsqueeze(0))
    # Return the negative log softmax, ponderate
    return - (log_softmax * w).sum(dim=1)


def constrative(output_features: Tensor,
                labels_query: Tensor,
                prototype_features: Tensor,
                alpha: float,
                beta: float,
                temp:float) -> Tensor:
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
                                                temp=temp)/labels_query.sum(dim=1)
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
                prototype: Tensor):
        # Apply projection on label features
        normalize_features_query = F.normalize(output_query, dim=-1, p=2)
        normalize_prototype = F.normalize(prototype, dim=-1, p=2)
        # Apply Contrastive function
        return constrative(output_features=normalize_features_query,
                           labels_query=labels_query,
                           prototype_features=normalize_prototype,
                           alpha=self.alpha,
                           beta=self.beta,
                           temp=self.temp)
