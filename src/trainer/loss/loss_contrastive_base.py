import torch
from torch import nn
from torch import Tensor
# Epsilon to avoid divided by 0
DEVICE = 'cuda'
EPS = 1e-8
import torch.nn.functional as F


def compute_or(query_tensor: Tensor, key_tensor: Tensor) -> Tensor:
    """ Compute or function between two tensors

    :param query_tensor: First tensor for the or: shape B x L
    :type query_tensor: Tensor
    :param key_tensor: second tensor for the or: shape B x L
    :type key_tensor: Tensor
    :return: Tensor which is an or between the tensor shape B x B
    :rtype: Tensor
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
    # Apply log softmax and remove mask
    log_prob = logits - torch.log((exp_logits * mask).sum(dim=1, keepdim=True))
    # Return log_prob
    return log_prob


def compute_loss_contrastive(output_features: Tensor,
                            labels_query: Tensor,
                            temp: float) -> Tensor:
    """ In this case we have no momentum encoder

    :param output_features: Features instances
    :type output_features: Tensor
    :param labels_query: Label of features
    :type labels_query: Tensor
    :param temp: Temperature features
    :type temp: float
    :return: return loss
    :rtype: Tensor
    """
    # Number of element inside the batch
    total = labels_query.size(0)
    # Mask to remove interaction between same instance
    mask_diagonal = torch.ones(total*total, device='cuda', dtype=torch.uint8)
    mask_diagonal[0::(total+1)] = 0
    mask_diagonal = mask_diagonal.reshape(total, total)
    # Compute similarities between query x query
    similarity_query_query = output_features @ output_features.T
    # Compute and function and remove directly the mask 
    mask_and_features = labels_query @ labels_query.T * mask_diagonal 
    # Compute or function
    mask_or_features = compute_or(query_tensor=labels_query, key_tensor=labels_query)
    # Compute the hamming scores, add EPS to avoid dividing by 0, we must add EPS
    # Because it is possible to have 0 value.
    mask_and_features = mask_and_features * (1/(mask_or_features + EPS))
    # normalize row to obtain the final mask, also possible to have 0, should add eps
    mask = mask_and_features/(mask_and_features.sum(dim=1, keepdim=True) + EPS)
    # Finally mask is just the diagonal mask for the denominator
    log_softmax = log_softmax_temp(matrix=similarity_query_query,
                                   temp=temp,
                                   mask=mask_diagonal)
    # It is possible that some instance to have no interaction we take it into account
    # Should add EPS for the case where no positive pair
    return - (log_softmax * mask).sum(dim=1) / ((mask.sum(dim=1) !=0).int().sum() + EPS)


def constrative(output_features: Tensor,
                labels_query: Tensor,
                temp:float) -> Tensor:
    # Normalize by the number of labels for each instance in the first time
    loss_contrastive = compute_loss_contrastive(output_features=output_features,
                                                labels_query=labels_query,
                                                temp=temp)
    # It is already normalize we do not have to use mean but sum instead of 
    return loss_contrastive.sum()


class LossContrastiveBase(nn.Module):
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
                prototype: Tensor=None) -> Tensor:
        """ Forward of the contrastive loss function

        :param output_query: The instances features
        :type output_query: Tensor
        :param labels_query: Label of the features
        :type labels_query: Tensor
        :param prototype: Prototypes are useless in this case, defaults to None
        :type prototype: Tensor, optional
        :return: Output of the loss function
        :rtype: Tensor
        """
        # Normalize features, to compute a classical cosine similarity
        normalize_features_query = F.normalize(output_query, dim=-1, p=2)
        # Apply Contrastive function
        return constrative(output_features=normalize_features_query,
                           labels_query=labels_query,
                           temp=self.temp)
