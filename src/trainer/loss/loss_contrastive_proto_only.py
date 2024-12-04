import torch
from torch import nn
from torch import Tensor
# Epsilon to avoid divided by 0
DEVICE = 'cuda'
EPS = 1e-8
import torch.nn.functional as F
# This loss is use with prototype only


def log_softmax_temp(matrix: Tensor, temp: float) -> Tensor:
    """ Log softmax for proto only, we don't need a mask because no interaction between same instance

    :param matrix: Similarity Matrix instance vs prototype
    :type matrix: Tensor
    :param temp: Temperature
    :type temp: float
    :return: temperature value
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
    # Apply log softmax, we have no mask due to simple fact that there is no possible same instance iteraction
    log_prob = logits - torch.log((exp_logits).sum(dim=1, keepdim=True))
    # Return log_prob
    return log_prob


def compute_loss_contrastive(output_features: Tensor,
                            labels_features: Tensor,
                            features_prototype: Tensor,
                            temp: float) -> Tensor:
    """ Our function uses for the contrastive loss

    :param output_features: Features of our instances
    :type output_features: Tensor
    :param labels_features: Label of our instances
    :type labels_features: Tensor
    :param features_prototype: Features of our prototypes
    :type features_prototype: Tensor
    :param temp: Temperature parameters apply during contrastive learning
    :type temp: float
    :return: Return tensor of loss for each batch
    :rtype: Tensor
    """

    # Compute similairty between prototype and features
    similarity_labels_prototype = output_features @ features_prototype.T
    # Create a mask to keep positive interaction, mask is equivalent to labels
    mask = labels_features
    # Apply our log softmax with temperature
    log_softmax = log_softmax_temp(matrix=similarity_labels_prototype,
                                   temp=temp)
    # Here we have to apply contrastive loss and nomalize by the mask rows
    # It is possible to notice that we note use Epsilon in this case
    # Due to the assumption that each instance has at least one label
    return - (log_softmax * mask).sum(dim=1) / mask.sum(dim=1)


def constrative(output_features: Tensor,
                labels_features: Tensor,
                features_prototype: Tensor,
                temp:float) -> Tensor:
    """_summary_

    :param output_features: Features of the CLS representation shape BxF
    :type output_features: Tensor
    :param labels_features: Label inside the batch of shape BxL
    :type labels_features: Tensor
    :param features_prototype: Features of prototype of shape LxF
    :type features_prototype: Tensor
    :param temp: Temperature for Contrastive Learning
    :type temp: float
    :return: Loss for contrastive Learning
    :rtype: Tensor
    """
    # We apply our contrastive loss function
    loss_contrastive = compute_loss_contrastive(output_features=output_features,
                                                labels_features=labels_features,
                                                features_prototype=features_prototype,
                                                temp=temp)
    # Have to normalize on the batch so we use a mean
    return loss_contrastive.mean()


class LossContrastiveProtoOnly(nn.Module):
    """ Contrastive Loss for Multi-Label

    :param nn: 
    :type nn: _type_
    """
    def __init__(self,
                 alpha: float,
                 beta: float,
                 temp: float=0.07) -> None:
        """ Create our contrastive loss function

        :param alpha: alpha is useless here just declares to have the same trainer function
        :type alpha: float
        :param beta: beta is useless here just declares to have the same trainer function
        :type beta: float
        :param temp: _description_, defaults to 0.07
        :type temp: float, optional
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temp = temp

    def forward(self,
                output_query: Tensor,
                labels_query: Tensor,
                prototype: Tensor) -> Tensor:
        """ This is the forward of our function

        :param output_query: Features of our text is used here
        :type output_query: Tensor
        :param labels_query: labels of our texts
        :type labels_query: Tensor
        :param prototype: Prototype for learning
        :type prototype: Tensor
        :return: Return the loss function
        :rtype: Tensor
        """
        # Normalize to have cosine similarity
        normalize_features_query = F.normalize(output_query, dim=-1, p=2)
        normalize_prototype = F.normalize(prototype, dim=-1, p=2)
        # Apply Contrastive function
        return constrative(output_features=normalize_features_query,
                           labels_features=labels_query,
                           features_prototype=normalize_prototype,
                           temp=self.temp)