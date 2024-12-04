import torch
from torch import nn
from torch import Tensor
# Epsilon to avoid divided by 0
DEVICE = 'cuda'
EPS = 1e-8
import torch.nn.functional as F
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def remove_ponderate(x: Tensor, alpha: float) -> Tensor:
    """ Apply the function f = (1 -x)^alpha, if alpha is negative, we do hard threshold

    :param x: |y_j/y_j inter y_i|
    :type x: Tensor
    :param alpha: controle the strength of the regulation
    :type alpha: float
    :return: Tensor
    :rtype: _type_
    """
    if alpha < 0:
        # particular case hard threshold
        x_copy = x.clone()
        x_copy[x <= 1 - EPS] = 0
        x_copy[x >= 1 - EPS] = 1
        return x_copy
    else:
        return (x)**alpha

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
    # Compute the sigmas score for remove gradient
    sigma = exp_logits/(exp_logits * mask).sum(dim=1, keepdim=True)
    # Return log_prob, and sigmas and detach detach
    return log_prob, sigma.detach()


def compute_loss_contrastive(output_features: Tensor,
                            labels_query: Tensor,
                            prototype_features: Tensor,
                            alpha: float,
                            temp:float) -> Tensor:
    # Number of element inside the batch in our case its query + proto
    total = labels_query.size(0) + labels_query.size(1)
    # Save Number of Labels
    nb_labels = prototype_features.size(0)
    # Mask to remove interaction between same instance, diagonal mask here
    mask_diagonal = torch.ones(total*total, device='cuda', dtype=torch.uint8)
    mask_diagonal[0::(total) +1 ] = 0
    mask_diagonal = mask_diagonal.reshape(total, total)
    # Concatenate all features: shape (B + L) x d
    all_features = torch.cat((output_features, prototype_features), dim=0)
    # Concatenate all labels: shape (B + L) x d
    all_labels = torch.cat((labels_query, torch.eye(nb_labels, device=labels_query.device)), dim=0)
    # Compute all similarity: shape (B + L)(B + L)
    similarity_query_query = all_features @ all_features.T
    # Compute similarity score and remove diag for simplicity in the following
    re_normalize = remove_ponderate((all_labels @ all_labels.T)/all_labels.sum(dim=1).unsqueeze(0), alpha) * mask_diagonal
    logging.debug(f"Les valeurs doivent être entre 0 et 1, c'est le score obtenu par la fonction f{re_normalize}")
    # Compute a mask and, we remove interaction for instance which does not have interaction: shape (B+L)(B+L)(L)
    mask_and_features = torch.einsum('ac, bc -> abc', all_labels, all_labels)
    # The normalisation term does not depends of the label normalisation, sum on the first dimension for labels normalisation shape (B + L) x L
    normalize = (re_normalize.unsqueeze(2) * mask_and_features).sum(dim=1)
    # Apply the label normalisation, weights w without label normalisation: shape (B + L)L
    w = ((re_normalize.unsqueeze(2) * mask_and_features)/(normalize.unsqueeze(1)+ EPS)).sum(dim=2)
    logging.debug(f"On doit avoir un vecteur du nombre de label de chaque anchor {w.sum(dim=1)}")
    # Normalize by the number of labels of each anchors
    w = w / (all_labels.sum(dim=1, keepdim=True))
    logging.debug(f"On calculer la normalisation doit être nun vecteur de 1 {w.sum(dim=1)}")
    # Normalize denominator
    # compute temp softmax add the mask for the denominator which is only the diagonal mask
    log_softmax, sigma = log_softmax_temp(matrix=similarity_query_query,
                                   temp=temp,
                                   mask=mask_diagonal)
    # Return the negative log softmax, ponderate
    
    # Add grad stabilize
    mask_remove_grad = (w != 0).float()
    # We compute the max part mutiply by remove grad mask
    weight_reg = torch.max(-w + sigma, torch.zeros_like(w, device=w.device)) * mask_remove_grad
    logging.debug(f'check reg {weight_reg}, {mask_remove_grad}')
    # Multiply the normalisation term by temperate cosine similairty
    reg_term = weight_reg * similarity_query_query/temp
    # it is possible that some instances to not have positive pair (only for prototype) in this case we normalize by the number
    # of anchor how have at least one interaction
    return - (log_softmax * w + reg_term).sum(dim=1)/(w.sum(dim=1) !=0).int().sum()


def constrative(output_features: Tensor,
                labels_query: Tensor,
                prototype_features: Tensor,
                alpha: float,
                temp: float) -> Tensor:
    """_summary_

    :param output_features: Features of the CLS representation shape BxF
    :type output_features: Tensor
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
                                                temp=temp)
    # It is already normalize we do not have to use a mean but a sum
    return loss_contrastive.sum()


class LossContrastiveMSCRG(nn.Module):
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
                           temp=self.temp)
