import torch
from torch import nn
from torch import Tensor


class Zlpr(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """ Forward of the ZLPR loss function

        :param x: logits of shape B x L
        :type x: Tensor
        :param y: B x L
        :type y: Tensor
        :return: Loss value of the ZLPR
        :rtype: Tensor
        """
        positive_logits = ((-x) * y).clone()
        positive_logits[positive_logits == 0] = float('-inf')
        negative_logits = (x * (1-y)).clone()
        negative_logits[negative_logits == 0] = float('-inf')
        max_positif, _ = torch.max(positive_logits, dim=1, keepdim=True)
        max_negatif, _ = torch.max(negative_logits, dim=1, keepdim=True)
        max_positif, max_negatif = max_positif.detach(), max_negatif.detach()
        positif = ((-x - max_positif).exp() * y).sum(dim=1, keepdim=True)
        negatif = ((x - max_negatif).exp() * (1-y)).sum(dim=1, keepdim=True)
        loss = torch.log(positif + ( - max_positif).exp()) + torch.log(negatif + ( - max_negatif).exp())
        return loss.mean()
    


class AsymmetricLoss(nn.Module):
    def __init__(self,
                 gamma_neg: float=1,
                 gamma_pos: float=0,
                 clip: float=0,
                 eps: float=1e-5):
        super().__init__()
        # Set all parameters
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping self.clip is equivalent to m in their paper
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss *= one_sided_w
        return - loss.mean()