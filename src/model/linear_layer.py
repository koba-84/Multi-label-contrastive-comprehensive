import torch
from torch import nn
from torch import Tensor
from typing import Optional, Iterator, List, Dict


class LinearEvaluation(nn.Module):
    """ Classical Linear Layer, for Multi-label

    :param nn: _description_
    :type nn: _type_
    """
    def __init__(self,
                 nb_labels: int,
                 hidden_size: int) -> None:
        """ Constructor of our Module

        :param nb_labels: Number of labels for the projection head
        :type nb_labels: int
        :param hidden_size: Dimension of the hidden size for the projection head
        :type hidden_size: int
        """
        super().__init__()
        # Define our classifier in Simple manner
        self.classifier = nn.Linear(hidden_size, nb_labels)
        # Weights are initialized with xavier uniform
        nn.init.xavier_uniform_(self.classifier.weight)
        # Bias are set to 0 at begining of the training
        nn.init.zeros_(self.classifier.bias)
        # Define activation function for prediction
        self.activation = nn.Sigmoid()
        # Define loss function for BCE
        self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self, x: Tensor, y: Optional[Tensor]=None) -> Tensor:
        """ Classical forward function

        :param x: Input of the model
        :type x: Tensor
        :param y: Labels, defaults to None
        :type y: Optional[Tensor], optional
        :return: Loss if the labels are given at inputs 0 otherwith
        :rtype: Tensor
        """
        # If y is not None we have to return the loss function
        if y is not None:
            return self.loss(self.classifier(x), y)
        else:
        # Otherwise we have to return probabiltites
            return self.activation(self.classifier(x))

    def return_parameters_no_decay(self) -> Iterator[Tensor]:
        """Return Parameter with no decay in this case it is basically only bias

        :yield: Return Tensor of parameter with no decay : bias
        :rtype: Iterator[Tensor]
        """
        for name, parameters in self.named_parameters():
            if 'bias' in name or 'LayerNorm' in name:
                yield parameters

    def return_parameters_decay(self) -> Iterator[Tensor]:
        """ Return Parmaeter which needs decay, normally weights of Linear

        :yield: Iterator on Linear weight
        :rtype: Iterator[Tensor]
        """
        for name, parameters in self.named_parameters():
            if 'bias' not in name and 'LayerNorm' not in name:
                yield parameters

    def parameters_training(self, lr, wd: float) -> List[Dict[str, Tensor]]:
        """ Return Parameter for the trainings

        :param lr: Learning rate
        :type lr: float
        :param wd: weight decay
        :type wd: float
        :return: Something
        :rtype: _type_
        """
        parameters = [{'params': self.return_parameters_no_decay(),
                       'lr': lr,
                       'weight_decay': 0},
                      {'params': self.return_parameters_decay(),
                       'lr': lr,
                       'weight_decay': wd}
                      ]
        return parameters