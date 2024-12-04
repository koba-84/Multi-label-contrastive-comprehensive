import torch
from torch import nn
from torch import Tensor
from typing import Optional, Iterator, List, Dict


class LinearEvaluationMultiple(nn.Module):
    """ Create Linear Evaluation with Multiple different inititialisation

    :param nn: _description_
    :type nn: _type_
    """
    def __init__(self,
                 nb_labels: int,
                 hidden_size: int,
                 nb_multiple: int=5) -> None:
        """_summary_

        :param nb_labels: Number of Label for the Linear Layer
        :type nb_labels: int
        :param hidden_size: The dimension of the hidden size
        :type hidden_size: int
        :param nb_multiple: number of individual projection head, defaults to 5
        :type nb_multiple: int, optional
        """
        super().__init__()
        # Keep constant of different number of linear layer in memory
        self.nb_multiple = nb_multiple
        # Create Module List, which will contain all individual Linear Layer
        self.classier_multiple = nn.ModuleList([nn.Linear(hidden_size, nb_labels) for _ in range(nb_multiple)])
        # Create activation function for prediction
        self.activation = nn.Sigmoid()
        # Create Loss function for simple training inside the model
        self.loss = nn.BCEWithLogitsLoss()
        # Init in a same way all Linear Layer which are in the ListModule
        self.init_weight_multiple()
    
    def init_weight_multiple(self) -> None:
        """Function to init weight of all linear layer in self.classifier Multiple
        """
        for linear_layer in self.classier_multiple:
            nn.init.xavier_uniform_(linear_layer.weight)
            nn.init.zeros_(linear_layer.bias)
    
    def forward(self, x: Tensor, y: Optional[Tensor]=None) -> Tensor:
        # if y is not None, we want to return Loss function
        # This loss function has to be the sum of all loss function
        if y is not None:
            loss = None
            # Init loss at None
            for linear_layer in self.classier_multiple:
                if loss is None:
                    # if Loss is None we just have to init with the first
                    loss = self.loss(linear_layer(x), y)
                else:
                    # Sum all loss
                    loss += self.loss(linear_layer(x), y)
            return loss
        else:
            # In the other case if y is None we want to output the mean of proba
            proba = None
            for linear_layer in self.classier_multiple:
                if proba is None:
                    proba = self.activation(linear_layer(x))
                else:
                    proba += self.activation(linear_layer(x))
            return proba/self.nb_multiple

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