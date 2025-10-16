import torch
from typing import Dict, Iterator, List
from torch import nn, Tensor
from transformers import AutoModel
from torch.nn.init import xavier_uniform_
from typing import Literal

from backbones.vision_backbones import get_vision_backbone

class Baseline(nn.Module):
    """ Baseline Contrastive Learning Model
    :class: nn.Module
    """
    def __init__(self,
                 backbone_path: str,
                 nb_labels: int,
                 projection_dim: int=128,
                 task_type: Literal['NLP', 'VISION']='NLP',
                 weights=None) -> None:
        """ Create basic Contrastive Model

        :param path: The path use to download a pre-trained model, roberta base in our experiments
        :type path: str
        :param nb_labels: Number of labels, uses to define prototype shape
        :type nb_labels: int
        :param projection_dim: Projection dimension for contrastive settings, defaults to 128
        :type projection_dim: int, optional
        """
        super().__init__()
        self.task_type = task_type
        if task_type == 'NLP':
            self.backbone = AutoModel.from_pretrained(backbone_path, add_pooling_layer=False)
            # save config
            self.config = self.backbone.config 
            # Catch hidden dimension
            self.hidden_size = self.backbone.config.hidden_size
        elif task_type == 'VISION':
            self.backbone,self.hidden_size = get_vision_backbone(model_name=backbone_path, pretrained=True,weights=weights)
        # Define activation function which is basically sigmoid function
        self.activation = nn.Sigmoid()
        # We define classical projection head composed of two linear layers with relu activation
        # We project the initial space into  projection_dim
        self.projection = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.hidden_size, projection_dim))
        # We create prototype here, with basic shape
        self.prototype = nn.Parameter(torch.rand(nb_labels, projection_dim))
        # Initialisation with Xavier uniform
        xavier_uniform_(self.prototype)
        # Initialisation with Xavier uniform
        self.projection.apply(self._init_weights_module)
        # add bce
        self.linear_bce = nn.Linear(self.hidden_size, nb_labels)
        self.linear_bce.apply(self._init_weights_module)
        nn.init.constant_(self.linear_bce.bias, 0)
        self.dropout = nn.Dropout(0.1)

    def _init_weights_module(self, module: nn.Module) -> None:
        """ Init weights correctly with xavier uniform

        :param module: Module in our model
        :type module: nn.Module
        """
        if isinstance(module, nn.Linear):
            xavier_uniform_(module.weight)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor=None) -> Tensor:
        """Basic forward of a model

        :param input_ids: Token indice
        :type input_ids: torch.Tensor
        :param attention_mask: Mask to take into account the padding
        :type attention_mask: torch.Tensor
        :rtype: Tensor
        """
        
        if self.task_type == 'NLP':
            assert attention_mask is not None, "Attention mask is mandatory for NLP task"
            # If we have key_encoder or linear evaluation, grad should be removed
            text_embeddings = self.backbone(input_ids=input_ids, attention_mask=attention_mask)[0]
            # Keep only the token CLS as global Representation
            final_representation = text_embeddings[:, 0, :]
            return self.linear_bce(self.dropout(final_representation)), final_representation
        elif self.task_type == 'VISION':
            #No need for attention mask, still return the projection and the final representation
            image_embeddings = self.backbone(input_ids)
            final_representation = image_embeddings
            return self.linear_bce(self.dropout(final_representation))
        
    def get_prototype(self) -> nn.Parameter:
        """ Equivalent to a getter to obtain prototype

        :return: Return our Prototypes
        :rtype: nn.Parameter
        """
        return self.prototype

    def return_parameters_no_decay(self, group: nn.Module) -> Iterator[Tensor]:
        """ Return parameters with no decay, we have to return bias or Layer norm parameters

        :yield: Iterate on parameters with no-decay
        :rtype: None
        """
        for name, parameters in group.named_parameters():
            if 'bias' in name or 'LayerNorm' in name:
                yield parameters

    def return_parameters_decay(self, group: nn.Module) -> Iterator[Tensor]:
        """ Return parameters with decay, no bias and no LayerNorm parameters

        :yield: Iterate on parameters with decay
        :rtype: torch.tensor
        """
        for name, parameters in group.named_parameters():
            if 'bias' not in name and 'LayerNorm' not in name:
                yield parameters

    def parameters_training(self, lr_backbone: float, lr_projection: float, wd: float) -> List[Dict[str, Tensor]]:
        """ Return Parameter for training, it is possible to have different lr for backbone and adding part
        :param lr_backbone: Learning rate for the backbone
        :type lr_backbone: float
        :param lr_projection: Learning Rate for the projection head
        :type lr_projection: float
        :param wd: weight decay parameter
        :type wd: float
        :return:  return parameter for a classical optimizer
        :rtype: List[Dict[str, Tensor]]
        """
        parameters = [{'params': self.return_parameters_decay(group=self.backbone),
                    'lr': lr_backbone,
                    'weight_decay': wd},
                    {'params': self.return_parameters_no_decay(group=self.backbone),
                    'lr': lr_backbone,
                    'weight_decay': 0},
                    {'params': self.return_parameters_decay(group=self.projection),
                    'lr': lr_projection,
                    'weight_decay': wd},
                    {'params': self.return_parameters_no_decay(group=self.projection),
                    'lr': lr_projection,
                    'weight_decay': 0},
                    {'params': self.prototype,
                    'lr': lr_projection,
                    'weight_decay': wd},
                    {'params': self.return_parameters_decay(group=self.linear_bce),
                    'lr': lr_projection,
                    'weight_decay': wd},
                    {'params': self.return_parameters_no_decay(group=self.linear_bce),
                    'lr': lr_projection,
                    'weight_decay': 0},
                    ]
        return parameters
