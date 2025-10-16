import torch
from typing import Dict,Any
from torch.optim import AdamW
#import nn


def set_optimizer(config: Dict[str, Any], model: torch.nn.Module) -> None:
    """ Set optimizer according to the configuration

    :param config: Configuration file
    :type config: Dict[str, Any]
    :param model: Model to optimize
    :type model: nn.Module
    """
    

    if config.get("optim") is None or config.get("optim") == "AdamW":
        training_parameters_dict = model.parameters_training(lr_backbone=config['lr'],
                                                             lr_projection=config['lr_adding'],
                                                             wd=config['wd'])

        #print_parameters(training_parameters_dict)
        optimizer = AdamW(params=training_parameters_dict)

    elif config["optim"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), config['lr'],
                               momentum=0.9,
                               weight_decay=config['wd'])
    elif config["optim"] == "AdamW1":
       training_parameters_dict = model.parameters_training(lr_backbone=1e-3,
                                                             lr_projection=1e-3,
                                                             wd=1e-2)
       optimizer = AdamW(params=training_parameters_dict)
    elif config["optim"] == "AdamW2":
       optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    
    return optimizer
