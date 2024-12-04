import torch
import math
from torch import nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from transformers import get_scheduler
from torch.utils.data import DataLoader
from typing import Tuple


# We set classical device to cuda
DEVICE = 'cuda'


@torch.no_grad()
def compute_val_loss(model: nn.Module,
                     dataloader_val: DataLoader,
                     loss_function: nn.Module,
                     batch_size_val_test: int) -> float:
    """ Return the loss function on val during contrastive Learning

    :param model: Model which is normaly classification layer
    :type model: nn.Module
    :param dataloader_val: Dataloader, which has to be datalaoder for val
    :type dataloader_val: DataLoader
    :param loss_function: Loss function, which is normaly the loss function for training
    :type loss_function: nn.Module
    :param batch_size_val_test: batch size for val which is normally higher than the training 
    :type batch_size_val_test: int
    :return: float to evaluate the loss
    :rtype: float
    """
    # init the loss function at 0
    loss_score = 0
    # We have to pass the model in eval mode
    model.eval()
    # We have to iterate on all batch in the validation set
    remove = 0
    for _, batch in enumerate(dataloader_val):
        # Contrastive depends a lot of batch, that is why the last is removed if 
       
        # Create current labels of the batch
        current_labels = batch['labels'].to(DEVICE)
        # Output features of the model
        if model.task_type == 'NLP':
             # it doesn't have the great size
            if batch['input_ids'].size(0) < batch_size_val_test:
                remove = 1
                continue
            output_query, _ = model(input_ids=batch['input_ids'].to(DEVICE),
                            attention_mask=batch['attention_mask'].to(DEVICE))
        else : # VISION
            output_query, _ = model(batch['input_ids'].to(DEVICE))
        # Compute the loss, in some loss it is possible that the 
        # prototypes were completly useless
        loss = loss_function(output_query=output_query,
                             labels_query=current_labels,
                             prototype=model.get_prototype())
        # Add loss
        loss_score += loss.item()
    # Model is in training mode after that
    model.train()
    # Custome normalisation of the loss, it is possible that 
    # only len(dataloader_val) is appropriated
    return loss_score/(len(dataloader_val) - remove)


@torch.no_grad()
def create_dataset(model: nn.Module,
                   dataloader: DataLoader,
                   batch_size: int, 
                   training_mode: bool=False,
                   hidden: bool=True) -> DataLoader:
    """ Create Dataloader to use only classical linear evaluation

    :param model: Our Classical Model for training
    :type model: nn.Module
    :param dataloader: Our dataloader with text that we want to transform
    :type dataloader: DataLoader
    :param batch_size: The batch size for our Dataloader
    :type batch_size: 
    :return: _description_
    :rtype: DataLoader
    """
    # Set our model in eval mode
    model.eval()
    # We init features and labels to None
    features = None
    labels = None
    # Iterate on element in dataloader
    for _, batch in enumerate(dataloader):
        # Get the features
        if model.task_type == 'NLP':
            contrastive_outputs, hidden_outputs = model(input_ids=batch['input_ids'].to(DEVICE), attention_mask=batch['attention_mask'].to(DEVICE))
        else : # VISION
            contrastive_outputs, hidden_outputs = model(batch['input_ids'].to(DEVICE))
            
        if hidden:
            output_query = hidden_outputs
        else:
            output_query = contrastive_outputs
        # If features are None we just have to change with the first element of the dataloader
        if features is None:
            features = output_query.detach().cpu()
            labels = batch['labels']
        else:
            # Otherwise we have to stack alog the 0 dim, the new features and labels, with the previous
            features = torch.cat((features, output_query.detach().cpu()), dim=0)
            labels = torch.cat((labels, batch['labels']), dim=0)
    # We have to the set the model in training mode
    model.train()
    # We use simple dataset custom, batch size is set to the given batch size
    # we shuffle, allow pind memory and drop last equal to false
    return DataLoader(DataSetCustom(features=features, labels=labels),
                       batch_size=batch_size,
                       shuffle=training_mode,
                       pin_memory=True,
                       num_workers=4,
                       drop_last=training_mode)


class DataSetCustom(Dataset):
    """:class: Custom Dataset

    :param dataframe: Dataset for training, developement and testing
    :type dataframe: pd.DataFrame
    """
    def __init__(self, features: Tensor, labels: Tensor) -> None:
        super().__init__()
        # Set self.dataframe
        self.features = features
        self.labels = labels

    def __getitem__(self, indice) -> Tuple[Tensor, Tensor]:
        # Return text in first time, and labels after that
        return self.features[indice], self.labels[indice]

    def __len__(self) -> int:
        # Return the length of the dataframe
        return len(self.features)


def create_dataloader_hidden_space(model: nn.Module,
                                   dataloader_train: DataLoader,
                                   dataloader_val: DataLoader,
                                   batch_size: int,
                                   hidden: bool=True,
                                   ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """ Change the daloader with Text to features to do the Linear evaluation

    :param model: Model that we what to use to transform text into features
    :type model: nn.Module
    :param dataloader_train: Dataloader for training
    :type dataloader_train: DataLoader
    :param dataloader_val: Dataloder for Validation
    :type dataloader_val: DataLoader
    :param dataloader_test: Dataloader for Test
    :type dataloader_test: DataLoader
    :param batch_size: batch size for this
    :type batch_size: int
    :return: Return tuple of new Dataloader for training, developement and testing
    :rtype: Tuple[DataLoader, DataLoader, DataLoader]
    """
    # Obtain dataloader for training
    dataloader_train = create_dataset(model=model, dataloader=dataloader_train, batch_size=batch_size, training_mode=True, hidden=hidden)
    # Obtain dataloader for validation
    dataloader_val = create_dataset(model=model, dataloader=dataloader_val, batch_size=batch_size, hidden=hidden)
    # Obtain Dataloder for testing
    return dataloader_train, dataloader_val


def create_dataloader_hidden_space_all(model: nn.Module,
                                        dataloader_train: DataLoader,
                                        dataloader_val: DataLoader,
                                        dataloader_test: DataLoader,
                                        batch_size: int,
                                        hidden: bool=True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """ Change the daloader with Text to features to do the Linear evaluation

    :param model: Model that we what to use to transform text into features
    :type model: nn.Module
    :param dataloader_train: Dataloader for training
    :type dataloader_train: DataLoader
    :param dataloader_val: Dataloder for Validation
    :type dataloader_val: DataLoader
    :param dataloader_test: Dataloader for Test
    :type dataloader_test: DataLoader
    :param batch_size: batch size for this
    :type batch_size: int
    :return: Return tuple of new Dataloader for training, developement and testing
    :rtype: Tuple[DataLoader, DataLoader, DataLoader]
    """
    # Obtain dataloader for training
    dataloader_train = create_dataset(model=model, dataloader=dataloader_train, batch_size=batch_size, training_mode=True, hidden=hidden)
    # Obtain dataloader for validation
    dataloader_val = create_dataset(model=model, dataloader=dataloader_val, batch_size=batch_size, hidden=hidden)
    # Obtain Dataloder for testing
    dataloader_test = create_dataset(model=model, dataloader=dataloader_test, batch_size=batch_size, hidden=hidden)
    # Return our Tuple
    return dataloader_train, dataloader_val, dataloader_test


def create_dataloader_hidden_space_test(model: nn.Module,
                                        dataloader_test: DataLoader,
                                        batch_size: int,
                                        hidden :bool=True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """ Change the daloader with Text to features to do the Linear evaluation

    :param model: Model that we what to use to transform text into features
    :type model: nn.Module
    :param dataloader_train: Dataloader for training
    :type dataloader_train: DataLoader
    :param dataloader_val: Dataloder for Validation
    :type dataloader_val: DataLoader
    :param dataloader_test: Dataloader for Test
    :type dataloader_test: DataLoader
    :param batch_size: batch size for this
    :type batch_size: int
    :return: Return tuple of new Dataloader for training, developement and testing
    :rtype: Tuple[DataLoader, DataLoader, DataLoader]
    """
    dataloader = create_dataset(model=model, dataloader=dataloader_test, batch_size=batch_size, hidden=hidden)
    # Obtain Dataloder for testing
    return dataloader


def traine_linear_classifier(linear_classifier: nn.Module,
                             dataloader: DataLoader,
                             optim) -> None:
    """ Train our Linear Layer, based on data which are in the dataloader

    :param linear_classifier: Module that we want to 
    :type linear_classifier: nn.Module
    :param dataloader: Dataloader to iterate on it
    :type dataloader: DataLoader
    :param batch_size: dimension of the batch size
    :type batch_size: int
    :param optim: Optimzie
    :type optim: _type_
    """
    # Create scheduler which is linear, we don't need warmup state
    lr_scheduler = get_scheduler("linear",
                                optimizer=optim,
                                num_warmup_steps=0,
                                num_training_steps=len(dataloader) * 40)
    # by default the number of epochs is blocked to 40
    for step in range(40):
        # Iterate of all element inside the batch
        loss_save = 0
        for _, batch in enumerate(dataloader):
            # Set Optim to zeros grad
            optim.zero_grad(set_to_none=True)
            # Compute loss function
            loss = linear_classifier(x=batch[0].to(DEVICE), y=batch[1].to(DEVICE))
            loss_save += loss.item()     
            # Backward
            loss.backward()
            # Step of optimizer
            optim.step()
            # scheduler step
            lr_scheduler.step()
        loss_save = 0