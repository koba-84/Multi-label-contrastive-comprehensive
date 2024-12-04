from xmlrpc.client import boolean
import torch
from typing import Dict, List
from torch.utils.data import DataLoader
import numpy as np
import os
import torch.nn as nn
import random
from sklearn.metrics import f1_score
from torch import Tensor
from sklearn.metrics import f1_score, hamming_loss
from typing import List


def set_seed(seed: int):
    """Set the seed for reproductibility
    :param seed: _description_
    :type seed: int
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def get_all_preds(model: torch.nn.Module,
                  dataloader: DataLoader,
                  device: str):
    """_summary_
    Args:
        model (torch.nn.Module): _description_
        dataloader (DataLoader): _description_
        device (str): _description_
        classification (bool, optional): _description_. Defaults to True.
    Returns:
        _type_: _description_
    """
    all_preds = torch.tensor([])
    all_labels = torch.tensor([])
    for batch in dataloader:
        preds = model(x=batch[0].to(device))
        all_preds = torch.cat((all_preds, preds.float().cpu()))
        all_labels = torch.cat((all_labels, batch[1].cpu()))
    return all_preds, all_labels


def save_best_model(model: nn.Module, config: Dict[str, int], score: float):
    """Save best model in "save"

    Args:
        model (nn.Module): model to save
        config (Dict[str, int]): dict which describes the training
        current_score (float): float best score
    """
    path_save = os.path.join('save', config["name_save"])
    path_result = os.path.join('save', config["name_save"], 'results.txt')
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    if not os.path.exists(path_result):
        os.mknod(path_result)
    torch.save(model.state_dict(), os.path.join(path_save, 'model.pt'))
    with open(path_result, 'w') as f:
        f.write(str(score) + '\n')
        for key, value in config.items():
            f.write(key + ' ' + str(value) + '\n')


def save_best_model_train(model: nn.Module, config: Dict[str, int]):
    """Save best model in "save"

    Args:
        model (nn.Module): model to save
        config (Dict[str, int]): dict which describes the training
        current_score (float): float best score
    """
    path_save = os.path.join('save', config["name_save"])
    torch.save(model.state_dict(), os.path.join(path_save, 'model_train.pt'))
    

def save_best_model_final(model: nn.Module, config: Dict[str, int]):
    """Save best model in "save"

    Args:
        model (nn.Module): model to save
        config (Dict[str, int]): dict which describes the training
        current_score (float): float best score
    """
    path_save = os.path.join('save', config["name_save"])
    #if the folder does not exist, create it :
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    torch.save(model.state_dict(), os.path.join(path_save, 'model_final.pt'))


def save_test_score(config: Dict[str, int], score: Dict[str, float]):
    """Save test scores

    :param config: basic model's config
    :type config: Dict[str, int]
    :param score: config score
    :type score: Dict[str, float]
    """
    path_save = os.path.join('save', config["name_save"])
    path_result = os.path.join(path_save, 'results.txt')
    path_model = os.path.join(path_save, 'model.pt')
    path_model_train = os.path.join(path_save, 'model_train.pt')
    path_model_final = os.path.join(path_save, 'model_final.pt')
    with open(path_result, 'a') as f:
        for key, value in score.items():
            f.write(key + ' ' + str(value) + '\n')
    os.remove(path=path_model)
    os.remove(path=path_model_train)
    os.remove(path=path_model_final)


def compute_test_metrics(y_true: np.array, y_pred: np.array, add_str: str, nb_class: int):
    """ Return dict with contains hamming loss, micro/macro F1

    :param y_true: True labels
    :type y_true: np.array
    :param y_pred: Model predictions
    :type y_pred: np.array
    :return: config
    :rtype: Dict[str, float]
    """
    config = {}
    config["hamming_loss" + ' ' + add_str] = hamming_loss(y_true, y_pred.round())
    config["f1 macro"+ ' ' + add_str] = f1_score(y_true, y_pred.round(), average='macro')
    config["f1 micro"+ ' ' + add_str] = f1_score(y_true, y_pred.round(), average='micro')
    return config


def compute_test_metrics_individual(y_true: np.array, y_pred: np.array):
    score_label = []
    for pred_value, true_value in zip(y_pred.T, y_true.T):
        score_label.append(f1_score(true_value, pred_value.round(), average='binary'))
    return score_label


def save_test_score_2(config: Dict[str, int], score: Dict[str, float]):
    """Save test scores

    :param config: basic model's config
    :type config: Dict[str, int]
    :param score: config score
    :type score: Dict[str, float]
    """
    path_save = os.path.join('save', config["name_save"])
    path_result = os.path.join(path_save, 'results.txt')
    path_model = os.path.join(path_save, 'model.pt')
    with open(path_result, 'a') as f:
        for key, value in score.items():
            f.write(key + ' ' + str(value) + '\n')
    os.remove(path=path_model)
    