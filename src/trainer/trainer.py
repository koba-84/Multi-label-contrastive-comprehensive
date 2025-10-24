import torch
import wandb
import os
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from torch.optim import AdamW
from utils.utils import set_seed, get_all_preds, save_best_model, save_test_score, compute_test_metrics, save_best_model_train, save_best_model_final, compute_test_metrics_individual
from utils.ablation import evaluate_embedding
from trainer.utils_trainer_vision import set_optimizer
from copy import deepcopy
from data import dataloader
from model.baseline_ours_bce import Baseline as Baseline_2
from model.baseline_ours import Baseline
from model.linear_layer import LinearEvaluation
from model.linear_layer_multiple import LinearEvaluationMultiple
from torch.amp import GradScaler
from torch import autocast
from data.read_dataset import read_dataset
from data.vision_datasets import get_vision_loaders
from typing import Dict, List
from .loss.loss_contrastive_proto_only import LossContrastiveProtoOnly
from .loss.loss_contrastive_msc import LossContrastiveMSC
from .loss.loss_contrastive_base import LossContrastiveBase
from .loss.loss_contrastive_mscrg import LossContrastiveMSCRG
from .loss.loss_contrastive_mscwrg import LossContrastiveMSCWRG
from .loss.loss_contrastive_mulsupcon import LossContrastiveMulSupCon
from .loss.loss_contrastive_nws import LossContrastiveNWS, compute_label_pair_similarity
from .loss.zlpr import Zlpr, AsymmetricLoss
from .basic_utils import (compute_val_loss, create_dataloader_hidden_space,
                          traine_linear_classifier, traine_linear_classifier_end_to_end,
                          create_dataloader_hidden_space_test)
import transformers
from itertools import product
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime
current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

BATCH_EVAL_TEST = 256
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
WARMUP = 0.05
MOMENTUM = 0.999

ALL_LR = [1e-1, 1e-2] 
ALL_WD = [1e-1, 1e-2, 1e-4]

# Vision datasets
VISION_DATASETS = ['voc2007', 'coco2014','coco2','flickr25k','nuswide']


def trainer(config: Dict, entity_name: str):
    """Train our best Neural Networks

    :param config: config for training
    :type config: Dict
    :param entity_name: Name entity for wandb
    :type entity_name: str
    """
    if "freeze_encoder" not in config:
        raise ValueError("freeze_encoder must be specified as true/false in config")

    TASK_TYPE = "NLP" if config.get(
        "dataset_name") not in VISION_DATASETS else "VISION"
    config["task_type"] = TASK_TYPE
    
    # define the collate args to "simple" if not defined
    if config.get("collate") is None:
        config["collate"] = "simple"    

    if config.get("project") is None:
        config['project'] = f"mulsupconv2testtest{config['dataset_name']}" if TASK_TYPE == "NLP" else f"mulsupconv2"

    if config.get("weights") is None:
        config["weights"] = "imagenet"
        
    

    # Set seed for reproductibility
    set_seed(config["SEED"])
    # Define Name for wdb
    print(config)
    
    config["merge_groupe"] = '_'.join([str(config['loss']),
                                       str(config["dataset_name"]),
                                       str(config["epochs"]),
                                       str(config["batch_size"]),
                                       str(config["lr"]),
                                       str(config["alpha"]),
                                       str(config["beta"]),
                                       str(config['temp'])])
    # if optim is given in config add it to the config merge groupe
    if config.get("optim") is not None:
        config["merge_groupe"] = config["merge_groupe"] + \
            "_" + config["optim"]

    print("========= Starting set Wandb ===========")
    config["name_save"] = config["merge_groupe"] + str(config["SEED"])
    # Name run as seed
    config["name_run"] = str(config['SEED'])

    if config["task_type"] == "NLP":
        # Set tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config['name_model'])
        # Create dataloader for training validation and testing
        dataloader_train, dataloader_val, dataloader_test = dataloader.dataloader(batch_size=config["batch_size"],
                                                                                  max_len=config["max_len"],
                                                                                  tokenizer=tokenizer,
                                                                                  dataset_name=config["dataset_name"],
                                                                                  batch_size_val_test=BATCH_EVAL_TEST,
                                                                                  training_mode=True)
        train_data, _, _ = read_dataset(name=config['dataset_name'])
        # textカラムとnan_maskカラムを除外した列数がラベル数
        config['nb_labels'] = train_data.shape[1] - 2
        if config["loss"] == "bce":
            # print("=== BCE Training === ")
            # train_BCE(config, dataloader_train, dataloader_val,
            #           dataloader_test, entity_name, num_labels=config['nb_labels'], loss='bce',bce_loss='bce')
            # print("=== ZLPR Training === ")
            # train_BCE(config, dataloader_train, dataloader_val,
            #           dataloader_test, entity_name, num_labels=config['nb_labels'], loss='ZLPR',bce_loss='ZLPR')
            print("=== Asymetric Training === ")
            train_BCE(config, dataloader_train, dataloader_val,
                      dataloader_test, entity_name, num_labels=config['nb_labels'], loss='asymetric',bce_loss='asymetric')
            print("=== Focal Training === ")
            train_BCE(config, dataloader_train, dataloader_val,
                      dataloader_test, entity_name, num_labels=config['nb_labels'], loss='focal',bce_loss='focal')
            return 0
        

        print("=== download data donnneeee! ====")
        # Create Model
        model = Baseline(backbone_path=config['name_model'],
                         nb_labels=config['nb_labels'],
                         projection_dim=config["projection_dim"],
                         task_type="NLP")
        model.to(DEVICE)
        # Reproductibility
        set_seed(config["SEED"])
    else:  # Vision

        dataloader_train, dataloader_val, dataloader_test, config['nb_labels'] = get_vision_loaders(dataset=config["dataset_name"],
                                                                                                    batch_size=config["batch_size"],
                                                                                                    img_size=config["img_size"],
                                                                                                    augmentation= config["augmentation"],
                                                                                                    seed=config["SEED"],
                                                                                                    collate=config["collate"],
                                                                                                    fraction=config["fraction"])
        if config["loss"] == "bce":
            # print("=== BCE Training === ")
            # train_BCE(config, dataloader_train, dataloader_val,
            #           dataloader_test, entity_name, num_labels=config['nb_labels'], loss='bce',bce_loss='bce')
            # print("=== ZLPR Training === ")
            # train_BCE(config, dataloader_train, dataloader_val,
            #           dataloader_test, entity_name, num_labels=config['nb_labels'], loss='ZLPR',bce_loss='ZLPR')
            print("=== Asymetric Training === ")
            train_BCE(config, dataloader_train, dataloader_val,
                      dataloader_test, entity_name, num_labels=config['nb_labels'], loss='asymetric',bce_loss='asymetric')
            print("=== Focal Training === ")
            train_BCE(config, dataloader_train, dataloader_val,
                      dataloader_test, entity_name, num_labels=config['nb_labels'], loss='focal',bce_loss='focal')
            return 0

        model = Baseline(backbone_path=config['name_model'],
                         nb_labels=config['nb_labels'],
                         projection_dim=config["projection_dim"],
                         task_type="VISION",
                         weights=config["weights"])

        model.to(DEVICE)



    queue_size = int(config.get("queue_size", 512))
    config["queue_size"] = queue_size

    key_encoder = deepcopy(model).to(DEVICE)
    for param in key_encoder.parameters():
        param.requires_grad = False

    prototype_dtype = model.get_prototype().dtype
    queue_feats = torch.zeros(queue_size,
                              config["projection_dim"],
                              device=DEVICE,
                              dtype=prototype_dtype)
    queue_labels = torch.zeros(queue_size,
                               config["nb_labels"],
                               device=DEVICE,
                               dtype=prototype_dtype)
    queue_ptr = 0
    queue_filled = 0

    # Create optimizer for the query model
 
    optim = set_optimizer(config, model, freeze_encoder=config["freeze_encoder"])

    # Create scheduler with warm-up and cosin decay
    lr_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=optim,
                                                                num_warmup_steps=int(
                                                                    len(dataloader_train) * config["epochs"] * WARMUP),
                                                                num_training_steps=len(dataloader_train) * config["epochs"])
    scaler = GradScaler()
    # Define loss function for training and testing
    #### Define our loss ####
    
    # NWS用のラベル類似度行列を事前計算
    sim_matrix_nws = None
    if config["loss"] == "nws":
        print("=== Computing label similarity matrix for NWS ===")
        all_labels = []
        for batch in tqdm(dataloader_train, desc="Collecting labels"):
            all_labels.append(batch['labels'])
        Y_train = torch.cat(all_labels, dim=0)  # (N, L)
        
        # 類似度行列を計算
        sim_matrix_nws = compute_label_pair_similarity(
            Y=Y_train,
            method=config["sim_method"]  # 'npmi' or 'jaccard'
        )
        print(f"Label similarity matrix computed: shape={sim_matrix_nws.shape}")
    
    if config["loss"] == "proto": 
        loss_contrastive = LossContrastiveProtoOnly(
            alpha=0, beta=0, temp=config['temp'])
        loss_contrastive_val = LossContrastiveProtoOnly(
            alpha=0, beta=0, temp=config['temp'])
    elif config["loss"] == "msc":  # (Comme dans le papier L_abalone)
        loss_contrastive = LossContrastiveMSC(
            alpha=1, beta=config["beta"], temp=config['temp'])
        loss_contrastive_val = LossContrastiveMSC(
            alpha=1, beta=config["beta"], temp=config['temp'])
    elif config["loss"] == "base":
        loss_contrastive = LossContrastiveBase(
            alpha=1, beta=1, temp=config['temp'])
        loss_contrastive_val = LossContrastiveBase(
            alpha=1, beta=1, temp=config['temp'])
    elif config["loss"] == 'mscrg':
        loss_contrastive = LossContrastiveMSCRG(
            alpha=config["alpha"], beta=1, temp=config['temp'])
        loss_contrastive_val = LossContrastiveMSCRG(
            alpha=config["alpha"], beta=1, temp=config['temp'])
    elif config["loss"] == 'mscwrg':
        loss_contrastive = LossContrastiveMSCWRG(
            alpha=config["alpha"], beta=1, temp=config['temp'])
        loss_contrastive_val = LossContrastiveMSCWRG(
            alpha=config["alpha"], beta=1, temp=config['temp'])
    elif config["loss"] =='mulsupcon':
        loss_contrastive = LossContrastiveMulSupCon(
            alpha=1, beta=1, temp=config['temp'])
        loss_contrastive_val = LossContrastiveMulSupCon(
            alpha=1, beta=1, temp=config['temp'])
    elif config["loss"] == "nws":
        loss_contrastive = LossContrastiveNWS(
            alpha=1, beta=config["beta"], temp=config['temp'],
            agg=config["agg"], sim=sim_matrix_nws)
        loss_contrastive_val = LossContrastiveNWS(
            alpha=1, beta=config["beta"], temp=config['temp'],
            agg=config["agg"], sim=sim_matrix_nws)
    else:
        raise ValueError(f"Loss {config['loss']} not implemented")
    ##########################################
    print("=== CL Training === ")

    wandb.init(project=config["project"] + "-cl",
               group=config["merge_groupe"],
               entity=entity_name,
               config=config)

    wandb.define_metric("cl_epoch")
    wandb.define_metric("cl_test_step")
    wandb.define_metric("cl_loss_train", step_metric="cl_epoch")
    wandb.define_metric("cl_loss_val", step_metric="cl_epoch")
    wandb.define_metric("f1 micro test", step_metric="cl_test_step")
    wandb.define_metric("f1 macro test", step_metric="cl_test_step")
    wandb.define_metric("hamming_loss test", step_metric="cl_test_step")
    wandb.define_metric("silhouette_test", step_metric="cl_test_step")
    wandb.define_metric("dbi_test", step_metric="cl_test_step")
    wandb.define_metric("classifier/epoch")
    wandb.define_metric("classifier/val_f1_micro", step_metric="classifier/epoch")
    wandb.define_metric("classifier/val_f1_macro", step_metric="classifier/epoch")
    wandb.define_metric("classifier/val_hamming_loss", step_metric="classifier/epoch")

    # Set name of the run in Wandb
    wandb.run.name = config["name_run"]

    total_loss = 0
    best_loss_save = 100
    best_loss_save_train = 100
    for step in tqdm(range(config["epochs"]), desc="Epochs"):
        for _, batch in enumerate(tqdm(dataloader_train, desc="Batches")):

            # Set gradient at 0 grad
            optim.zero_grad(set_to_none=True)
            current_labels = batch['labels'].to(DEVICE).to(queue_labels.dtype)
            # Cast into precision mixte
            with autocast(device_type='cuda', dtype=torch.float16):
                if TASK_TYPE == "NLP":
                    output_query, _ = model(input_ids=batch['input_ids'].to(DEVICE),
                                            attention_mask=batch['attention_mask'].to(DEVICE))
                else:
                    output_query, _ = model(input_ids=batch['input_ids'].to(DEVICE))

            with torch.no_grad():
                if TASK_TYPE == "NLP":
                    output_key, _ = key_encoder(input_ids=batch['input_ids'].to(DEVICE),
                                                attention_mask=batch['attention_mask'].to(DEVICE))
                else:
                    output_key, _ = key_encoder(input_ids=batch['input_ids'].to(DEVICE))

            output_query = output_query.float()
            output_key = output_key.float()
            normalize_query = F.normalize(output_query, dim=-1)
            normalize_key = F.normalize(output_key, dim=-1)

            if queue_filled > 0:
                queue_feats_for_loss = queue_feats[:queue_filled]
                queue_labels_for_loss = queue_labels[:queue_filled]
            else:
                queue_feats_for_loss = None
                queue_labels_for_loss = None

            loss_kwargs = dict(output_query=normalize_query,
                               labels_query=current_labels,
                               prototype=model.get_prototype())

            if config["loss"] in {"mulsupcon", "msc", "nws"}:
                loss_kwargs.update(queue_feats=queue_feats_for_loss,
                                   queue_labels=queue_labels_for_loss,
                                   key_feats=normalize_key,
                                   key_labels=current_labels)

            loss = loss_contrastive(**loss_kwargs)
            # Backward
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            # Gradient clipping to avoid gradiant explosion
            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0, norm_type=2)
            # Optimiser step
            scaler.step(optim)
            scaler.update()
            # scheduler step
            lr_scheduler.step()
            with torch.no_grad():
                for param_q, param_k in zip(model.parameters(),
                                            key_encoder.parameters()):
                    param_k.data.mul_(MOMENTUM).add_(
                        param_q.data * (1.0 - MOMENTUM))

                batch_size_curr = normalize_key.shape[0]
                if batch_size_curr == 0:
                    pass
                else:
                    if batch_size_curr > queue_size:
                        normalize_key = normalize_key[:queue_size]
                        labels_for_queue = current_labels[:queue_size]
                        batch_size_curr = queue_size
                    else:
                        labels_for_queue = current_labels[:batch_size_curr]

                    labels_for_queue = labels_for_queue.detach().to(queue_labels.dtype)
                    end_ptr = queue_ptr + batch_size_curr
                    if end_ptr <= queue_size:
                        queue_feats[queue_ptr:end_ptr] = normalize_key[:batch_size_curr]
                        queue_labels[queue_ptr:end_ptr] = labels_for_queue
                    else:
                        first_len = queue_size - queue_ptr
                        queue_feats[queue_ptr:] = normalize_key[:first_len]
                        queue_labels[queue_ptr:] = labels_for_queue[:first_len]
                        remaining = batch_size_curr - first_len
                        if remaining > 0:
                            queue_feats[:remaining] = normalize_key[first_len:first_len + remaining]
                            queue_labels[:remaining] = labels_for_queue[first_len:first_len + remaining]
                    queue_ptr = (queue_ptr + batch_size_curr) % queue_size
                    queue_filled = min(queue_filled + batch_size_curr, queue_size)
            total_loss += loss.item()
            # print(loss.item())
        current_val_loss = compute_val_loss(model=model,
                                            dataloader_val=dataloader_val,
                                            loss_function=loss_contrastive_val,
                                            batch_size_val_test=BATCH_EVAL_TEST)
        wandb.log({
            "cl_epoch": step,
            "cl_loss_train": total_loss / len(dataloader_train),
            "cl_loss_val": current_val_loss
        })
        if current_val_loss <= best_loss_save:
            best_loss_save = current_val_loss
            save_best_model(model=model, config=config, score=current_val_loss)
        if total_loss/len(dataloader_train) <= best_loss_save_train:
            best_loss_save_train = total_loss/len(dataloader_train)
            save_best_model_train(model=model, config=config)
        total_loss = 0
    save_best_model_final(model=model, config=config)

    step += 1

    print(" === Reload Dataset ===")
    # Reload Dataset with training model equal to false because we not use it for training
    if config["task_type"] == "NLP":

        dataloader_train, dataloader_val, dataloader_test = dataloader.dataloader(batch_size=config["batch_size"],
                                                                                  max_len=config["max_len"],
                                                                                  tokenizer=tokenizer,
                                                                                  dataset_name=config["dataset_name"],
                                                                                  batch_size_val_test=BATCH_EVAL_TEST,
                                                                                  training_mode=False)
    else:  # Vision
        dataloader_train, dataloader_val, dataloader_test, _ = get_vision_loaders(dataset=config["dataset_name"],
                                                                                  batch_size=config["batch_size"],
                                                                                  img_size=config["img_size"],
                                                                                  augmentation= config["augmentation"],
                                                                                  seed=config["SEED"],
                                                                                  collate=config["collate"],
                                                                                  fraction=config["fraction"])
    print("=== The Training === ")
    
    print("Eval Model === ")
    if config["freeze_encoder"]:
        eval_fn = eval_model_linear_probe
    else:
        eval_fn = eval_model_finetune

    _, projection_final = eval_fn(
        step=step, name='val', model=model, dataloader_train=dataloader_train, dataloader_val=dataloader_val, config=config)
    # Reload best backbone
    print("Create Dataloader === ")
    dataloader_test_hidden = create_dataloader_hidden_space_test(model=model,
                                                                 dataloader_test=dataloader_test,
                                                                 batch_size=BATCH_EVAL_TEST,
                                                                 hidden=True)
    print("Saving preds === ")
    pred_test, target_test = get_all_preds(model=projection_final,
                                           dataloader=dataloader_test_hidden,
                                           device=DEVICE)
    print("Compute Metrics === ")
    config_res = compute_test_metrics(target_test.numpy(
    ), pred_test.numpy(), add_str='test', nb_class=config['nb_labels'])
    cl_test_step = 0
    log_data = {"cl_test_step": cl_test_step}
    log_data.update(config_res)

    # ablation study: evaluate embeddings
    run_ablation = config.get("run_ablation", True)

    if run_ablation:
        print("Collect hidden features for Representation Analysis === ")
        X_list, Y_list = [], []
        model.eval()
        with torch.no_grad():
            for hidden_batch, y_batch in dataloader_test_hidden:
                X_list.append(hidden_batch.cpu())
                Y_list.append(y_batch.cpu())
        X_test = torch.cat(X_list, dim=0).numpy()   # (N, D)
        Y_test = torch.cat(Y_list, dim=0).numpy()   # (N, L) multi-hot

        sil, dbi = evaluate_embedding(
            X_test, Y_test,
            keep_fraction=config["fraction"]  # 頻出ラベル組合せの上位50%
        )
        # アブレーション結果も同じステップのログにまとめて送信する
        log_data["silhouette_test"] = sil
        log_data["dbi_test"] = dbi
    else:
        print("Skipping ablation study (run_ablation=False)")

    wandb.log(log_data)

    wandb.finish()
    save_test_score(config, config_res)


def eval_model_linear_probe(step: int, name: str, model: nn.Module, dataloader_train, dataloader_val, config, nb_linear=3):
    """ Eval Model

    :param step: Step for Wandb
    :type step: int
    :param name: name of the checkpoint that we want to use
    :type name: str
    :param model: _description_
    :type model: nn.Module
    :param dataloader_train: Dataloader train
    :type dataloader_train: datalaoder
    :param dataloader_val: Dataloader for the developpement
    :type dataloader_val: _type_
    :param config: _description_
    :type config: _type_
    :param nb_linear: _description_, defaults to 3
    :type nb_linear: int, optional
    :return: _description_
    :rtype: _type_
    """
    # Save only the features of the backbone model
    # Create a grid search

    all_projection = {}
    # Build our dataloader Base on the saved version
    dataloader_train_h, dataloader_val_h = create_dataloader_hidden_space(model=model,
                                                                          dataloader_train=dataloader_train,
                                                                          dataloader_val=dataloader_val,
                                                                          batch_size=BATCH_EVAL_TEST,
                                                                          hidden=True)
    # Create all possible combinaison
    combinaisons = list(product(ALL_LR, ALL_WD))
    save_indiv_parameter = [["unknown", -100]
                            for _ in range(config['nb_labels'])]
    # Iterate on all combinaison
    for current_comb in combinaisons:
        lr, wd = current_comb[0], current_comb[1]
        key = str(lr) + "_" + str(wd)
        # Create our Multiple classifier
        linear_classifier_multiple = LinearEvaluationMultiple(
            nb_labels=config['nb_labels'], hidden_size=model.hidden_size).to(DEVICE)
        # Create Our classical optimizer
        for index in range(nb_linear):
            # Create single Linear
            single_linear = LinearEvaluation(
                nb_labels=config['nb_labels'], hidden_size=model.hidden_size).to(DEVICE)
            # Create our Optimizer
            optimizer = AdamW(
                params=single_linear.parameters_training(lr=lr, wd=wd))
            # train our classifier
            traine_linear_classifier(
                linear_classifier=single_linear, dataloader=dataloader_train_h, optim=optimizer)
            # Prepare Singke Multi linear layer
            set_multi_linear(multi_linear=linear_classifier_multiple,
                             single_lin=single_linear, index=index)
        # Pred on val with our new classifier
        pred_val, target_val = get_all_preds(model=linear_classifier_multiple,
                                             dataloader=dataloader_val_h,
                                             device=DEVICE)
        all_projection[key] = deepcopy(linear_classifier_multiple)
        # Save Best results individually
        indiv = compute_test_metrics_individual(
            y_true=target_val.numpy(), y_pred=pred_val.numpy())
        # save best parameter for each label inidividually
        save_indiv_parameter = [previous_label_score if previous_label_score[1] >= new_label_score else [
            key, new_label_score] for previous_label_score, new_label_score in zip(save_indiv_parameter, indiv)]
    print(save_indiv_parameter)
    print("======= Build Final Layer =========")
    final_projection_model = build_final_projection(
        all_projection=all_projection, best_param=save_indiv_parameter, config=config, hidden_size=model.hidden_size)
    print("========== Predict Final Layer ===========")
    pred_val, target_val = get_all_preds(model=final_projection_model,
                                         dataloader=dataloader_val_h,
                                         device=DEVICE)
    # Config res for our val
    config_res = compute_test_metrics(target_val.numpy(), pred_val.numpy(
    ), add_str=name, nb_class=config['nb_labels'])
    wandb.log({
        "classifier/epoch": step,
        "classifier/val_f1_micro": config_res["f1 micro " + name],
        "classifier/val_f1_macro": config_res["f1 macro " + name],
        "classifier/val_hamming_loss": config_res["hamming_loss " + name]
    })
    print(config_res)
    return config_res["f1 micro " + name], final_projection_model


def eval_model_finetune(step: int, name: str, model: nn.Module, dataloader_train, dataloader_val, config):
    linear_classifier = LinearEvaluation(
        nb_labels=config['nb_labels'], hidden_size=model.hidden_size).to(DEVICE)
    parameter_groups = model.parameters_training(lr_backbone=config["lr"],
                                                 lr_projection=config["lr_adding"],
                                                 wd=config["wd"])
    parameter_groups += linear_classifier.parameters_training(lr=config["lr_adding"],
                                                              wd=config["wd"])
    optimizer = AdamW(params=parameter_groups)
    traine_linear_classifier_end_to_end(linear_classifier=linear_classifier,
                                        model=model,
                                        dataloader=dataloader_train,
                                        dataloader_val=dataloader_val,
                                        optim=optimizer)
    _, dataloader_val_h = create_dataloader_hidden_space(model=model,
                                                         dataloader_train=dataloader_train,
                                                         dataloader_val=dataloader_val,
                                                         batch_size=BATCH_EVAL_TEST,
                                                         hidden=True)
    pred_val, target_val = get_all_preds(model=linear_classifier,
                                         dataloader=dataloader_val_h,
                                         device=DEVICE)
    config_res = compute_test_metrics(target_val.numpy(), pred_val.numpy(),
                                      add_str=name, nb_class=config['nb_labels'])
    wandb.log({
        "classifier/epoch": step,
        "classifier/val_f1_micro": config_res["f1 micro " + name],
        "classifier/val_f1_macro": config_res["f1 macro " + name],
        "classifier/val_hamming_loss": config_res["hamming_loss " + name]
    })
    print(config_res)
    return config_res["f1 micro " + name], linear_classifier


def set_multi_linear(multi_linear: nn.Module, single_lin: nn.Module, index: int):
    """ Save single linear inside the multi linear layer

    :param multi_linear: Saved Version
    :type multi_linear: nn.Module
    :param single_lin: Single linear
    :type single_lin: nn.Module
    :param index: Index to save
    :type index: int
    """
    # Set requires grad to false beacause we doesn't care
    for param in multi_linear.parameters():
        param.requires_grad = False
    # Set the parameters
    multi_linear.classier_multiple[index].weight.data = single_lin.classifier.weight.data.detach(
    )
    multi_linear.classier_multiple[index].bias.data = single_lin.classifier.bias.data.detach(
    )


def build_final_projection(all_projection, best_param, config, hidden_size: int) -> nn.Module:
    """Build the final projection layer

    :param all_projection: all projection saved by the model
    :type all_projection: List[nn.Module]
    :param best_param: 
    :type best_param: _type_
    :param config: Just for nb_label
    :type config: dict
    :return: Final Layer
    :rtype: nn.Module
    """
    final_linear = LinearEvaluationMultiple(
        nb_labels=config['nb_labels'], hidden_size=hidden_size).to(DEVICE)
    for param in final_linear.parameters():
        param.requires_grad = False
    for index, param in enumerate(best_param):
        for index_layer in range(len(final_linear.classier_multiple)):
            final_linear.classier_multiple[index_layer].weight[index] = all_projection[param[0]
                                                                                       ].classier_multiple[index_layer].weight[index].detach()
            final_linear.classier_multiple[index_layer].bias[index] = all_projection[param[0]
                                                                                     ].classier_multiple[index_layer].bias[index].detach()
    return final_linear


def train_BCE(config: Dict, dataloader_train: DataLoader, dataloader_val: DataLoader, dataloader_test: DataLoader, entity_name: str, num_labels: int,loss='bce',bce_loss='bce'):
    """Train the model using BCE loss or ZLPR loss (they will be put in the same wandb group called bce, which
    actually contains all non contrastive losses)
    
    :param config: Config for training
    :param dataloader_train: DataLoader for training
    :param dataloader_val: DataLoader for validation
    :param dataloader_test: DataLoader for testing
    :param entity_name: Entity name for wandb
    """
    config["loss"] = loss
    config["bce_loss"] = bce_loss
    config["merge_groupe"] = config["merge_groupe"].replace('bce', loss)
    wandb.init(project=config["project"] + '-bce',
               group=config["merge_groupe"],
               entity=entity_name,
               config=config)

    wandb.define_metric("classifier/epoch")
    wandb.define_metric("bce_loss_train", step_metric="classifier/epoch")
    wandb.define_metric("bce_loss_val", step_metric="classifier/epoch")
    wandb.define_metric("bce_loss_test", step_metric="classifier/epoch")
    wandb.define_metric("encoder_frozen", step_metric="classifier/epoch")
    wandb.define_metric("f1 micro val", step_metric="classifier/epoch")
    wandb.define_metric("f1 macro val", step_metric="classifier/epoch")
    wandb.define_metric("hamming_loss val", step_metric="classifier/epoch")
    wandb.define_metric("f1 micro test", step_metric="classifier/epoch")
    wandb.define_metric("f1 macro test", step_metric="classifier/epoch")
    wandb.define_metric("hamming_loss test", step_metric="classifier/epoch")
    wandb.define_metric("silhouette_test", step_metric="classifier/epoch")
    wandb.define_metric("dbi_test", step_metric="classifier/epoch")
    
    dic_loss = {"bce": nn.BCEWithLogitsLoss(), "ZLPR": Zlpr(), "asymetric": AsymmetricLoss(gamma_neg=3, gamma_pos=0, clip=0.3), "focal": AsymmetricLoss(gamma_neg=2, gamma_pos=0)}	
    # replace the string 'bce' in  config["merge_groupe"] by the loss function used
    
    loss_function = dic_loss[loss]
    
    bce_model = Baseline_2(backbone_path=config['name_model'],
                          nb_labels=config['nb_labels'],
                          projection_dim=config["projection_dim"],
                          task_type="NLP",
                          weights=config["weights"]).to(DEVICE)

    freeze_flag = config["freeze_encoder"]
    if freeze_flag:
        for param in bce_model.backbone.parameters():
            param.requires_grad = False
        bce_model.backbone.eval()

    optimizer = set_optimizer(config, bce_model, freeze_encoder=config["freeze_encoder"])

    lr_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                                num_warmup_steps=int(
                                                                    len(dataloader_train) * config["epochs"] * WARMUP),
                                                                num_training_steps=len(dataloader_train) * config["epochs"])
    scaler = GradScaler()

    best_f1_micro_val = float('-inf')
    best_model_state = None
    best_epoch = -1

    for epoch in range(config["epochs"]):
        bce_model.train()
        total_loss = 0
        for batch in tqdm(dataloader_train, desc="Training Batches"):
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda', dtype=torch.float16):
                inputs, attention_mask, labels = batch['input_ids'].to(
                    DEVICE), batch['attention_mask'].to(DEVICE), batch['labels'].to(DEVICE)
                outputs, _ = bce_model(inputs, attention_mask=attention_mask)
                loss = loss_function(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                bce_model.parameters(), max_norm=1.0, norm_type=2)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(dataloader_train)
        val_loss = 0.0

        # Reset metrics at the start of validation
        bce_model.eval()
        # Initialize tensors for storing predictions and labels
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for batch in dataloader_val:
                inputs, attention_mask, labels = batch['input_ids'].to(
                    DEVICE), batch['attention_mask'].to(DEVICE), batch['labels'].to(DEVICE)
                outputs, _ = bce_model(inputs, attention_mask=attention_mask)
                loss = loss_function(outputs, labels)
                val_loss += loss.item()
                all_val_preds.append(torch.sigmoid(outputs).cpu())
                all_val_labels.append(labels.cpu())

        all_val_preds = torch.cat(all_val_preds).numpy()
        all_val_labels = torch.cat(all_val_labels).numpy()

        metric_dic_val = compute_test_metrics(
            all_val_labels, all_val_preds, add_str='val', nb_class=num_labels)
        wandb.log({
            "classifier/epoch": epoch,
            "bce_loss_train": avg_train_loss,
            "bce_loss_val": val_loss / len(dataloader_val),
            **metric_dic_val
        })

        if metric_dic_val["f1 micro val"] > best_f1_micro_val:
            best_f1_micro_val = metric_dic_val["f1 micro val"]
            best_model_state = deepcopy(bce_model.state_dict())
            best_epoch = epoch
            print(f"New best model at epoch {epoch} with val f1 micro: {best_f1_micro_val:.4f}")

    bce_model.load_state_dict(best_model_state)
    print(f"Loaded best model weights from epoch {best_epoch} (val f1 micro: {best_f1_micro_val:.4f})")

    bce_model.eval()
    test_loss = 0.0
    all_test_preds = []
    all_test_labels = []
    X_list, Y_list = [], []

    with torch.no_grad():
        for batch in dataloader_test:
            inputs, attention_mask, labels = batch['input_ids'].to(
                DEVICE), batch['attention_mask'].to(DEVICE), batch['labels'].to(DEVICE)
            nan_mask = batch['nan_mask'].to(DEVICE)
            outputs, rep = bce_model(inputs, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs)
            if bool(nan_mask.any().item()):
                mask = nan_mask.unsqueeze(-1).bool()
                probs = probs.masked_fill(mask, 0.0)
            all_test_preds.append(probs.cpu())
            all_test_labels.append(labels.cpu())
            test_loss += loss_function(outputs, labels).item()
            X_list.append(rep.cpu())
            Y_list.append(labels.cpu())

    all_test_preds = torch.cat(all_test_preds).numpy()
    all_test_labels = torch.cat(all_test_labels).numpy()
    metric_dic_test = compute_test_metrics(
        all_test_labels, all_test_preds, add_str='test', nb_class=num_labels)

    run_ablation = config.get("run_ablation", True)

    if run_ablation:
        print("Collect hidden features for Representation Analysis === ")
        X_test = torch.cat(X_list, dim=0).numpy()
        Y_test = torch.cat(Y_list, dim=0).numpy()
        sil, dbi = evaluate_embedding(
            X_test, Y_test,
            keep_fraction=config.get("fraction", 0.5)
        )
    else:
        print("Skipping ablation study (run_ablation=False)")
        sil, dbi = None, None

    wandb.summary["bce_loss_test"] = test_loss / len(dataloader_test)
    wandb.summary["best_epoch"] = best_epoch
    wandb.summary["best_val_f1_micro"] = best_f1_micro_val
    for key, value in metric_dic_test.items():
        wandb.summary[key] = value
    if sil is not None:
        wandb.summary["silhouette_test"] = sil
        wandb.summary["dbi_test"] = dbi

    wandb.finish()
