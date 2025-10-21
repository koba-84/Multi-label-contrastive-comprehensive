import torch
import time
import pandas  as pd
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List
from .read_dataset import read_dataset
from transformers import AutoTokenizer


class DataSetCustom(Dataset):
    """:class: Custom Dataset

    :param dataframe: Dataset for training, developement and testing
    :type dataframe: pd.DataFrame
    """
    def __init__(self, dataframe: pd.DataFrame):
        super().__init__()
        # Set self.dataframe
        self.dataframe = dataframe

    def __getitem__(self, indice):
        # Return text, label vector, and nan_mask flag
        row = self.dataframe.iloc[indice]
        text = row.iloc[0]
        labels = row.iloc[1:-1]
        nan_mask = row.iloc[-1]
        return text, labels.to_numpy(), nan_mask

    def __len__(self):
        # Return the length of the dataframe
        return len(self.dataframe)


class MyCollateFunction():
    """:class: Collate Function to have an automatic padding

    :param dataframe: Dataset for training, developement and testing
    :type dataframe: pd.DataFrame
    """
    def __init__(self,
                 max_len: int,
                 tokenizer: AutoTokenizer):
        self.max_len = max_len
        self.tokenizer = tokenizer
        
    def __call__(self, batch):
        texts = []
        labels = []
        nan_masks = []
        for exemple in batch:
            # Save all text
            texts.append(exemple[0])
            # Save all labels
            try:
                labels.append(torch.tensor(exemple[1].astype(int)))
            except ValueError as e:
                print(f"Label values: {exemple[1]}")
                print(f"Label dtype: {exemple[1].dtype}")
                print(f"Contains NaN: {pd.isna(exemple[1]).any()}")
                print(f"Text: {exemple[0][:100]}...")  # 最初の100文字
                print(f"nan_mask: {exemple[2]}")
                raise e
            nan_masks.append(int(exemple[2]))
        inputs = self.tokenizer(texts,
                                truncation=True,
                                padding=True,
                                max_length=self.max_len)
        # Return dict which contains input, paddind, and labels
        return {'input_ids': torch.tensor(inputs['input_ids'], dtype=int).long(),
                'attention_mask': torch.tensor(inputs['attention_mask']).long(),
                'labels': torch.stack(labels).float(),
                'nan_mask': torch.tensor(nan_masks).long()}


def dataloader(batch_size: int,
               max_len: int,
               tokenizer: AutoTokenizer,
               dataset_name: str,
               batch_size_val_test: int,
               training_mode: bool):
    """Create dataloader for training, developement, testing

    :param batch_size: the length of the batch size
    :type batch_size: int
    :param max_len: the max len of the text
    :type max_len: int
    :param augmentation: boolean if we want to use data augmentation
    :type augmentation: bool
    :param tokenizer: tokenizer used to transform the text into token
    :type tokenizer: Tokenizer from hugging face
    :return: Dataloader for training, dev, validation
    :rtype: Tuple(dataloader)
    """
    # Download our dataset
    train, dev, test = read_dataset(name=dataset_name)
    my_collate = MyCollateFunction(max_len=max_len, tokenizer=tokenizer)
    train = DataLoader(DataSetCustom(dataframe=train),
                       batch_size=batch_size,
                       collate_fn=my_collate,
                       shuffle=True,
                       pin_memory=True,
                       num_workers=4,
                       drop_last=training_mode)
    dev = DataLoader(DataSetCustom(dataframe=dev),
                       batch_size=batch_size_val_test,
                       collate_fn=my_collate,
                       shuffle=False,
                       num_workers=4,
                       pin_memory=True)
    test = DataLoader(DataSetCustom(dataframe=test),
                       batch_size=batch_size_val_test,
                       collate_fn=my_collate,
                       shuffle=False,
                       num_workers=4,
                       pin_memory=True)
    return train, dev, test
