import re
import os
import pandas as pd
from typing import Tuple


def clean_text(text: str) -> str:
    """ Basic function this function is a preprocessing function which remove basics special 
    characters

    :param text: The text where the preprocessing is applied
    :type text: str
    :return: Preprocessing text
    :rtype: str
    """
    text = re.sub(r"[^A-Za-z0-9(),!?\'`.]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.lower().strip()


def read_dataset(name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ Return train, val, test dataframe

    :param name: The name which describs the dataset
    :type name: str
    :return: Return of dataframe which contains train, val, test
    :rtype: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
    current_path = os.path.abspath(os.path.dirname(__file__))
    if name == 'aapd':
        path_train = os.path.join(current_path, "data/aapd/train.csv")
        path_dev = os.path.join(current_path, "data/aapd/dev.csv")
        path_test = os.path.join(current_path, "data/aapd/test.csv")
        train = pd.read_csv(path_train)
        train['abstract'] = train['abstract'].apply(lambda x: clean_text(x))
        dev = pd.read_csv(path_dev)
        dev['abstract'] = dev['abstract'].apply(lambda x: clean_text(x))
        test = pd.read_csv(path_test)
        test['abstract'] = test['abstract'].apply(lambda x: clean_text(x))
        # Construct correct train, dev split
        return train, dev, test
    elif name == "reuter":
        path_train = os.path.join(current_path, "data/reuter/train.csv")
        path_dev = os.path.join(current_path, "data/reuter/dev.csv")
        path_test = os.path.join(current_path, "data/reuter/test.csv")
        train = pd.read_csv(path_train)
        train.iloc[:, 0] = train.iloc[:, 0].apply(lambda x: clean_text(x))
        dev = pd.read_csv(path_dev)
        dev.iloc[:, 0] = dev.iloc[:, 0].apply(lambda x: clean_text(x))
        test = pd.read_csv(path_test)
        test.iloc[:, 0] = test.iloc[:, 0].apply(lambda x: clean_text(x))
        # Construct correct train, dev split
        return train, dev, test
    elif name == "rcv1":
        path_train = os.path.join(current_path, "data/rcv1/train.csv")
        path_dev = os.path.join(current_path, "data/rcv1/dev.csv")
        path_test = os.path.join(current_path, "data/rcv1/test.csv")
        train = pd.read_csv(path_train)
        train.iloc[:, 0] = train.iloc[:, 0].apply(lambda x: clean_text(x))
        dev = pd.read_csv(path_dev)
        dev.iloc[:, 0] = dev.iloc[:, 0].apply(lambda x: clean_text(x))
        test = pd.read_csv(path_test)
        test.iloc[:, 0] = test.iloc[:, 0].apply(lambda x: clean_text(x))
        # Construct correct train, dev split
        return train, dev, test
    elif name == 'uklex':
        path_train = os.path.join(current_path, "data/uklex/train.csv")
        path_dev = os.path.join(current_path, "data/uklex/dev.csv")
        path_test = os.path.join(current_path, "data/uklex/test.csv")
        train = pd.read_csv(path_train)
        train.iloc[:, 0] = train.iloc[:, 0].apply(lambda x: clean_text(x))
        dev = pd.read_csv(path_dev)
        dev.iloc[:, 0] = dev.iloc[:, 0].apply(lambda x: clean_text(x))
        test = pd.read_csv(path_test)
        test.iloc[:, 0] = test.iloc[:, 0].apply(lambda x: clean_text(x))
        # Construct correct train, dev split
        return train, dev, test
    elif name == "bgc":
        path_train = os.path.join(current_path, "data/bgc/train.csv")
        path_dev = os.path.join(current_path, "data/bgc/dev.csv")
        path_test = os.path.join(current_path, "data/bgc/test.csv")
        train = pd.read_csv(path_train)
        train.iloc[:, 0] = train.iloc[:, 0].apply(lambda x: clean_text(x))
        dev = pd.read_csv(path_dev)
        dev.iloc[:, 0] = dev.iloc[:, 0].apply(lambda x: clean_text(x))
        test = pd.read_csv(path_test)
        test.iloc[:, 0] = test.iloc[:, 0].apply(lambda x: clean_text(x))
        # Construct correct train, dev split
        return train, dev, test
    elif name == "aapd-10":
        path_train = os.path.join(current_path, "data/aapd-10/train.csv")
        path_dev = os.path.join(current_path, "data/aapd-10/dev.csv")
        path_test = os.path.join(current_path, "data/aapd-10/test.csv")
        train = pd.read_csv(path_train)
        train.iloc[:, 0] = train.iloc[:, 0].apply(lambda x: clean_text(x))
        dev = pd.read_csv(path_dev)
        dev.iloc[:, 0] = dev.iloc[:, 0].apply(lambda x: clean_text(x))
        test = pd.read_csv(path_test)
        test.iloc[:, 0] = test.iloc[:, 0].apply(lambda x: clean_text(x))
        # Construct correct train, dev split
        return train, dev, test
    elif name == "bgc-10":
        path_train = os.path.join(current_path, "data/bgc-10/train.csv")
        path_dev = os.path.join(current_path, "data/bgc-10/dev.csv")
        path_test = os.path.join(current_path, "data/bgc-10/test.csv")
        train = pd.read_csv(path_train)
        train.iloc[:, 0] = train.iloc[:, 0].apply(lambda x: clean_text(x))
        dev = pd.read_csv(path_dev)
        dev.iloc[:, 0] = dev.iloc[:, 0].apply(lambda x: clean_text(x))
        test = pd.read_csv(path_test)
        test.iloc[:, 0] = test.iloc[:, 0].apply(lambda x: clean_text(x))
        # Construct correct train, dev split
        return train, dev, test
    elif name == "rcv1-10":
        path_train = os.path.join(current_path, "data/rcv1-10/train.csv")
        path_dev = os.path.join(current_path, "data/rcv1-10/dev.csv")
        path_test = os.path.join(current_path, "data/rcv1-10/test.csv")
        train = pd.read_csv(path_train)
        train.iloc[:, 0] = train.iloc[:, 0].apply(lambda x: clean_text(x))
        dev = pd.read_csv(path_dev)
        dev.iloc[:, 0] = dev.iloc[:, 0].apply(lambda x: clean_text(x))
        test = pd.read_csv(path_test)
        test.iloc[:, 0] = test.iloc[:, 0].apply(lambda x: clean_text(x))
        # Construct correct train, dev split
        return train, dev, test
    else:
        raise ValueError("This dataset is undefined")

if __name__ == '__main__':
    train, dev, test = read_dataset(name='rcv1')
    print(train.shape, dev.shape, test.shape)
    
    