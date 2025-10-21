# --- 共通化バージョン ---
import re
import os
import pandas as pd
from typing import Tuple

# データセットごとのテキストカラム名を管理
TEXT_COLS = {
    "aapd": "abstract",
    "rcv1": "abstract",
    "rcv1s": "abstract",
    # 必要に応じて他も追加
}

def clean_text(text: str) -> str:
    """ Basic function: テキストの基本的な前処理（特殊文字除去など） """
    if pd.isna(text):
        return ""
    text = re.sub(r"[^A-Za-z0-9(),!?\'`.]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.lower().strip()

def _load_split_with_nan_mask(path: str, dataset_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    text_col_name = TEXT_COLS.get(dataset_name, df.columns[0])
    df["nan_mask"] = df[text_col_name].isna().astype("int8")
    df[text_col_name] = df[text_col_name].apply(clean_text)
    return df

def read_dataset(name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    current_path = os.path.abspath(os.path.dirname(__file__))
    def get_path(split):
        return os.path.join(current_path, f"data/{name}/{split}.csv")
    train = _load_split_with_nan_mask(get_path("train"), name)
    dev = _load_split_with_nan_mask(get_path("dev"), name)
    test = _load_split_with_nan_mask(get_path("test"), name)
    return train, dev, test

if __name__ == '__main__':
    train, dev, test = read_dataset(name='rcv1')
    print(train.shape, dev.shape, test.shape)
    
    