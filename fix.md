# RCV1 NaN Error Investigation

## Error
```
ValueError: cannot convert float NaN to integer
```

発生箇所: `src/data/dataloader.py` line 54
```python
labels.append(torch.tensor(exemple[1].astype(int)))
```

## 原因の特定

### 1. 元データの問題
- `src/data/data/rcv1/rcv1v2-test.csv` に空のテキスト（NaN）を持つ行が **4行** 存在
  - 行番号: 4096, 92090, 463168, 766096
  - これらの行の `text` カラムが `NaN` (Python の Not a Number)

### 2. データ処理スクリプトの問題
- `src/data/data/rcv1/data_aapd.py` の190行目付近
```python
texts = df["text"].astype(str)  # 無加工
```
- **問題**: `astype(str)` を使うと、`NaN` が文字列 `"nan"` に変換される
- この `"nan"` 文字列がCSVファイル (`test.csv`) に保存される

### 3. データ読み込み時の問題
- `pd.read_csv()` がCSV内の文字列 `"nan"` を Python の `NaN` として解釈
- `src/data/read_dataset.py` の `clean_text()` 関数:
```python
def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    # ...
```
- これにより空文字列 `""` に変換され、`nan_mask` が正しく設定される

### 4. なぜエラーが発生するのか
- 問題の行では、ラベル列が `object` 型になっている可能性がある
- または、CSV内のデータの不整合により、ラベル値に `NaN` が混入している可能性

## 確認した事実
1. ✅ `rcv1v2-test.csv` に4行のNaNテキストが存在
2. ✅ `data_aapd.py` が `astype(str)` で `NaN` → `"nan"` 変換
3. ✅ `test.csv` に文字列 `"nan"` が含まれる (確認: `sed -n '4098p'`)
4. ✅ `read_dataset.py` は `nan_mask` を正しく設定
5. ❓ なぜ `dataloader.py` でエラーが出るのか（ラベル列の型の問題？）


## 方針・結論
- **前処理（data_aapd.py）でNaN対応は不要**
  - 学習・評価で使うのは `train.csv`, `dev.csv`, `test.csv` であり、これらは `read_dataset.py` 側でNaN/空文字列を正しく処理している
  - `clean_text()` でNaNは空文字列に変換され、`nan_mask`も正しく付与される
- **ラベル列にNaNが混入していなければ、現状の読み込み処理でエラーは発生しない**
  - 実際に `train.csv`, `dev.csv`, `test.csv` のラベル列にNaNは存在しないことを確認済み
- **よって、前処理側での特別なNaN除去・補完は不要**

## 今回のエラーについて
- エラーは「ラベル列にNaNが混入している場合」にのみ発生する
- しかし、現状のデータ生成・読み込みフローではラベル列にNaNは混入しない
- よって、現状のままでエラーは再現しない・解決される

---
（備考）
もし今後ラベル列にNaNが混入するようなデータ生成バグが発生した場合は、
`read_dataset.py` 側で `fillna(0)` などの追加処理を検討する。
