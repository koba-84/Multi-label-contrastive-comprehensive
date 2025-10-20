# Ablation実験の条件付き実行に関する要件定義

## 概要
RCV1データセットでのablation実験（埋め込み評価: Silhouette係数, Davies-Bouldin指数）は計算時間がかかるため、ハイパーパラメータ探索時には実行をスキップできるようにする。

## 要件

### 1. 設定パラメータの追加
- `config`に`run_ablation`フラグを追加
  - デフォルト値: `True`（既存の動作を維持）
  - `False`に設定するとablation実験をスキップ

### 2. 実装箇所

#### 2.1. `trainer`関数（対照学習の場合）
- **場所**: 約400-417行目、対照学習後のテスト評価部分
- **対象コード**:
  ```python
  print("Collect hidden features for Representation Analysis === ")
  X_list, Y_list = [], []
  model.eval()
  with torch.no_grad():
      for hidden_batch, y_batch in dataloader_test_hidden:
          X_list.append(hidden_batch.cpu())
          Y_list.append(y_batch.cpu())
  X_test = torch.cat(X_list, dim=0).numpy()
  Y_test = torch.cat(Y_list, dim=0).numpy()
  
  sil, dbi = evaluate_embedding(
      X_test, Y_test,
      keep_fraction=config["fraction"]
  )
  wandb.log({"silhouette_test": sil, "dbi_test": dbi}, step=step)
  ```

#### 2.2. `train_BCE`関数
- **場所**: BCE学習の各エポックのテスト評価部分
- **対象処理**:
  - 埋め込み特徴量の収集
  - `evaluate_embedding()`の呼び出し
  - best_metric_dicへのsilhouette/dbiの追加

### 3. 実装方針

#### 3.1. 条件分岐の追加
- `config.get("run_ablation", True)`でフラグを取得
- `if run_ablation:`で埋め込み評価処理を囲む
- `False`の場合はスキップメッセージを出力

#### 3.2. wandbログの処理
- `run_ablation=True`: 従来通り`silhouette_test`, `dbi_test`をログ
- `run_ablation=False`: これらのメトリクスはログしない

#### 3.3. `train_BCE`での特別な処理
- best_metric_dicへの追加も条件付きにする
- `run_ablation=False`の場合、sil/dbiの計算自体をスキップ

### 4. 使用例

#### ハイパーパラメータ探索時（ablation実験をスキップ）
```python
config = {
    "run_ablation": False,
    "name": "rcv1",
    # ...その他のパラメータ
}
```

#### 通常の実験時（ablation実験を実行）
```python
config = {
    "run_ablation": True,  # または省略（デフォルト）
    "name": "rcv1",
    # ...その他のパラメータ
}
```

### 5. 期待される効果
- ハイパーパラメータ探索時の実行時間を大幅に短縮
- 必要に応じてablation実験の実行を柔軟に制御可能
- 既存の動作（デフォルトで実行）は維持

### 6. 注意事項
- `run_ablation=False`の場合、wandbにはsilhouette/dbiの値が記録されない
- 最終的な実験報告時は`run_ablation=True`で実行すること

## 実装の詳細

### 実装方針の確定事項

#### 1. `train_BCE`関数での変数初期化
- `run_ablation=False`の場合: `sil, dbi = None, None`を設定
- `run_ablation=True`の場合: `evaluate_embedding()`を呼び出して値を取得
- `best_metric_dic`への追加は`if run_ablation and sil is not None:`で条件分岐

#### 2. printメッセージの出力
- "Collect hidden features for Representation Analysis === "のメッセージも条件付きで出力
- `run_ablation=True`の場合のみ出力
- `run_ablation=False`の場合は"Skipping ablation study (run_ablation=False)"を出力

#### 3. X_list, Y_listの収集について
- `train_BCE`関数では、ablationをスキップする場合でも`X_list.append(rep.cpu())`と`Y_list.append(labels.cpu())`は実行される
- 理由: 条件分岐を複雑にしないため（パフォーマンスへの影響は軽微）
- `X_list`, `Y_list`の収集処理自体は変更しない

### 修正箇所の詳細

#### A. `trainer`関数（401-416行目）
**修正前:**
```python
print("Collect hidden features for Representation Analysis === ")
X_list, Y_list = [], []
model.eval()
with torch.no_grad():
    for hidden_batch, y_batch in dataloader_test_hidden:
        X_list.append(hidden_batch.cpu())
        Y_list.append(y_batch.cpu())
X_test = torch.cat(X_list, dim=0).numpy()
Y_test = torch.cat(Y_list, dim=0).numpy()

sil, dbi = evaluate_embedding(
    X_test, Y_test,
    keep_fraction=config["fraction"]
)
wandb.log({"silhouette_test": sil, "dbi_test": dbi}, step=step)
```

**修正後:**
```python
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
    X_test = torch.cat(X_list, dim=0).numpy()
    Y_test = torch.cat(Y_list, dim=0).numpy()

    sil, dbi = evaluate_embedding(
        X_test, Y_test,
        keep_fraction=config["fraction"]
    )
    wandb.log({"silhouette_test": sil, "dbi_test": dbi}, step=step)
else:
    print("Skipping ablation study (run_ablation=False)")
```

#### B. `train_BCE`関数（705-716行目）
**修正前:**
```python
print("Collect hidden features for Representation Analysis === ")
X_test = torch.cat(X_list, dim=0).numpy()
Y_test = torch.cat(Y_list, dim=0).numpy()
# compute and log the best test metrics based on the best validation f1 micro :
sil, dbi = evaluate_embedding(
    X_test, Y_test,
    keep_fraction=config["fraction"]
)

if metric_dic_val["f1 micro val"] > best_f1_micro_val:
    best_metric_dic = compute_test_metrics(
        all_test_labels, all_test_preds, add_str='test(best)', nb_class=num_labels)
    best_metric_dic["silhouette"] = sil
    best_metric_dic["dbi"] = dbi
```

**修正後:**
```python
run_ablation = config.get("run_ablation", True)

if run_ablation:
    print("Collect hidden features for Representation Analysis === ")
    X_test = torch.cat(X_list, dim=0).numpy()
    Y_test = torch.cat(Y_list, dim=0).numpy()
    sil, dbi = evaluate_embedding(
        X_test, Y_test,
        keep_fraction=config["fraction"]
    )
else:
    print("Skipping ablation study (run_ablation=False)")
    sil, dbi = None, None

# compute and log the best test metrics based on the best validation f1 micro :
if metric_dic_val["f1 micro val"] > best_f1_micro_val:
    best_metric_dic = compute_test_metrics(
        all_test_labels, all_test_preds, add_str='test(best)', nb_class=num_labels)
    if run_ablation and sil is not None:
        best_metric_dic["silhouette"] = sil
        best_metric_dic["dbi"] = dbi
```
