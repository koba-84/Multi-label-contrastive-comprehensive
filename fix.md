# freeze_encoder=True 時の Optimizer エラーと対処

## 現象
- コマンド例：`python3 src/main.py --config ...`
- `freeze_encoder=true` で学習を始めると、`torch.optim.optimizer.add_param_group` から `ValueError: can't optimize a non-leaf Tensor` が発生する。

## 原因
- `src/model/baseline_ours.py` および `src/model/baseline_ours_bce.py` の `parameters_training` で、プロトタイプ行列を `'params': self.prototype` のように単体で登録している。
- `nn.Parameter` は iterable として扱われるため、`_filter_parameter_groups` 内の `list(group.get("params", []))` でテンサの 0 次元ごとの値へ展開され、leaf ではない `Tensor` が出来てしまう。
- `freeze_encoder=true` のときは backbone 関連のパラメータが除外されるので、この non-leaf Tensor だけが optimizer に渡り、`ValueError: can't optimize a non-leaf Tensor` が発生する。

## 修正方針
1. プロトタイプをリストで包んで optimizer へ渡すよう変更する。
   - `src/model/baseline_ours.py` と `src/model/baseline_ours_bce.py` の `parameters_training` の該当行を `'params': [self.prototype]` とする。
2. `_filter_parameter_groups` では `group["params"] = params` を行い、空リストや `requires_grad=False` の要素は除外する（既に実装済み）。

## 実装例
```python
# src/model/baseline_ours.py
{'params': [self.prototype],
 'lr': lr_projection,
 'weight_decay': wd}

# src/model/baseline_ours_bce.py でも同様に修正
```

## 確認手順
- 上記修正後に `freeze_encoder=true` で再度実行し、Optimizer の初期化が通るか確認する。
- GradScaler の AssertionError も併発しないことを合わせてチェックする。
