# freeze_encoder 実装タスクリスト

目的: `freeze_encoder` を必須フラグ化し、線形プローブと fine-tune を明示的に切り替えられるようにする。

## 1. 設定ファイルと入力チェック
- `src/config/config.json`: `freeze_encoder: true|false` を追加。
- `src/trainer/trainer.py`: `if "freeze_encoder" not in config: raise ValueError(...)` を追加。

## 2. trainer 内の処理分岐
- `eval_model` を `eval_model_linear_probe` にリネーム。
  - 定義部 (`src/trainer/trainer.py`) と同ファイル内の全呼び出し・コメント・ログを置換。
- 新関数 `eval_model_finetune` を追加。
- `trainer` の評価パートで `if config["freeze_encoder"]` による分岐:
  ```python
  if config["freeze_encoder"]:
      score, projection = eval_model_linear_probe(...)
  else:
      score, projection = eval_model_finetune(...)
  ```

## 3. eval_model_linear_probe
- 既存の挙動（`ALL_LR`×`ALL_WD` グリッドサーチ→`build_final_projection`）を維持。
- 返り値: `(f1 micro val, final_projection_model)`。
- wandb ログ: 現行コードと同じ `wandb.log(config_res, step=step)`。

## 4. eval_model_finetune
- グリッドサーチなし。`config` に記載の単一パラメータで学習。
- 学習処理:
  - `src/trainer/basic_utils.py` に `traine_linear_classifier_end_to_end` を追加。
    - 目的: 線形層 + エンコーダを同時更新。
    - 40 epoch 固定・戻り値 `None`・スケジューラ運用は `traine_linear_classifier` と同じ。
  - fine-tune では `LinearEvaluationMultiple` / `set_multi_linear` を使わず、単一の `LinearEvaluation` を学習。
  - Optimizer は `model.parameters_training(lr_backbone=config["lr"], lr_projection=config["lr_adding"], wd=config["wd"])` に、線形層の `parameters_training(lr=config["lr_adding"], wd=config["wd"])` を結合（線形層も `lr_adding` を共有）。
- 返り値: `(f1 micro val, trained_linear_evaluation)`。
- wandb ログ: `eval_model_linear_probe` と同じ形式で `wandb.log(config_res, step=step)`。
- 評価処理は共通: `create_dataloader_hidden_space(..., hidden=True)` を `torch.no_grad()` で呼び出す。

## 5. basic_utils.py 変更
- `traine_linear_classifier_end_to_end` を追加。
  - 引数: `(linear_classifier, model, dataloader, optimizer)`＋必要最小限。
  - 既存ヘルパーと同等のデータロード／スケジューラ構成。

## 6. Optimizer 周り
- `set_optimizer` のシグネチャを `set_optimizer(config, model, freeze_encoder)` に変更。
- `freeze_encoder=True` の場合はバックボーン関連パラメータグループを生成しない。
- 呼び出し箇所を全て更新: `trainer`, `train_BCE`, その他 `rg "set_optimizer"` でヒットするところ。

## 7. BCE (`train_BCE`)
- 先頭で `freeze_flag = config["freeze_encoder"]`。
- `freeze_flag` が True の場合: `bce_model.backbone.requires_grad_(False); bce_model.backbone.eval()`。
- `set_optimizer` 呼び出しを新シグネチャに合わせる。
- wandb に `encoder_frozen` を追加。

## 8. ドキュメント・設定ファイル
- すべての設定ファイル／ノートブックで `freeze_encoder` を必須項目として追記。
- README や論文用資料に新フラグの説明と、凍結・微調整の両条件を明記。

## 9. テストチェックリスト
- `freeze_encoder=true`: 線形プローブ実行、エンコーダ勾配ゼロ、ログが現行と一致。
- `freeze_encoder=false`: fine-tune 経路が実行され、バックボーンに勾配あり、返り値とログ形式が線形プローブと一致。

以上。
