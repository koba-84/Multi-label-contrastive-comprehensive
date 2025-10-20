# trainer.py の nan_mask 対応計画（2025-10-20）

## 前提
- DataLoader から返るバッチ dict には `input_ids`, `attention_mask`, `labels`, `nan_mask` が含まれる。
- `nan_mask==1` のサンプルは評価時に出力を 0 へマスクしたい。

## 修正方針
1. **ラベル数計算の見直し**
   - すでに `config['nb_labels'] = train_data.shape[1] - 2` で対応済み。追加作業は不要。

2. **BCE 系評価／テスト処理の更新（`trainer.py::train_BCE`）**
   - `nan_mask == 1` となるのはテスト split のみ（train/dev は常に 0）なので、学習中の勾配処理は従来どおりで問題なし。
   - `train_BCE` 関数内の評価（validation loop）・テスト（test loop）で `batch['nan_mask']` を取得し、`torch.where(nan_mask.unsqueeze(-1) == 1, 0, predictions)` のように推論結果をゼロ化する。

3. **線形評価フェーズ（`trainer/basic_utils.py::create_dataset` など）**
   - `create_dataset` および `traine_linear_classifier_end_to_end` で生成する hidden-space DataLoader に `nan_mask` を保持させる必要があるか確認。
   - 必要であれば `DataSetCustom`（trainer/basic_utils.py 内の）や `create_dataset` に `nan_mask` を通過させ、`get_all_preds`（`utils/utils.py`）でゼロ化処理を適用する。

4. **ユーティリティ関数の調整（`utils/utils.py::get_all_preds` ほか）**
   - `get_all_preds` をはじめ、タプルアクセスに依存している関数を `Dict` 形式の batch を受け付けるように改修。
   - テスト時のみ `batch['nan_mask']` を受け取り、`nan_mask==1` をゼロ化してから `torch.cat` する処理を追加する。

5. **評価時のゼロ埋めロジック（`trainer.py` 内部 or 新規ヘルパー関数）**
   - `compute_test_metrics` 呼び出し前に `apply_nan_mask(preds, mask)` のようなヘルパーを定義し、`nan_mask==1` のサンプルについて `y_pred` をゼロ化する。
   - `train_BCE` のテストループや `eval_model_finetune` 内で共通的に呼べる形にすると再利用しやすい。

## 実装順序
1. `trainer.py` の BCE パスを中心に、`batch` アクセスを dict ベースへ統一。
2. `utils.get_all_preds` など共通処理を更新し、`nan_mask` を受け取れるようにする。
3. 評価結果から `nan_mask` サンプルを除外 or ゼロ化する処理を挿入。
4. 線形評価系で `nan_mask` が必要かを確認し、同様の処理を適用。

## メモ
- `nan_mask` は `int8` だが PyTorch 側では `long` テンソルとして扱っているため、そのまま 0/1 判定に使用できる。
- ミニバッチ単位で `nan_mask` を適用する際は、ブロードキャストしやすいよう `unsqueeze(-1)` でラベル次元へ合わせると便利。
