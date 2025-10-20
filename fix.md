
# trainer.py修正要件定義（traine_linear_classifier_end_to_endのdataloader_val対応）

## 目的
- basic_utils.pyのtraine_linear_classifier_end_to_endがdataloader_val引数を新たに受け取るようになったため、trainer.py側の呼び出しもそれに合わせて修正する。

## 要件
1. **eval_model_finetune内の呼び出し修正**
    - traine_linear_classifier_end_to_endの呼び出し時、dataloader_valを新たに引数として渡す。
    - それ以外の引数・返り値・処理は変更しない。

2. **dataloader_valの選択**
    - 既存のバリデーション用dataloader（hidden特徴空間でなく、通常のもの）をそのまま渡す。
    - dataloader_val_h（hidden特徴空間用）はwandbログや評価用で使うが、traine_linear_classifier_end_to_endには渡さない。

3. **他の箇所は一切変更しない**
    - traine_linear_classifier_end_to_end以外の呼び出しや、他の関数・ロジックは修正しない。

## 備考
- 既存のAPI互換性を保つため、必要最小限の修正に留める。
- 返り値やwandbログの扱いは現状維持。
