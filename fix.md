# wandb.define_metricを用いた適切なログ管理の要件定義

## 問題の背景

### 現状の問題
```
wandb: WARNING Tried to log to step 1 that is less than the current step 40. 
Steps must be monotonically increasing, so this data will be ignored.
```

- wandbの`step`は単調増加（monotonically increasing）である必要がある
- 現在のコードでは、**2つの異なる学習フェーズ**（Contrastive LearningとClassifier Learning）が存在し、それぞれが独自のepochカウンターを持つべきだが、step値が混在している
- そのため、特にClassifier Learning後のtest結果やablation結果がログされない問題が発生している

### 根本原因

本プロジェクトには**2つの独立した学習パイプライン**が存在する：



1. **Contrastive Learning（対照学習）系手法のパイプライン - `trainer`関数**:
     - **Phase 1**: Contrastive Learning Training
         - `cl_epoch = 0, 1, 2, ..., cl_epochs-1`でログ
         - `cl_loss_train`, `cl_loss_val`などをwandbに記録
     - **Phase 2**: Classifier Learning（finetune）
         - encoderと分類器（MLP等）をjointに学習する
         - train_BCEは通らず、finetuneの実装は`src/trainer/basic_utils.py`の`traine_linear_classifier_end_to_end`等で管理されている
         - linear probeは現状未対応（無視してよい）
         - `classifier/epoch = 0, 1, 2, ..., classifier_epochs-1`でwandbにログ
         - 現状のログは`finetune/val_f1_micro`など`finetune/`接頭辞付き（`traine_linear_classifier_end_to_end`参照）だが、finetuneとlinear probeの両方を共通化するため`classifier/`プレフィックスに統一する
         - 実装例：
```python
for epoch in range(classifier_epochs):
    # ... classifier training code ...
    wandb.log({
        "classifier/epoch": epoch,
        "classifier/val_f1_micro": f1_micro_val,
        "classifier/val_f1_macro": f1_macro_val,
        "classifier/val_hamming_loss": hamming_loss_val
    })
```
     - **Phase 3**: Test（最終評価）
         - 学習済み分類器でテストデータを評価
         - `cl_test_step = 0`（1回のみ）でテスト・ablation metrics（silhouette_test, dbi_test等）をwandbに記録
     - **問題**: これら3つのphaseでstep値が混在すると衝突が起きるため、wandb.define_metricで明確に分離する必要がある

2. **Classifier Learning（分類器学習）パイプライン - `train_BCE`関数**:
   - **単一のPhase**: End-to-End Classifier Training
     - `step = epoch (0, 1, 2, ..., classifier_epochs-1)`でログ
     - 各epochで: `bce_loss_train`, `bce_loss_val`, validation/test metrics, ablation metricsを全てログ
     - **問題**: 同じepoch値で複数回wandb.log()を呼び出すため、後続のログが無視される可能性がある

### 問題の本質

- **Contrastive LearningとClassifier Learningは別々の学習プロセス**であり、それぞれ独立したepoch軸を持つべき
- 現在の実装では単一のstep軸を共有しようとしているため、step値の衝突が発生
- 特に、Contrastive Learningが40 epochで完了した後、Classifier Learningが epoch=1 から開始しようとすると、「step 1 < step 40」という警告が発生

## 解決方針

### wandb.define_metricの活用
`wandb.define_metric()`を使用して、**2つの独立した学習パイプライン**に対して別々のepoch軸を定義する。

#### 学習パイプラインの分類

本プロジェクトには以下の2つの**独立した学習パイプライン**が存在：


1. **Contrastive Learning Pipeline (`trainer`関数)**
    - **CL Training Phase**: 表現学習のためのContrastive Learning
    - **Classifier Learning Phase (finetune)**: encoder+分類器のjoint学習（`traine_linear_classifier_end_to_end`等）
    - **Test/Ablation Phase**: 最終評価

2. **Classifier Learning Pipeline (`train_BCE`関数)**
   - **Classifier Training Phase**: End-to-Endでの分類器学習
   - 各epochでtrain/val/testを全て実行


#### Epoch軸の設計


各学習パイプライン・phaseに専用のepoch/step軸を割り当て：

- **`cl_epoch`**: Contrastive Learning Trainingのepoch進捗（0, 1, 2, ..., cl_epochs-1）
- **`classifier/epoch`**: Classifier Learning（finetune/BCE両方）のepoch進捗（0, 1, 2, ..., classifier_epochs-1）
- **`cl_test_step`**: CLパイプラインの最終テスト・ablation評価（常に0、1回のみ実行）

これにより、各phaseが完全に独立し、step値の衝突が発生しない。

### 実装戦略


#### 1. `trainer`関数（Contrastive Learning Pipeline）の修正

**Epoch/step軸の定義:**
- `cl_epoch`: Contrastive Learning Trainingで使用（0, 1, 2, ..., cl_epochs-1）
- `cl_test_step`: CLパイプラインの最終テスト・ablation評価で使用（常に0、1回のみ）



**wandb.define_metricの設定:**
```python
# wandb.init()の直後に設定
wandb.define_metric("cl_epoch")
wandb.define_metric("cl_test_step")

# Contrastive Learning Training Phase metrics
wandb.define_metric("cl_loss_train", step_metric="cl_epoch")
wandb.define_metric("cl_loss_val", step_metric="cl_epoch")

# Test/Ablation metrics（CLパイプラインの最終評価）
wandb.define_metric("f1 micro test", step_metric="cl_test_step")
wandb.define_metric("f1 macro test", step_metric="cl_test_step")
wandb.define_metric("hamming_loss test", step_metric="cl_test_step")
wandb.define_metric("silhouette_test", step_metric="cl_test_step")
wandb.define_metric("dbi_test", step_metric="cl_test_step")

# Classifier Learning Phase (finetune / linear probe) metrics
wandb.define_metric("classifier/epoch")
wandb.define_metric("classifier/val_f1_micro", step_metric="classifier/epoch")
wandb.define_metric("classifier/val_f1_macro", step_metric="classifier/epoch")
wandb.define_metric("classifier/val_hamming_loss", step_metric="classifier/epoch")
```

**ログの修正:**
```python
# ===== Phase 1: Contrastive Learning Training =====
for step in range(config["epochs"]):
    # ... training code ...
    wandb.log({
        "cl_epoch": step,
        "cl_loss_train": total_loss/len(dataloader_train),
        "cl_loss_val": current_val_loss
    })

# ===== Phase 2: Classifier Learning (finetune / linear probe) =====
for epoch in range(classifier_epochs):
    # ... classifier training code ...
    wandb.log({
        "classifier/epoch": epoch,
        "classifier/val_f1_micro": f1_micro_val,
        "classifier/val_f1_macro": f1_macro_val,
        "classifier/val_hamming_loss": hamming_loss_val
    })

# ===== Phase 3: Test/Ablation (CLパイプライン最終評価) =====
cl_test_step = 0
wandb.log({
    "cl_test_step": cl_test_step,
    "f1 micro test": f1_micro,
    "f1 macro test": f1_macro,
    "hamming_loss test": hamming_loss
})
wandb.log({
    "cl_test_step": cl_test_step,
    "silhouette_test": sil, 
    "dbi_test": dbi
})
```

#### 2. `train_BCE`関数（Classifier Learning Pipeline）の修正

**Epoch軸の定義:**
- `classifier/epoch`: Classifier Learningのepochで使用（0, 1, 2, ..., classifier_epochs-1）
- 各epoch内でtrain/val/testを全て実行し、同じ`classifier/epoch`値でログ
- Best metricsは`wandb.summary`で記録（epoch軸とは独立）

**wandb.define_metricの設定:**
```python
# wandb.init()の直後に設定
wandb.define_metric("classifier/epoch")

# Classifier Training Phase metrics（各epochで記録）
wandb.define_metric("bce_loss_train", step_metric="classifier/epoch")
wandb.define_metric("bce_loss_val", step_metric="classifier/epoch")
wandb.define_metric("bce_loss_test", step_metric="classifier/epoch")
wandb.define_metric("encoder_frozen", step_metric="classifier/epoch")

# Validation metrics（各epochで記録）
wandb.define_metric("f1 micro val", step_metric="classifier/epoch")
wandb.define_metric("f1 macro val", step_metric="classifier/epoch")
wandb.define_metric("hamming_loss val", step_metric="classifier/epoch")

# Test metrics（各epochで記録）
wandb.define_metric("f1 micro test", step_metric="classifier/epoch")
wandb.define_metric("f1 macro test", step_metric="classifier/epoch")
wandb.define_metric("hamming_loss test", step_metric="classifier/epoch")

# Ablation metrics（test時のみ、run_ablation=Trueの場合）
wandb.define_metric("silhouette_test", step_metric="classifier/epoch")
wandb.define_metric("dbi_test", step_metric="classifier/epoch")
```

**ログの修正:**
```python
# ===== Classifier Learning Training =====
for epoch in range(config["epochs"]):
    # ... training code ...
    
    # Validation metrics
    wandb.log({
        "classifier/epoch": epoch,
        "encoder_frozen": freeze_flag,
        "bce_loss_train": avg_train_loss,
        "bce_loss_val": val_loss / len(dataloader_val),
        # 例: metric_dic_valは f1 micro val, f1 macro val, hamming_loss val を含める
        **metric_dic_val
    })
    
    # Test metrics
    wandb.log({
        "classifier/epoch": epoch,
        "bce_loss_test": test_loss / len(dataloader_test),
        # 例: metric_dic_testは f1 micro test, f1 macro test, hamming_loss test を含める
        **metric_dic_test
    })
    
    # Ablation metrics（run_ablation=Trueの場合のみ、testタイミングで）
    if run_ablation:
        wandb.log({
            "classifier/epoch": epoch,
            "silhouette_test": sil, 
            "dbi_test": dbi
        })

# Best metricsはwandb.summaryで記録（epoch軸とは独立）
for key, value in best_metric_dic.items():
    wandb.summary[key] = value
```

## 実装の詳細

### 修正箇所

#### A. `trainer`関数（Contrastive Learning Pipeline）

**修正箇所1: wandb.init()直後にdefine_metricを追加**
- 場所: 約240行目、`wandb.init()`の直後
- 追加内容: 
```python
# Define custom step metrics for Contrastive Learning Pipeline
wandb.define_metric("cl_epoch")
wandb.define_metric("cl_test_step")

# Contrastive Learning Training Phase metrics
wandb.define_metric("cl_loss_train", step_metric="cl_epoch")
wandb.define_metric("cl_loss_val", step_metric="cl_epoch")

# Contrastive Learning Test/Ablation Phase metrics
wandb.define_metric("f1 micro test", step_metric="cl_test_step")
wandb.define_metric("f1 macro test", step_metric="cl_test_step")
wandb.define_metric("hamming_loss test", step_metric="cl_test_step")
wandb.define_metric("silhouette_test", step_metric="cl_test_step")
wandb.define_metric("dbi_test", step_metric="cl_test_step")

# Classifier Learning Phase (finetune / linear probe) metrics
wandb.define_metric("classifier/epoch")
wandb.define_metric("classifier/val_f1_micro", step_metric="classifier/epoch")
wandb.define_metric("classifier/val_f1_macro", step_metric="classifier/epoch")
wandb.define_metric("classifier/val_hamming_loss", step_metric="classifier/epoch")
```

**修正箇所2: `traine_linear_classifier_end_to_end`等のログキー統一**
- 場所: `src/trainer/basic_utils.py` の `traine_linear_classifier_end_to_end`（必要に応じて同様のログ処理を行う関数）
- 変更内容:
```python
# 修正前
wandb.log({
    "finetune/val_f1_micro": metrics["f1 micro finetune"],
    "finetune/val_f1_macro": metrics["f1 macro finetune"],
    "finetune/val_hamming_loss": metrics["hamming_loss finetune"],
    "finetune/epoch": epoch
})

# 修正後
wandb.log({
    "classifier/epoch": epoch,
    "classifier/val_f1_micro": metrics["f1 micro finetune"],
    "classifier/val_f1_macro": metrics["f1 macro finetune"],
    "classifier/val_hamming_loss": metrics["hamming_loss finetune"]
})
```
- `classifier/epoch`をstep軸にし、finetune/linear probeのログを共通フォーマットに統一する
- `trainer.py`内の`eval_model_linear_probe`と`eval_model_finetune`でも、`finetune/`接頭辞のキーをすべて`classifier/`接頭辞に揃える（例: `classifier/val_f1_micro`）ことでフェーズ間の命名整合性を保つ

**修正箇所3: CL Training loopのログ修正**
- 場所: 約340-344行目（Contrastive Learning Training loop内のwandb.log()呼び出し）
- 変更内容: 
```python
# 修正前
wandb.log({"cl_loss_train": total_loss/len(dataloader_train), 
           "cl_loss_val": current_val_loss}, step=step)

# 修正後
wandb.log({
    "cl_epoch": step,
    "cl_loss_train": total_loss/len(dataloader_train), 
    "cl_loss_val": current_val_loss
})

```

**修正箇所4: CL Evaluationのtest metricsログ修正**
- 場所: 約395-400行目（CL Training完了後のconfig_resをログする箇所）
- 変更内容:
```python
# 修正前
wandb.log(config_res, step=step)

# 修正後
cl_test_step = 0
log_data = {"cl_test_step": cl_test_step}
log_data.update(config_res)
wandb.log(log_data)
```

**修正箇所5: CL Evaluationのablation metricsログ修正**
- 場所: 約415行目（silhouette_test, dbi_testをログする箇所）
- 変更内容:
```python
# 修正前
wandb.log({"silhouette_test": sil, "dbi_test": dbi}, step=step)

# 修正後
wandb.log({
    "cl_test_step": cl_test_step,
    "silhouette_test": sil, 
    "dbi_test": dbi
})
```

#### B. `train_BCE`関数（Classifier Learning Pipeline）

**修正箇所1: wandb.init()直後にdefine_metricを追加**
- 場所: 約600行目、`wandb.init()`の直後
- 追加内容:
```python
# Define custom step metrics for Classifier Learning Pipeline
wandb.define_metric("classifier/epoch")

# Classifier Learning Training Phase metrics
wandb.define_metric("bce_loss_train", step_metric="classifier/epoch")
wandb.define_metric("bce_loss_val", step_metric="classifier/epoch")
wandb.define_metric("bce_loss_test", step_metric="classifier/epoch")
wandb.define_metric("encoder_frozen", step_metric="classifier/epoch")

# Validation metrics (各epochで記録)
wandb.define_metric("f1 micro val", step_metric="classifier/epoch")
wandb.define_metric("f1 macro val", step_metric="classifier/epoch")
wandb.define_metric("hamming_loss val", step_metric="classifier/epoch")

# Test metrics (各epochで記録)
wandb.define_metric("f1 micro test", step_metric="classifier/epoch")
wandb.define_metric("f1 macro test", step_metric="classifier/epoch")
wandb.define_metric("hamming_loss test", step_metric="classifier/epoch")

# Ablation metrics (run_ablation=Trueの場合のみ)
wandb.define_metric("silhouette_test", step_metric="classifier/epoch")
wandb.define_metric("dbi_test", step_metric="classifier/epoch")
```

**修正箇所2: Validation metricsのログ修正**
- 場所: 約665-675行目（validation後のwandb.log()）
- 変更内容:
```python
# 修正前
wandb.log({
    "encoder_frozen": freeze_flag,
    "bce_loss_train": avg_train_loss,
    "bce_loss_val": val_loss / len(dataloader_val),
    "learning_rate": lr_scheduler.get_last_lr()[0],
    **metric_dic_val
}, step=epoch)

# 修正後
wandb.log({
    "classifier/epoch": epoch,
    "encoder_frozen": freeze_flag,
    "bce_loss_train": avg_train_loss,
    "bce_loss_val": val_loss / len(dataloader_val),
    **metric_dic_val
})
```

**修正箇所3: Test metricsのログ修正**
- 場所: 約695-700行目（test後のwandb.log()）
- 変更内容:
```python
# 修正前
wandb.log({
    "bce_loss_test": test_loss / len(dataloader_test),
    **metric_dic_test
}, step=epoch)

# 修正後
wandb.log({
    "classifier/epoch": epoch,
    "bce_loss_test": test_loss / len(dataloader_test),
    **metric_dic_test
})
```

**修正箇所4: Ablation metricsのログ修正**
- 場所: 約715-720行目（silhouette/dbiのログ）
- 変更内容:
```python
# 修正前
wandb.log({"silhouette_test": sil, "dbi_test": dbi}, step=epoch)

# 修正後
        wandb.log({
            "classifier/epoch": epoch,
    "silhouette_test": sil, 
    "dbi_test": dbi
})
```

**修正箇所5: Best metricsの記録方法変更**
- 場所: 約735行目（best_metric_dicのログ）
- 変更内容:
```python
# 修正前
wandb.log(best_metric_dic)

# 修正後
# Best metricsはwandb.summaryで記録（epoch軸とは独立）
for key, value in best_metric_dic.items():
    wandb.summary[key] = value
```

## 期待される効果

1. **Step値の衝突完全回避**: 
   - **Contrastive Learning Pipeline**: `cl_epoch`（Training）と`cl_test_step`（Evaluation）で完全に独立
   - **Classifier Learning Pipeline**: `classifier/epoch`で独立
   - 2つのパイプライン間で一切のstep値衝突が発生しない
   
2. **ログの完全性**: 
   - 全てのメトリクスが正しく記録される
   - 「step must be monotonically increasing」エラーが発生しない
   
3. **可読性向上**: 
   - wandb UIで**2つの学習パイプライン**が明確に区別される
   - Contrastive Learningの進捗とClassifier Learningの進捗が独立して表示される
   - 各パイプラインのグラフが混在せず、分析が容易
   
4. **正確な実験追跡**: 
   - Contrastive Learning: Training進捗（`cl_epoch`）と最終評価（`cl_test_step=0`）を明確に分離
   - Classifier Learning: 各epochでのtrain/val/test進捗を一貫して追跡
   
5. **柔軟性**: 
   - 将来的に新しい学習パイプラインを追加する際も、独立したepoch軸を定義するだけで対応可能
   - メトリクスの追加も容易

6. **明確な命名**: 
   - `cl_epoch`: Contrastive Learning Training epoch
   - `cl_test_step`: Contrastive Learning Evaluation step
   - `cl_loss_*`: Contrastive Learningの損失ログ
   - `classifier/epoch`および`classifier/*`: finetune/linear probeのメトリクス群
   - `classifier/epoch`と`bce_loss_*`: BCE系パイプラインのステップと損失ログ
   - パイプラインと用途が一目瞭然

## 参考資料

- wandb公式ドキュメント: https://docs.wandb.ai/ref/python/log
- define_metric: https://docs.wandb.ai/guides/track/log/customize-logging-axes
