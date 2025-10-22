# LossContrastiveNWS 修正要件（DCL＋負例重み付け／ハイパラ更新）

## 対象
- `src/trainer/loss/loss_contrastive_nws.py` の内部実装（外部APIは最小限の変更方針を明記）

## ゴール
- NWS に Decoupled Contrastive Learning（分母＝負例のみ）を導入し、さらに負例にラベル間類似度に基づく重み付けを行う。
- ハイパラを整理（beta は維持、agg と sim を追加）。
- 学習前にデータ全体からラベルペア類似度行列 S (L×L) を算出し、Loss のインスタンス化時に渡す。


## NWS（DCL 適用, MoCo準拠）主公式＋補足

主要式（インスタンス i の損失と全体損失）
$$
\ell_i \;=\; -\frac{1}{|y_i|}\sum_{r \in \mathcal{P}_i^{+}} w_{ir}\; \log 
\frac{\exp\bigl((z_i^{\top} v_r)/\tau\bigr)}{\sum_{r' \in \mathcal{N}_i^{-}} \beta_{r'}\, a_{ir'}\, \exp\bigl((z_i^{\top} v_{r'})/\tau\bigr)}\,,
\qquad
\mathcal{L} \;=\; \frac{1}{B}\sum_{i=1}^B \ell_i.
$$

集合定義（DCL 適用, MoCo準拠）
- 参照集合 $\mathcal{R}_i = \mathcal{R}_i^{\text{key}} \cup \mathcal{R}_i^{\text{que}} \cup \mathcal{R}_i^{\text{proto}}$（現ミニバッチ内の in-batch 参照は含めない）。
- 正例集合 $\mathcal{P}_i^{+}$ と負例集合 $\mathcal{N}_i^{-}$：
	- キー/キュー: $r\in \mathcal{P}_i^{+}$ もし $\sum_{c=1}^L y_{i,c}\, y_{r,c} \ge 1$、それ以外は $\mathcal{N}_i^{-}$。
	- プロトタイプ: $p_c\in \mathcal{P}_i^{+}$ もし $y_{i,c}=1$、$p_c\in \mathcal{N}_i^{-}$ もし $y_{i,c}=0$。
- 分母は負例のみの総和: $\sum_{r'\in \mathcal{N}_i^{-}}$。

補足（定義）
- 参照と温度・内積
	- $\mathcal{R}_i$: 参照の集合（key、queue、プロトタイプ $c$）。
	- $\tau$: 温度。類似度は内積 $z_i^{\top} v_r$（$v_r\in\{k_k,q_q,p_c\}$）。
- 正例・負例（DCL）
	- 正例集合 $\mathcal{P}_i^{+}$、負例集合 $\mathcal{N}_i^{-}$ をマルチラベルの一致で定義（prototype は $L_r=\{c\}$ とみなす）。
- 分母の重み（β と類似度）
	- 類似度行列 $S\in[0,1]^{L\times L}$、インスタンスのラベル集合 $L_i=\{c\mid y_{i,c}=1\}$、参照のラベル集合 $L_r$ を用意（prototype は $L_r=\{c\}$、それ以外は対応ラベル集合）。
	- 集約類似度（agg=mean|max）
		$$
		\mathrm{sim}_{\mathrm{agg}}(i,r)=\begin{cases}
		\dfrac{1}{|L_i|\,|L_r|}\sum_{c\in L_i}\sum_{d\in L_r} S_{cd}\;\; (|L_i|\,|L_r|>0), & \text{mean},\\
		\max\limits_{c\in L_i,\, d\in L_r} S_{cd}\;\; (|L_i|>0,|L_r|>0), & \text{max}.
		\end{cases}
		$$
	- 分母の重み: 
		- キー/キュー: $a_{ir}=1-\mathrm{sim}_{\mathrm{agg}}(i,r)$
		- プロトタイプ: $a_{ir}=1$（プロトタイプは常に1）
	- セクション係数: $\beta_r=\beta$（キー/キュー）、$\beta_r=1$（プロトタイプ）（$\beta_{r}$ は参照 $r$ の所属セクションで決まる）。
- 分子重み $w_{ir}$ は MSC の定義を踏襲（本仕様では詳細省略）。
  - 実装規定: MSC の `w = torch.cat(parts_weights, dim=1) * mask_diagonal` パターンを使用。DCL 適用では、この w に正例マスク `pos_mask_concat` を乗算して `w_numerator = w * pos_mask_concat` とし、分子の重みとする。分子重みには類似度重み $a_{ir}$ を適用しない（分母側のみ適用）。

備考
- $z_i, p_c$ は L2 正規化済みを想定。
- 本式は現状実装の挙動（self 除外、β によるセクション重み、プロトタイプ重み=1、インスタンス内ラベル数での正規化）を忠実に表す。

## 仕様（契約）

### 1) API 変更（クラス生成）
- 旧: `LossContrastiveNWS(alpha, beta, temp)`
- 新: `LossContrastiveNWS(alpha, beta, temp, agg, sim)`
	- beta は引き続き使用（分母のセクション係数）
	- agg: ラベル間類似度の集約方法（'mean' または 'max'、デフォルトなし・必須引数）
	- sim: L×L のラベルペア類似度行列（`compute_label_pair_similarity` の結果を渡す、np.ndarray または torch.Tensor）
	- コンストラクタ内で sim を `self.sim = torch.as_tensor(sim, dtype=torch.float32)` で Tensor 化し保持（forward 時に device へ移動）
- `forward` の引数シグネチャは不変（呼び出し側の変更影響を最小化）

### 2) 事前計算関数（新規）
関数名（提案）: `compute_label_pair_similarity(Y, method) -> np.ndarray[L, L]`
- method: 'npmi' または 'jaccard'（デフォルトなし、呼び出し側で明示指定、未指定時はエラー）
- 入力: N×L の multi-hot 行列（学習データ全体のラベル指示、np.ndarray または torch.Tensor）
- 出力: 対称 L×L 行列（float32、np.ndarray）
- npmi:
	- p_i = count(Y[:,i]=1)/N, p_ij = count(Y[:,i]=1 & Y[:,j]=1)/N
	- PMI = log(p_ij / (p_i p_j)), NPMI = PMI / (-log p_ij)
	- 範囲 [-1,1] を [0,1] に線形変換: sim = (NPMI+1)/2。
	- 未定義ケースの扱い: p_ij=0 または p_i=0 または p_j=0 のときは sim=0 とする。
	- 対称性保証: 数値誤差対策として `S = (S + S.T) / 2` で強制対称化を推奨。
- jaccard:
	- 行列演算版: `intersection = Y.T @ Y`, `sum_i = Y.sum(0)`, `union = sum_i[:, None] + sum_i[None, :] - intersection`, `S = intersection / (union + 1e-10)` （ゼロ除算回避）
	- sim = |A∩B| / |A∪B|
- 対角は1.0。
- 配置: `src/trainer/loss/loss_contrastive_nws.py`（損失クラスと同一ファイル内）

### 3) DCL（分母＝負例のみ, MoCo準拠）
- 正例/負例の定義（マルチラベル）:
	- key: `pos_mask_key = (labels_query @ key_labels.T > 0).float()`, `neg_mask_key = 1 - pos_mask_key`
	- queue: `pos_mask_queue = (labels_query @ queue_labels.T > 0).float()`, `neg_mask_queue = 1 - pos_mask_queue`
	- prototype: `pos_mask_proto = labels_query` (B×L、y_{i,c}=1 なら正例), `neg_mask_proto = 1 - labels_query`
- 分母は負例集合のみで構成:
	- 各セクションの neg_mask に、セクション係数 `beta_section`（key/que=beta, proto=1）と類似度重み `a_ir` を乗算。
	- 全セクション（key, queue, proto）を concat して一括処理する（MSC の concatenation パターン）。
		```python
		# 参照総数（バッチ内は含めない）
		total_refs = K + Q + L
		# セクション係数ベクトル（concat 後の全参照に対応）
		normalize_mask = torch.full((total_refs,), beta, device=...)
		normalize_mask[total_refs - nb_labels:] = 1  # prototype 部分を1に
		# 分母マスク（DCL 用に負例のみ）
		mask_denom = neg_mask_concat * normalize_mask.unsqueeze(0) * a_ir_concat
		```
	- 参照集合に self は存在しないため、対角マスク（self 除外）は不要。`log_softmax_temp` に渡す mask は `mask_denom`（正例は分母から除外される）。
- 前提: キュー・プロトタイプを利用する本実装では、分母が空になる状況は発生しないため、フォールバックは不要。

### 4) 負例重み付け（ラベルペア類似度 → 集約 → 1から減算）
- 目的: ラベル的に関連する（共起しやすい）負例は弱く、全く無関係な負例は強く扱う（関連度で負例を重み付け）。
- 類似度行列 S（L×L, [0,1]）と各インスタンスのラベル集合を用いて、各セクションの集約類似度 sim_agg∈[0,1] を計算（MoCo準拠で key/queue/proto のみ）:
	- **mean 集約**（行列演算、高速）:
		- key (B×K): `sim_agg_key = (labels_query @ S @ key_labels.T) / (labels_query.sum(1, keepdim=True) @ key_labels.sum(1, keepdim=True).T)`
		- queue (B×Q): `sim_agg_queue = (labels_query @ S @ queue_labels.T) / (labels_query.sum(1, keepdim=True) @ queue_labels.sum(1, keepdim=True).T)`
		- prototype (B×L): `sim_agg_proto = (labels_query @ S) / (labels_query.sum(1, keepdim=True))`（prototype は L 個の one-hot とみなす）
	- **max 集約**（ブロードキャスト＋masked max）:
		- key (B×K):
			```python
			# S: L×L, labels_query: B×L, key_labels: K×L
			mask = labels_query.bool().unsqueeze(1).unsqueeze(3) & key_labels.bool().unsqueeze(0).unsqueeze(2)  # B×K×L×L
			sim_all = S.unsqueeze(0).unsqueeze(0).expand(B, K, -1, -1)
			sim_agg_key = sim_all.masked_fill(~mask, float('-inf')).amax(dim=(2,3))  # B×K
			```
		- queue (B×Q): 同様に `key_labels` を `queue_labels` に置換
		- prototype (B×L): `sim_agg_proto = (labels_query.unsqueeze(2) * S.unsqueeze(0)).amax(dim=1)[0]`（B×L）
- 負例の重み:
	- key/queue: `a_ir = 1 - sim_agg`
	- プロトタイプ: `a_ir_proto = torch.ones_like(sim_agg_proto)`（常に1、形状 B×L、類似度に基づく重み付けは行わない）
- 分母の重み: 参照ごとに `beta_section * a_ir` を neg_mask に乗算（セクション3参照）

### 5) 非変更点
- 分子の重み w、alpha の扱い、温度 temp、勾配計算の流れは維持
- `forward` の引数シグネチャはそのまま

### 6) マイグレーション注意
- trainer 側の変更（本仕様範囲外、別途対応）: 
	- 現状 trainer.py では `LossContrastiveNWS(alpha=1, beta=config["beta"], temp=config['temp'])` で beta を渡している。
	- 新仕様では、これに加えて `agg` と `sim` を引数に追加: `LossContrastiveNWS(alpha=1, beta=config["beta"], temp=config['temp'], agg='mean', sim=sim_matrix)`
	- sim_matrix の事前計算: trainer.py で loss インスタンス化の前に、全訓練データのラベル行列 Y_train (N×L) を収集し、`sim_matrix = compute_label_pair_similarity(Y_train, method='npmi')` を実行して結果を渡す。

---

## 実装手順（MoCo準拠, in-batch を参照に含めない）
1. `compute_label_pair_similarity` を `loss_contrastive_nws.py` 内に追加（N×L → L×L）。`npmi`/`jaccard` に対応。
2. NWS のコンストラクタを新仕様に更新: `__init__(self, alpha, beta, temp, agg, sim)` で agg と sim を受け取る。sim は内部で `self.sim = torch.as_tensor(sim, dtype=torch.float32)` で Tensor 化。
3. `compute_loss_contrastive` (または forward 内) で:
	 - sim を `self.sim.to(output_query.device)` で同一デバイスへ移動。
	 - **DCL 正例/負例マスク構築**（各セクションで pos_mask, neg_mask を作成）:
		 - key/queue/prototype のみ（in-batch は含めない）
	 - **集約類似度 sim_agg の計算**（agg=mean|max に応じて）:
		 - mean: 行列演算（B×K, B×Q, B×L の各形状に対応）
		 - max: ブロードキャスト＋masked max（B×K, B×Q, B×L）
	 - **負例重み a_ir の計算**:
		 - key/queue: `a_ir = 1 - sim_agg`
		 - prototype: `a_ir = torch.ones_like(sim_agg_proto)`（常に1、形状 B×L）
	 - **セクション係数 beta_r の適用**: 
		 - MSC の `normalize_mask` パターンを参考に、全セクションを concat した後に適用
		 - key/queue 部分に beta、prototype 部分に 1 を配置
	 - **分母マスクの構築**: `mask_denom = neg_mask_concat * normalize_mask.unsqueeze(0) * a_ir_concat`（各セクションを concat して一つのテンソルに）
	 - **分子重みの取得**: MSC の concat パターンを踏襲し、`w_numerator = w_concat * pos_mask_concat`（分子には a_ir を適用しない）
	 - **log-sum-exp 計算**: 
		 - MSC の `log_softmax_temp(matrix=final_features, temp=temp, mask=...)` を活用
		 - DCL 適用のため、mask には `mask_denom` を渡す（正例は分母から除外）
		 - 最終損失: `loss = - (log_softmax * w_numerator).sum(dim=1)`（MSC と同様のパターン）
4. 形状・dtype・device と数値安定性を点検。温度 τ は MSC の実装をそのまま使用（`log_softmax_temp` 関数）。
5. 全セクション統合方法: MSC の `torch.cat(parts_features, dim=1)` および `torch.cat(parts_weights, dim=1)` パターンを踏襲（各セクションの logits と weights を dim=1 で concat）。in-batch セクションは存在しない。

## 検証（最低限）
- [ ] ダミー（B=4, L=3）で forward/勾配が通る（key/queue on/off）
- [ ] 分母に正例が入らない（テンポラリ出力で確認）
- [ ] agg=mean と agg=max で形状が崩れない

## 補足（パフォーマンス）
- agg=max はラベル次元 L が大きい場合に計算負荷が高くなるため、`mean` をデフォルト推奨。
- 必要に応じて `max` 実装はブロック分割や近似で最適化可能（将来検討）。

## 備考
- 本メモは `loss_contrastive_nws.py` 専用。不要になったら随時更新/削除すること。

---


