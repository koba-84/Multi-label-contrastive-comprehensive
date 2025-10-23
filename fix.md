# NWS 損失関数（現行実装に基づく定義）

## メインの損失式

バッチ $\mathcal{B}=\{(z_i, y_i)\}_{i=1}^B$ に対する NWS 損失は以下で定義される：

$$
\mathcal{L}_{\text{NWS}} = \frac{1}{B} \sum_{i=1}^B \frac{\mathcal{L}_i}{|y_i| + \varepsilon}
$$

ここで、サンプル $i$ の損失 $\mathcal{L}_i$ は

$$
\mathcal{L}_i = -\sum_{r \in \mathcal{R}} w_{i,r} \cdot \mathbb{1}[\text{pos}(i,r)] \cdot \log \frac{\exp(s_{i,r}/\tau)}{\sum_{u \in \mathcal{R}} \mathbb{1}[\text{neg}(i,u)] \cdot \beta_u \cdot a_{i,u} \cdot \exp(s_{i,u}/\tau) + \varepsilon}
$$

参照集合 $\mathcal{R} = K \cup Q \cup P$（key、queue、prototype）に対する確率は



類似度は $s_{i,r} = z_i^\top v_r$（正規化済み特徴の内積）。

---

## 記号の定義

- $z_i \in \mathbb{R}^F$: クエリ $i$ の特徴（$\|z_i\|_2=1$）
- $v_r \in \mathbb{R}^F$: 参照 $r$ の特徴（key/queue/proto、正規化済み）
- $y_i \in \{0,1\}^L$: クエリのマルチラベル
- $\tau > 0$: 温度パラメータ
- $\varepsilon > 0$: 数値安定化の小定数
- $\alpha, \beta > 0$: 重み調整係数
- $S \in \mathbb{R}^{L \times L}$: ラベル間類似度行列（対角は1）
- $\texttt{agg} \in \{\text{mean}, \text{max}\}$: 集約モード

---

## 補足1: 正例・負例の判定

参照 $r$ が正例か負例かは以下で判定される：

$$
\text{pos}(i,r) = \begin{cases}
\mathbb{1}[\langle y_i, y_k \rangle > 0] & \text{if } r=k \in K \\
\mathbb{1}[\langle y_i, y_q \rangle > 0] & \text{if } r=q \in Q \\
y_{i,\ell} & \text{if } r=\ell \in P
\end{cases}
$$

$$
\text{neg}(i,r) = 1 - \text{pos}(i,r)
$$

---

## 補足2: セクション係数 $\beta_r$ と非類似度 $a_{i,r}$

セクション係数（母数の正規化用）：

$$
\beta_r = \begin{cases}
\beta & \text{if } r \in K \cup Q \\
1 & \text{if } r \in P
\end{cases}
$$

非類似度（ラベル間の差異に基づく負例重み）：

$$
a_{i,r} = \begin{cases}
1 - \text{sim\_agg}(y_i, y_r) & \text{if } r \in K \cup Q \\
1 & \text{if } r \in P
\end{cases}
$$

ここで

$$
\text{sim\_agg}(y_i, y_r) = \begin{cases}
\dfrac{y_i^\top S y_r}{\|y_i\|_1 \|y_r\|_1 + \varepsilon} & (\texttt{agg}=\text{mean}) \\[10pt]
\max\limits_{c \in \text{supp}(y_i),\, d \in \text{supp}(y_r)} S_{cd} & (\texttt{agg}=\text{max})
\end{cases}
$$

---

## 補足3: 分子側の重み $w_{i,r}$（MSC由来）

各参照 $r$ に対する重みは、まずラベルごとの貢献度を計算し、正規化項 $D_{i,c}$ で除算して求める：

**1. key/queue の貢献度**（ラベル $c$ ごと）:

$$
M^{(K)}_{i,k,c} = \alpha \cdot \frac{y_{i,c} y_{k,c}}{|y_i \lor y_k|}
$$

$$
M^{(Q)}_{i,q,c} = \alpha \cdot \frac{y_{i,c} y_{q,c}}{|y_i \lor y_q|}
$$

**2. 正規化項** $D_{i,c}$（prototype 補正を含む）:

$$
D_{i,c} = \sum_{k \in K} M^{(K)}_{i,k,c} + \sum_{q \in Q} M^{(Q)}_{i,q,c} + \left(1 - \frac{\alpha}{|y_i|}\right) y_{i,c}
$$

**3. 参照ごとの重み**:

$$
w_{i,k} = \sum_{c=1}^L \frac{M^{(K)}_{i,k,c}}{D_{i,c} + \varepsilon}
$$

$$
w_{i,q} = \sum_{c=1}^L \frac{M^{(Q)}_{i,q,c}}{D_{i,c} + \varepsilon}
$$

$$
w_{i,\ell} = \frac{y_{i,\ell}}{D_{i,\ell} + \varepsilon} \quad (\ell \in P)
$$

---

## 補足4: 数値安定化（温度付きsoftmax）

実装では、指数演算のオーバーフローを防ぐため以下を適用：

$$
\text{logits}_{i,r} = \frac{s_{i,r}}{\tau} - \max_{u \in \mathcal{R}} \frac{s_{i,u}}{\tau}
$$

$$
\exp(\text{logits}_{i,r}) = e^{\text{logits}_{i,r}}
$$

これを用いて $p_{i,r}$ は

$$
p_{i,r} = \frac{\exp(\text{logits}_{i,r})}{\sum_{u \in \mathcal{R}} \mathbb{1}[\text{neg}(i,u)] \cdot \beta_u \cdot a_{i,u} \cdot \exp(\text{logits}_{i,u}) + \varepsilon}
$$

---

## 注意事項

- **DCL (Decoupled Contrastive Learning)** の設計に従い、**母数は負例のみ**で構成される。
- そのため、正例 $r$ に対して $\log p_{i,r}$ が正になり得るため、$\mathcal{L}_i$ が負値になる可能性がある。
- 損失を常に非負にするには、各正例の分子を母数に含める形（通常のsoftmax形式）への変更が必要。

---

**出典**: `src/trainer/loss/loss_contrastive_nws.py`（現行実装）
