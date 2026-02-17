# MATWM Implementation - Components, Hyperparameters, and Loss Functions

## 📋 概要

本ドキュメントは、MATWM (Multi-Agent Transformer World Model) 実装における全コンポーネントのハイパーパラメータ、学習設定、損失関数を体系的に整理したものです。

**実装ファイル:**
- `matwm_implementation.py`: コア実装（World Model, Replay Buffer）
- `matwm_agent.py`: エージェント実装（Actor-Critic, 学習ループ）
- `curiosity_reward.py`: 好奇心報酬モジュール
- `2026_MATWM_simple_tag_Implementation.ipynb`: 統合実行ノートブック

**参考論文:**
- MATWM: Deihim et al. (2025). "Transformer World Model for Sample Efficient Multi-Agent Reinforcement Learning". arXiv:2506.18537
- γ-Progress: Kim et al. (2020). "Active World Model Learning with Progress Curiosity"

---

# 1️⃣ MATWMConfig（全体設定）

## 1.1 環境パラメータ

| パラメータ名 | 値 | 説明 |
|------------|-----|------|
| `max_cycles` | 25 | エピソード最大ステップ数 |
| `num_agents` | 4 | エージェント数（adversary×3 + prey×1） |
| `obs_dims` | `{'adversary_0': 16, 'adversary_1': 16, 'adversary_2': 16, 'agent_0': 14}` | エージェント別観測次元 |
| `max_obs_dim` | 16 | 統一観測次元（ゼロパディング用） |
| `action_dim` | 5 | 行動空間次元（0-4: no-op, left, right, down, up） |

**学習:** なし（環境定義のみ）

---

## 1.2 World Model アーキテクチャ（論文 Table C.6）

| パラメータ名 | 値 | 説明 | 論文対応 |
|------------|-----|------|---------|
| `latent_dim` | 32 | 潜在変数次元 | Table C.6 |
| `num_classes` | 32 | カテゴリカル分布のクラス数 | Table C.6 |
| `hidden_dim` | 512 | Transformer隠れ層次元 | Table C.6 |
| `num_layers` | 2 | Transformerレイヤー数 | Table C.6（修正: 4→2） |
| `num_heads` | 8 | Attentionヘッド数 | Table C.6 |
| `encoder_hidden_dim` | 512 | Encoder MLP隠れ層次元 | Table C.6 |
| `encoder_hidden_layers` | 3 | Encoder MLPレイヤー数 | Table C.6 |

**学習:** あり（World Model全体で学習）

---

## 1.3 Teammate Predictor アーキテクチャ

| パラメータ名 | 値 | 説明 |
|------------|-----|------|
| `teammate_hidden_dim` | 256 | MLP隠れ層次元 |

**学習:** あり（World Model損失の一部として学習）

**特記事項:** 
- 入力に stop-gradient を適用（論文 L140）
- 他エージェントの行動を予測（社会的世界モデリング）

---

## 1.4 Agent（Actor-Critic）アーキテクチャ

| パラメータ名 | 値 | 説明 |
|------------|-----|------|
| `actor_hidden_dim` | 256 | Actor MLP隠れ層次元 |
| `critic_hidden_dim` | 256 | Critic MLP隠れ層次元 |

**学習:** あり（Actor, Critic それぞれ個別に学習）

---

## 1.5 訓練パラメータ（論文 Table C.6 + Appendix C）

| パラメータ名 | 値 | 説明 | 論文対応 |
|------------|-----|------|---------|
| `wm_batch_size` | 16 | World Model バッチサイズ（シーケンス数） | Table C.6 |
| `wm_batch_length` | 64 | World Model 訓練時のシーケンス長 | Table C.6 |
| `agent_batch_size` | 768 | Agent バッチサイズ（4-6エージェント設定） | Appendix C |
| `sequence_length` | 64 | ~~最大シーケンス長~~（未使用、`wm_batch_length`と同義） | Table C.6 |
| `imagination_horizon` | 12 | Imagination rollout 長（4-6エージェント設定） | Appendix C |
| `imagination_context_length` | 8 | Imagination 開始用コンテキスト長 | - |

**学習:** なし（訓練ハイパーパラメータ）

**補足:**
- 論文 Appendix C: 4-6エージェント環境では `batch_size=768`, `imagination_horizon=12` を推奨
- `simple_tag` は4エージェントなのでこの設定を採用
- **注意:** `sequence_length` と `wm_batch_length` は現在同じ値（64）で、実装上は `wm_batch_length` のみ使用されている。`sequence_length` は将来的に可変長シーケンス対応のために残されているが、現状は冗長なパラメータ

---

## 1.6 学習率とGradient Clipping（論文 Table C.6）

| パラメータ名 | 値 | 説明 | 論文対応 |
|------------|-----|------|---------|
| `wm_learning_rate` | 3e-5 | World Model 学習率 | Table C.6 |
| `agent_learning_rate` | 3e-4 | Actor+Critic 学習率 | Table C.6 |
| `gradient_clip_wm` | 1000.0 | World Model gradient clipping | Table C.6 |
| `gradient_clip_agent` | 10.0 | Actor+Critic gradient clipping（修正: 100→10） | Table C.6（修正版） |

**学習:** なし（最適化ハイパーパラメータ）

**修正理由:** `gradient_clip_agent=100` では Critic のスパイクが発生したため、10 に下げて安定化

---

## 1.7 強化学習パラメータ（論文 Equations 11-14）

| パラメータ名 | 値 | 説明 | 論文対応 |
|------------|-----|------|---------|
| `gamma` | 0.99 | 割引率 | - |
| `lambda_gae` | 0.95 | GAE λ（λ-return計算用） | Equation 12 |
| `entropy_coef` | 0.01 | エントロピー正則化係数 η | Equation 11 |
| `critic_ema_decay` | 0.98 | Critic EMA 減衰率 σ | Equation 14 |

**学習:** なし（RL損失計算パラメータ）

**補足:**
- `lambda_gae`, `critic_ema_decay` は論文に明示されていないが、DreamerV3のデフォルト値を採用

---

## 1.8 γ-Progress パラメータ（Kim et al. 2020）

| パラメータ名 | 値 | 説明 | 論文対応 |
|------------|-----|------|---------|
| `use_gamma_progress` | False | γ-Progress を有効化（デフォルト: 無効） | - |
| `gamma_progress` | 0.9995 | World Model EMA 減衰率（θ_old更新用） | Kim et al. Eq.11 |
| `gamma_progress_weight` | 1.0 | γ-Progress 内発的報酬の重み | - |
| `gamma_progress_normalize` | True | γ-Progress 報酬を正規化 | - |

**学習:** なし（好奇心報酬パラメータ）

**補足:**
- Ablation study 用: `use_gamma_progress=True` で γ-Progress を有効化可能
- γ-Progress: 学習進捗（World Model の予測精度向上）を報酬化

---

## 1.9 Replay Buffer パラメータ（論文 Table C.6）

| パラメータ名 | 値 | 説明 | 論文対応 |
|------------|-----|------|---------|
| `buffer_size` | 50000 | Replay buffer サイズ（自動調整: max(total_steps, 50000)） | Table C.6 |
| `warmup_steps` | 1000 | ランダム行動ステップ数（学習開始前） | - |
| `priority_decay` | 0.9998 | Prioritized sampling 優先度減衰率 | Table C.6 |

**学習:** なし（データサンプリングパラメータ）

**補足:**
- `priority_decay=0.9998` は非常に緩やか。より速い適応には 0.995-0.997 を推奨

---

## 1.10 World Model 損失重み（論文 Equation 3, Table C.6）

| パラメータ名 | 値 | 説明 | 論文対応 |
|------------|-----|------|---------|
| `kl_weight` | 0.5 | β₁: Dynamics loss 重み | Table C.6 |
| `representation_weight` | 0.1 | β₂: Representation loss 重み | Table C.6 |
| `free_nats` | 1.0 | KL損失の下限（Free bits） | Equations 9a, 9b |

**学習:** なし（損失計算重み）

**補足:**
- `L_rec`, `L_rew`, `L_con`, `L_team` の重みは論文で明示されていないため、暗黙的に 1.0

---

## 1.11 訓練スケジュール

| パラメータ名 | 値 | 説明 |
|------------|-----|------|
| `train_wm_every` | 1 | World Model を N ステップごとに訓練 |
| `train_agent_every` | 1 | Agent を N ステップごとに訓練 |
| `total_steps` | 50000 | 総訓練ステップ数（論文推奨: simple環境で50K） |

**学習:** なし（訓練スケジュール）

---

## 1.12 ログ・保存

| パラメータ名 | 値 | 説明 |
|------------|-----|------|
| `log_interval` | 100 | ログ出力間隔 |
| `save_interval` | 5000 | モデル保存間隔 |
| `eval_interval` | 1000 | 評価間隔 |

**学習:** なし（ロギングパラメータ）

---

---

# 2️⃣ World Model 学習

## 2.1 概要

**学習対象:** Encoder, Decoder, DynamicsModel, RewardPredictor, ContinuationPredictor, TeammatePredictor

**学習の有無:** ✅ **あり**

**最適化:**
- Optimizer: Adam
- Learning rate: `3e-5`
- Gradient clipping: `1000.0`

---

## 2.2 損失関数（論文 Equation 3）

### 総損失

```
L(φ) = 1/(BT) Σ[L_rec + L_rew + L_con + L_team + β₁L_dyn + β₂L_rep]
```

where:
- `B`: バッチサイズ（16シーケンス）
- `T`: シーケンス長（64ステップ）
- `β₁ = 0.5` (`kl_weight`)
- `β₂ = 0.1` (`representation_weight`)

---

### 2.2.1 Reconstruction Loss `L_rec`（Equation 4）

**定義:**
```
L_rec = (ô_t - o_t)²
```

**実装:**
```python
obs_recon = world_model.decode(z)
recon_loss = F.mse_loss(obs_recon, obs_batch)
```

**説明:**
- Decoder が潜在変数 `z` から観測 `o` を再構成
- MSE（平均二乗誤差）で観測の復元精度を測定

**学習対象:** Encoder, Decoder

**重み:** 1.0（暗黙的）

---

### 2.2.2 Reward Loss `L_rew`（Equation 5）

**定義:**
```
L_rew = L_sym(r̂_t, r_t)
```
- `L_sym`: Symlog two-hot cross-entropy loss

**実装:**
```python
reward_logits = world_model.predict_reward(z_next_pred)
reward_symlog = symlog(reward_batch)
reward_target = two_hot_encode(reward_symlog)
reward_loss = F.cross_entropy(
    reward_logits.reshape(-1, 255),
    reward_target.reshape(-1, 255).argmax(dim=-1)
)
```

**説明:**
- 次状態潜在変数 `z_next` から報酬を予測
- Symlog 変換で報酬値の範囲を圧縮（-20〜20の範囲に正規化）
- Two-hot encoding で連続値を離散分布に変換（255 bins）

**学習対象:** RewardPredictor

**重み:** 1.0（暗黙的）

---

### 2.2.3 Continuation Loss `L_con`（Equation 6）

**定義:**
```
L_con = c_t log ĉ_t + (1-c_t)log(1-ĉ_t)
```
- `c_t = 1 - done_t`: 継続フラグ

**実装:**
```python
cont_logits = world_model.predict_continuation(z_next_pred)
cont_target = 1.0 - done_batch
cont_loss = F.binary_cross_entropy_with_logits(cont_logits, cont_target)
```

**説明:**
- 次状態潜在変数 `z_next` からエピソード継続確率を予測
- Binary cross-entropy でエピソード終了を学習

**学習対象:** ContinuationPredictor

**重み:** 1.0（暗黙的）

---

### 2.2.4 Teammate Loss `L_team`（Equation 8）

**定義:**
```
L_team = -ΣΣ δ(a_t,i=a) log p̂_t,i^(a)
```
- `δ(a_t,i=a)`: 他エージェント i の実際の行動が a のとき1
- `p̂_t,i^(a)`: 他エージェント i の行動 a の予測確率

**実装:**
```python
# ★ CRITICAL: Stop-gradient on input (論文 L140)
z_detached = z.detach()
teammate_logits_dict = world_model.predict_teammates(z_detached, focal_agent_idx)

teammate_loss = 0.0
for other_agent_idx, logits in teammate_logits_dict.items():
    actual_action = other_actions_by_idx.get(other_agent_idx)
    if actual_action is not None:
        target = torch.LongTensor([actual_action]).to(device)
        teammate_loss += F.cross_entropy(logits, target)
teammate_loss = teammate_loss / count
```

**説明:**
- 自エージェントの潜在変数 `z` から他エージェントの行動を予測
- **重要:** 入力 `z` に stop-gradient を適用（論文 L140 で明示）
- Cross-entropy で他エージェント行動の予測精度を測定

**学習対象:** TeammatePredictor

**重み:** 1.0（暗黙的）

**特記事項:**
- Stop-gradient により、Encoder が TeammatePredictor の勾配の影響を受けない
- これにより Encoder の学習安定化とノイズ耐性が向上

---

### 2.2.5 Dynamics Loss `L_dyn`（Equation 9a）

**定義:**
```
L_dyn = max(1, KL[sg(q(z_t+1|o_t+1)) || g^D(ẑ_t+1|h_t)])
```
- `sg(·)`: Stop-gradient
- `q(z_t+1|o_t+1)`: Encoder による次状態分布（target）
- `g^D(ẑ_t+1|h_t)`: DynamicsModel による次状態予測分布

**実装:**
```python
# Target: Encoder distribution with stop-gradient
z_next_target_dist = F.softmax(z_next_logits_target.detach(), dim=-1)
# Prediction: DynamicsModel distribution
z_next_pred_dist = F.softmax(z_next_pred_logits, dim=-1)

dynamics_loss = F.kl_div(
    F.log_softmax(z_next_pred_logits.reshape(-1, num_classes), dim=-1),
    z_next_target_dist.reshape(-1, num_classes),
    reduction='batchmean'
)
dynamics_loss = torch.maximum(dynamics_loss, torch.tensor(free_nats))
```

**説明:**
- DynamicsModel の予測分布を Encoder の真の分布に近づける
- Target（Encoder）に stop-gradient を適用
- KL divergence の下限を `free_nats=1.0` に設定（過学習防止）

**学習対象:** DynamicsModel

**重み:** `β₁ = 0.5` (`kl_weight`)

---

### 2.2.6 Representation Loss `L_rep`（Equation 9b）

**定義:**
```
L_rep = max(1, KL[q(z_t+1|o_t+1) || sg(g^D(ẑ_t+1|h_t))])
```
- Encoder 分布（no stop-gradient）を DynamicsModel 分布（stop-gradient）に近づける

**実装:**
```python
# Target: Encoder distribution (no stop-gradient)
z_next_target_dist_no_sg = F.softmax(z_next_logits_target, dim=-1)
# Prediction: DynamicsModel distribution with stop-gradient
z_next_pred_dist_sg = F.softmax(z_next_pred_logits.detach(), dim=-1)

representation_loss = F.kl_div(
    F.log_softmax(z_next_logits_target.reshape(-1, num_classes), dim=-1),
    z_next_pred_dist_sg.reshape(-1, num_classes),
    reduction='batchmean'
)
representation_loss = torch.maximum(representation_loss, torch.tensor(free_nats))
```

**説明:**
- Encoder が DynamicsModel にとって予測しやすい表現を学習
- Prediction（DynamicsModel）に stop-gradient を適用
- Encoder と DynamicsModel の協調学習を促進

**学習対象:** Encoder

**重み:** `β₂ = 0.1` (`representation_weight`)

---

## 2.3 World Model 学習の特徴

### Stop-Gradient の役割

| 損失 | Stop-gradient 適用先 | 目的 |
|-----|-------------------|------|
| `L_team` | 入力潜在変数 `z` | Encoder が TeammatePredictor の勾配を受けない |
| `L_dyn` | Target 分布（Encoder） | DynamicsModel が Encoder に近づく（一方向） |
| `L_rep` | Prediction 分布（DynamicsModel） | Encoder が DynamicsModel に近づく（一方向） |

### 学習の流れ

1. **データサンプリング:** Prioritized Replay Buffer から `16シーケンス × 64ステップ` をサンプル
2. **Forward pass:** 全損失を計算
3. **Backward pass:** 勾配計算（stop-gradient に注意）
4. **Optimizer step:** Adam で更新（lr=3e-5, grad_clip=1000.0）

---

---

# 3️⃣ Actor-Critic 学習

## 3.1 概要

**学習対象:** Actor, Critic, Critic EMA

**学習の有無:** ✅ **あり**

**最適化:**
- Optimizer: Adam（Actor, Critic 別々）
- Learning rate: `3e-4`
- Gradient clipping: `10.0`

---

## 3.2 Imagination Rollout

Actor-Critic は World Model で生成した **想像上のトラジェクトリ** で学習する。

### Rollout プロセス

1. Replay buffer から `768個` の開始状態をサンプル
2. 各開始状態で `imagination_horizon=12` ステップの rollout を実行
   - Actor でaction選択 → World Model で次状態・報酬・継続予測
3. Rollout 全体で `768 × 12 = 9,216` のトラジェクトリを生成

---

## 3.3 λ-Return の計算（Equation 12）

**定義:**
```
G_t^λ = r_t + γc_t[(1-λ)V(s_t+1) + λG_t+1^λ]
```
- `λ = 0.95` (`lambda_gae`)
- `c_t`: 継続フラグ（1 - done）

**実装（GAE）:**
```python
# TD error
deltas = rewards + gamma * continuations * values[:, 1:] - values[:, :-1]

# GAE
advantages = []
gae = 0
for t in reversed(range(imagination_horizon)):
    gae = deltas[:, t] + gamma * lambda_gae * continuations[:, t] * gae
    advantages.insert(0, gae)
advantages = torch.stack(advantages, dim=1)

# λ-return
lambda_returns = advantages + values[:, :-1]
```

**説明:**
- GAE (Generalized Advantage Estimation) で λ-return を計算
- Advantage = λ-return - Value

---

## 3.4 Actor Loss（Equation 11, 13）

**定義:**
```
L(θ) = -sg((G_t^λ - V)/max(1,S)) ln π - η H(π)
```
- `S = max(1, percentile(G, 95) - percentile(G, 5))`: Percentile-based normalization
- `η = 0.01` (`entropy_coef`)
- `H(π)`: エントロピー（探索促進）

**実装:**
```python
# Percentile normalization (Equation 13)
percentile_95 = torch.quantile(lambda_returns.reshape(-1), 0.95)
percentile_5 = torch.quantile(lambda_returns.reshape(-1), 0.05)
S = torch.maximum(torch.tensor(1.0), percentile_95 - percentile_5)

# Normalized advantages
advantages_normalized = (lambda_returns - values) / S

# Actor loss
actor_loss = -(action_log_probs * advantages_normalized.detach()).mean() \
             - entropy_coef * entropies.mean()
```

**説明:**
- Policy gradient with advantage normalization
- Percentile normalization で外れ値に頑健
- Entropy bonus で探索を促進

**学習対象:** Actor

**重み:** Policy gradient 項（1.0）, Entropy 項（0.01）

---

## 3.5 Critic Loss（Equation 11）

**定義:**
```
L(ψ) = (V - sg(G_t^λ))² + (V - sg(V_EMA))²
```
- 2項の MSE loss
  1. λ-return との誤差
  2. EMA Critic との誤差（正則化）

**実装:**
```python
# Two components of critic loss
critic_loss_lambda = F.mse_loss(values, lambda_returns.detach())
critic_loss_ema = F.mse_loss(values, values_ema.detach())
critic_loss = critic_loss_lambda + critic_loss_ema
```

**説明:**
- λ-return で価値関数を学習
- EMA Critic で過学習を抑制

**学習対象:** Critic

**重み:** 各項 1.0（合計 2.0）

---

## 3.6 Critic EMA Update（Equation 14）

**定義:**
```
ψ_t+1^EMA = σψ_t^EMA + (1-σ)ψ_t
```
- `σ = 0.98` (`critic_ema_decay`)

**実装:**
```python
with torch.no_grad():
    for ema_param, param in zip(critic_ema.parameters(), critic.parameters()):
        ema_param.data.mul_(critic_ema_decay).add_(
            param.data, alpha=1.0 - critic_ema_decay
        )
```

**説明:**
- Exponential Moving Average で Critic の安定版を保持
- EMA Critic は勾配更新しない（`requires_grad=False`）

**学習対象:** Critic EMA（勾配更新なし、EMA更新のみ）

---

## 3.7 Actor-Critic 学習の流れ

1. **Replay buffer から開始状態をサンプル:** `768個`
2. **Imagination rollout:** World Model で12ステップ先までシミュレート
3. **λ-return 計算:** GAE で advantage 計算
4. **Actor 更新:** Policy gradient + Entropy bonus
5. **Critic 更新:** λ-return + EMA の2項MSE
6. **Critic EMA 更新:** σ=0.98 で緩やかに追従

---

---

# 4️⃣ Curiosity Reward（好奇心報酬）

## 4.1 概要

**学習対象:** なし（World Model の予測誤差を利用するため、追加ネットワークなし）

**学習の有無:** ❌ **なし**（計算のみ）

**目的:**
- 環境報酬が希薄な場合の探索促進
- 他エージェント行動の予測外れを検出（社会的好奇心）

---

## 4.2 CuriosityConfig パラメータ

### 4.2.1 計算型好奇心の重み

| パラメータ名 | 値 | 説明 |
|------------|-----|------|
| `dynamics_curiosity_weight` | 1.0 | 状態予測誤差の重み |
| `reward_curiosity_weight` | 0.5 | 報酬予測誤差の重み |
| `social_curiosity_weight` | 2.0 | チームメイト予測誤差の重み（★重要） |

**補足:**
- `social_curiosity_weight` を重めに設定: 非中央集権型MARLでは他エージェント行動が最大の不確実性源

---

### 4.2.2 好奇心の正規化

| パラメータ名 | 値 | 説明 |
|------------|-----|------|
| `curiosity_normalize` | True | 実行平均で正規化 |
| `curiosity_ema_decay` | 0.99 | 指数移動平均の減衰率 |

---

### 4.2.3 好奇心の減衰

| パラメータ名 | 値 | 説明 |
|------------|-----|------|
| `curiosity_decay_method` | "adaptive" | 減衰方法（"fixed", "count", "adaptive"） |
| `curiosity_initial_weight` | 1.0 | 初期好奇心重み |
| `curiosity_min_weight` | 0.1 | 最小好奇心重み |
| `curiosity_decay_steps` | 10000 | fixed: この歩数で min_weight に到達 |

**減衰方法:**
- **fixed:** 固定スケジュールで線形減衰
- **count:** 訪問カウントベース（`1/√count`）
- **adaptive:** World Model の予測精度に連動

---

### 4.2.4 状態空間の離散化

| パラメータ名 | 値 | 説明 |
|------------|-----|------|
| `state_bin_resolution` | 0.2 | 訪問カウント用の離散化解像度 |

---

### 4.2.5 LLM 意味的好奇心

| パラメータ名 | 値 | 説明 |
|------------|-----|------|
| `use_llm_curiosity` | True | LLM 好奇心を有効化 |
| `llm_api_key` | "" | OpenRouter API Key |
| `llm_base_url` | "https://openrouter.ai/api/v1/chat/completions" | API エンドポイント |
| `llm_model` | "google/gemma-3-4b-it:free" | 使用モデル |
| `llm_temperature` | 0.3 | サンプリング温度 |
| `llm_max_tokens` | 1024 | 最大生成トークン数 |
| `llm_timeout` | 30.0 | API タイムアウト（秒） |
| `llm_max_retries` | 2 | リトライ回数 |
| `llm_eval_every_n_episodes` | 1 | N エピソードごとに評価 |
| `semantic_curiosity_weight` | 0.5 | LLM 好奇心の重み |

---

## 4.3 好奇心報酬の計算

### 4.3.1 Dynamics Curiosity（状態予測誤差）

**計算式:**
```
dynamics_error = ||z_pred - z_actual||²
```

**実装:**
```python
z_pred, _ = world_model.predict_next(z.unsqueeze(1), action_t.unsqueeze(1))
z_actual, _ = world_model.encode(next_obs_t)
dynamics_error = F.mse_loss(z_pred.reshape(-1), z_actual.reshape(-1))
```

**説明:**
- World Model の予測状態と実際の次状態の MSE
- 予測外れ = 新規性 → 好奇心報酬

**学習:** なし（World Model の出力を利用）

**重み:** 1.0 (`dynamics_curiosity_weight`)

---

### 4.3.2 Reward Curiosity（報酬予測誤差）

**計算式:**
```
reward_error = |r_pred - r_actual|
```

**実装:**
```python
reward_logits = world_model.predict_reward(z_pred)
reward_dist = F.softmax(reward_logits, dim=-1)
reward_pred = symexp(two_hot_decode(reward_dist))
reward_error = abs(reward_pred - reward)
```

**説明:**
- 予測報酬と実際の報酬の絶対誤差
- 報酬予測外れ = 環境の未学習領域

**学習:** なし（World Model の出力を利用）

**重み:** 0.5 (`reward_curiosity_weight`)

---

### 4.3.3 Social Curiosity（チームメイト予測誤差）★

**計算式:**
```
social_error = CE(teammate_pred, teammate_actual)
```
- `CE`: Cross-entropy

**実装:**
```python
teammate_logits = world_model.predict_teammates(z, self.agent_idx)

social_error = 0.0
for other_agent_idx, logits in teammate_logits.items():
    actual_action = other_actions_by_idx.get(other_agent_idx)
    if actual_action is not None:
        target = torch.LongTensor([actual_action]).to(device)
        ce = F.cross_entropy(logits, target)
        social_error += ce
social_error /= count
```

**説明:**
- TeammatePredictor の予測と実際の他エージェント行動の cross-entropy
- 他エージェントの予測外れ = 社会的新規性

**学習:** なし（World Model の出力を利用）

**重み:** 2.0 (`social_curiosity_weight`)

**特記事項:**
- **非中央集権型MARLの核心:** 他エージェントの行動が環境の主要な不確実性源
- Social Curiosity を重視することで、協調/競争パターンの探索を促進

---

### 4.3.4 総好奇心報酬

**計算式:**
```
total_curiosity = (w_dyn * dynamics_norm + w_rew * reward_norm + w_soc * social_norm) 
                  * current_weight * visit_bonus
```

**実装:**
```python
total = (
    config.dynamics_curiosity_weight * dynamics_norm +
    config.reward_curiosity_weight * reward_norm +
    config.social_curiosity_weight * social_norm
) * self._current_weight * visit_bonus
```

**説明:**
- 3種の好奇心を重み付き合算
- `current_weight`: 減衰方法に応じて変動
- `visit_bonus`: 訪問カウントベースの減衰（count method のみ）

---

## 4.4 LLM 意味的好奇心

**目的:** 計算型好奇心（量的）を意味的（質的）に補完

**プロセス:**
1. エピソード終了時にトラジェクトリを LLM に送信
2. LLM が以下を評価:
   - `novelty_score`: 全体的新規性（0.0-1.0）
   - `social_novelty`: 社会的新規性
   - `spatial_novelty`: 空間的新規性
   - `strategic_novelty`: 戦略的新規性
   - `exploration_phase`: "explore" | "exploit" | "transition"
3. 評価結果を次エピソードの好奇心重みに反映

**学習:** なし（LLM は外部API）

**重み:** 0.5 (`semantic_curiosity_weight`)

---

---

# 5️⃣ γ-Progress Curiosity（Kim et al. 2020）

## 5.1 概要

**学習対象:** なし（World Model の学習進捗を測定するのみ）

**学習の有無:** ❌ **なし**（計算のみ）

**目的:**
- 標準的好奇心（ICM, RNDなど）の「白色ノイズ問題」を克服
- 学習可能な複雑性（learnable complexity）にフォーカス

**アイデア:**
- 予測誤差ではなく、**学習進捗**（予測精度の向上）を報酬化
- ランダムノイズ: 予測誤差は大きいが学習進捗はゼロ → 報酬なし
- 学習可能な環境: 予測精度が向上 → 報酬あり

---

## 5.2 パラメータ（MATWMConfig）

| パラメータ名 | 値 | 説明 | 論文対応 |
|------------|-----|------|---------|
| `use_gamma_progress` | False | γ-Progress を有効化（デフォルト: 無効） | - |
| `gamma_progress` | 0.9995 | World Model EMA 減衰率（θ_old更新用） | Kim et al. Eq.11 |
| `gamma_progress_weight` | 1.0 | γ-Progress 内発的報酬の重み | - |
| `gamma_progress_normalize` | True | γ-Progress 報酬を正規化 | - |

**補足:**
- Ablation study 用: `use_gamma_progress=True` で γ-Progress のみを有効化可能

---

## 5.3 損失計算（なし、報酬計算のみ）

### 5.3.1 γ-Progress 報酬（Kim et al. Equation 11）

**定義:**
```
r_progress = L(θ_old, x) - L(θ_new, x)
```
- `θ_new`: 現在の World Model パラメータ
- `θ_old`: EMA World Model パラメータ（θ_old ← γ·θ_old + (1-γ)·θ_new）
- `L(θ, x)`: 経験 x に対する World Model 予測損失

**実装:**
```python
with torch.no_grad():
    # Current World Model loss
    loss_new = self._compute_world_model_loss(
        self.world_model, obs_t, action_t, next_obs_t, reward_t, done_t
    )
    
    # EMA World Model loss
    loss_old = self._compute_world_model_loss(
        self.world_model_ema, obs_t, action_t, next_obs_t, reward_t, done_t
    )
    
    # γ-Progress reward
    progress_reward = (loss_old - loss_new).item()
```

**説明:**
- 正: 現在のモデルが EMA より良い → 学習進捗あり → 報酬
- ゼロ/負: 学習進捗なし → 報酬なし

**学習:** なし（World Model の予測損失を比較するのみ）

**重み:** 1.0 (`gamma_progress_weight`)

---

### 5.3.2 World Model Loss for γ-Progress

γ-Progress 計算に使用する World Model 損失は以下の合成:

```
L_progress = L_rew + L_con + L_team + β₁L_dyn
```

**実装:**
```python
total_loss = (
    reward_loss +
    continuation_loss +
    teammate_loss +
    config.kl_weight * dynamics_loss
)
```

**説明:**
- `L_rec` は計算コストが高いため除外
- Dynamics, Reward, Continuation, Teammate の損失で十分な進捗シグナル

**学習:** なし（損失計算のみ、勾配更新なし）

---

## 5.4 World Model EMA Update

**定義:**
```
θ_old ← γ·θ_old + (1-γ)·θ_new
```
- `γ = 0.9995` (`gamma_progress`)

**実装:**
```python
with torch.no_grad():
    for param_ema, param_new in zip(
        world_model_ema.parameters(),
        world_model.parameters()
    ):
        param_ema.data.mul_(gamma).add_(
            param_new.data, alpha=1.0 - gamma
        )
```

**説明:**
- World Model の学習後、EMA パラメータを緩やかに更新
- `γ=0.9995` → 約2000ステップの移動平均

**学習:** なし（EMA更新のみ、勾配更新なし）

---

## 5.5 γ-Progress の利点

| 項目 | 標準的好奇心（ICM, RND等） | γ-Progress |
|-----|------------------------|-----------|
| **白色ノイズ問題** | 高報酬（予測誤差大） | ゼロ報酬（学習進捗なし） |
| **学習可能な環境** | 高報酬（予測誤差大、初期のみ） | 高報酬（学習進捗あり、継続的） |
| **自然な減衰** | 手動で減衰スケジュール必要 | World Model 収束で自動減衰 |
| **追加ネットワーク** | 必要（RND: 2つのネットワーク） | 不要（EMA のみ） |

---

---

# 6️⃣ 損失関数と学習のまとめ

## 6.1 コンポーネント別学習設定

| コンポーネント | 学習 | 最適化 | 学習率 | Grad Clip | 損失関数 |
|------------|------|--------|--------|-----------|---------|
| **Encoder** | ✅ | Adam | 3e-5 | 1000.0 | `L_rec + L_rep` |
| **Decoder** | ✅ | Adam | 3e-5 | 1000.0 | `L_rec` |
| **DynamicsModel** | ✅ | Adam | 3e-5 | 1000.0 | `β₁L_dyn` |
| **RewardPredictor** | ✅ | Adam | 3e-5 | 1000.0 | `L_rew` |
| **ContinuationPredictor** | ✅ | Adam | 3e-5 | 1000.0 | `L_con` |
| **TeammatePredictor** | ✅ | Adam | 3e-5 | 1000.0 | `L_team` |
| **Actor** | ✅ | Adam | 3e-4 | 10.0 | Policy Gradient + Entropy |
| **Critic** | ✅ | Adam | 3e-4 | 10.0 | MSE(λ-return) + MSE(EMA) |
| **Critic EMA** | ❌ | EMA | - | - | - |
| **World Model EMA** | ❌ | EMA | - | - | - |
| **CuriosityReward** | ❌ | - | - | - | - |
| **GammaProgressReward** | ❌ | - | - | - | - |

---

## 6.2 論文との対応（完全版）

| 論文の記号/用語 | コード変数名 | 値/説明 | 論文箇所 |
|--------------|------------|---------|---------|
| **World Model** ||||
| `φ` | `world_model.parameters()` | World Model 全パラメータ | Eq.3 |
| `L(φ)` | `total_loss` | World Model 総損失 | Eq.3 |
| `L_rec` | `recon_loss` | Reconstruction loss (MSE) | Eq.4 |
| `L_rew` | `reward_loss` | Reward loss (Symlog two-hot) | Eq.5 |
| `L_con` | `cont_loss` | Continuation loss (BCE) | Eq.6 |
| `L_team` | `teammate_loss` | Teammate loss (CE, stop-grad) | Eq.8 |
| `L_dyn` | `dynamics_loss` | Dynamics loss (KL, sg(target)) | Eq.9a |
| `L_rep` | `representation_loss` | Representation loss (KL, sg(pred)) | Eq.9b |
| `β₁` | `config.kl_weight` | 0.5 | Table C.6 |
| `β₂` | `config.representation_weight` | 0.1 | Table C.6 |
| `sg(·)` | `.detach()` | Stop-gradient | L140, Eq.9a/b |
| **Actor-Critic** ||||
| `θ` | `actor.parameters()` | Actor パラメータ | Eq.11 |
| `ψ` | `critic.parameters()` | Critic パラメータ | Eq.11 |
| `L(θ)` | `actor_loss` | Actor loss | Eq.11 |
| `L(ψ)` | `critic_loss` | Critic loss | Eq.11 |
| `G_t^λ` | `lambda_returns` | λ-return (GAE) | Eq.12 |
| `λ` | `config.lambda_gae` | 0.95 | Eq.12 |
| `S` | `S` | Percentile normalization factor | Eq.13 |
| `η` | `config.entropy_coef` | 0.01 | Eq.11 |
| `H(π)` | `entropies` | Entropy | Eq.11 |
| `V_EMA` | `critic_ema` | EMA Critic | Eq.14 |
| `σ` | `config.critic_ema_decay` | 0.98 | Eq.14 |
| **γ-Progress** ||||
| `θ_new` | `world_model` | Current World Model | Kim Eq.11 |
| `θ_old` | `world_model_ema` | EMA World Model | Kim Eq.11 |
| `γ` | `config.gamma_progress` | 0.9995 | Kim Eq.11 |
| `r_progress` | `progress_reward` | Learning progress reward | Kim Eq.11 |

---

## 📚 参考文献

1. **MATWM:** Deihim et al. (2025). "Transformer World Model for Sample Efficient Multi-Agent Reinforcement Learning". arXiv:2506.18537
   - Section 3.1: World Model Loss (Equations 3-9)
   - Section 3.2: Training Structure
   - Equations 11-14: Actor-Critic Loss
   - Table C.6: Hyperparameter Settings
   - Line 140: Teammate predictor stop-gradient

2. **γ-Progress:** Kim et al. (2020). "Active World Model Learning with Progress Curiosity"
   - Equation 11: γ-Progress intrinsic reward
   - Learning progress-based curiosity

3. **DreamerV3:** Hafner et al. (2023). "Mastering Diverse Domains through World Models"
   - GAE implementation
   - Symlog transformation
   - Two-hot encoding

---

## 更新日時

2026-02-11

---

## 付録: ハイパーパラメータ一覧（クイックリファレンス）

### 環境
- `max_cycles`: 25
- `num_agents`: 4
- `action_dim`: 5
- `max_obs_dim`: 16

### World Model
- `latent_dim`: 32
- `num_classes`: 32
- `hidden_dim`: 512
- `num_layers`: 2
- `num_heads`: 8
- `encoder_hidden_layers`: 3

### 学習率・最適化
- `wm_learning_rate`: 3e-5
- `agent_learning_rate`: 3e-4
- `gradient_clip_wm`: 1000.0
- `gradient_clip_agent`: 10.0

### 強化学習
- `gamma`: 0.99
- `lambda_gae`: 0.95
- `entropy_coef`: 0.01
- `critic_ema_decay`: 0.98

### Batch・Rollout
- `wm_batch_size`: 16
- `wm_batch_length`: 64
- `agent_batch_size`: 768
- `imagination_horizon`: 12

### 損失重み
- `kl_weight`: 0.5
- `representation_weight`: 0.1
- `free_nats`: 1.0

### γ-Progress
- `use_gamma_progress`: False
- `gamma_progress`: 0.9995
- `gamma_progress_weight`: 1.0

### 好奇心
- `dynamics_curiosity_weight`: 1.0
- `reward_curiosity_weight`: 0.5
- `social_curiosity_weight`: 2.0
- `curiosity_decay_method`: "adaptive"

### Replay Buffer
- `buffer_size`: 50000
- `priority_decay`: 0.9998

### 訓練
- `total_steps`: 50000
- `train_wm_every`: 1
- `train_agent_every`: 1
