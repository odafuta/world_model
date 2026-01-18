# 最終課題: 社会的世界モデル プロジェクト構造

## プロジェクト概要

- **テーマ**: 「他のエージェントの行動」を予測する社会的世界モデル
- **環境**: PettingZoo の simple_tag_v3
- **期間**: 2026年1月14日 ～ 2026年2月上旬（約3週間）
- **ベースモデル**: MATWM (Multi-Agent Transformer World Model)

---

## タスク詳細: Simple Tag

### 環境仕様

- **観測空間**: 
  - adversary: 16次元ベクトル (self_vel, self_pos, landmarks, other agents)
  - agent (good): 14次元ベクトル (self_vel, self_pos, landmarks, other agents)
- **行動空間**: 0～4の離散的な整数値
  - `0`: no_action
  - `1`: move_left
  - `2`: move_right
  - `3`: move_down
  - `4`: move_up
- **エージェント数**: 4
  - 3 adversaries (red) - 遅いが、goodエージェントを捕まえると報酬+10
  - 1 good agent (green) - 速いが、adversariesに捕まると報酬-10
- **最大エピソード長**: 25ステップ (デフォルト)
- **グローバル状態**: 62次元ベクトル

### 評価指標

- 各エージェントの累積報酬
- 協調/競争の成功率（good agentの生存率、adversariesの捕獲成功率）
- サンプル効率（目標: 100K環境ステップ以内で良好な性能）

---

## プロジェクト&ファイル構造

```
最終課題/
│
├── PROJECT_STRUCTURE.md                      # 本ファイル（プロジェクト構造説明）
├── README.md                                 # プロジェクト概要・選定理由
├── simple_tag.md                             # Simple Tag環境の詳細仕様
│
├── 2026_SocialWorldModel_simple_tag_Baseline.ipynb  # ★メインNotebook★（旧版）
├── 2026_MATWM_simple_tag_Implementation.ipynb       # ★MATWMフル実装版★（新規作成）
│   │
│   └── 主要セクション:
│       ├── 1. セットアップ・環境確認
│       ├── 2. データ収集（Replay Buffer）
│       ├── 3. 世界モデルの実装
│       │   ├── Encoder/Decoder (Vector → Categorical Latent)
│       │   ├── Dynamics Model (Transformer-based)
│       │   ├── Reward Predictor
│       │   ├── Continuation Predictor
│       │   └── Teammate Predictor ★社会的世界モデルのコア★
│       ├── 4. 世界モデルの学習（Prioritized Replay）
│       ├── 5. エージェントの実装
│       │   ├── Actor Network
│       │   ├── Critic Network (Semi-centralized)
│       │   └── Imagination-based Training
│       ├── 6. エージェントの学習
│       ├── 7. 評価・可視化
│       └── 8. 参考文献
│
├── data/                                      # データディレクトリ
│   ├── replay_buffers/                        # Replay Buffer保存先
│   │   ├── adversary_0/
│   │   ├── adversary_1/
│   │   ├── adversary_2/
│   │   └── agent_0/
│   └── episodes/                              # エピソード記録
│       ├── episode_0000.npz
│       ├── episode_0001.npz
│       └── ...
│
├── results/                                   # 学習結果・モデル保存先
│   ├── world_model/                           # 世界モデルのチェックポイント
│   │   └── YYYY_MM_DD_HH_MM_SS/
│   │       ├── checkpoints/
│   │       │   ├── step_0000/
│   │       │   │   ├── encoder_decoder.pt
│   │       │   │   ├── dynamics.pt
│   │       │   │   ├── reward_predictor.pt
│   │       │   │   ├── continuation_predictor.pt
│   │       │   │   ├── teammate_predictor.pt
│   │       │   │   └── optimizer.pt
│   │       │   ├── step_10000/
│   │       │   └── ...
│   │       └── visualizations/
│   │           ├── latent_reconstruction_step_0000.png
│   │           └── ...
│   │
│   └── agents/                                # エージェントのチェックポイント
│       └── YYYY_MM_DD_HH_MM_SS/
│           ├── checkpoints/
│           │   ├── step_0000/
│           │   │   ├── actor.pt
│           │   │   ├── critic.pt
│           │   │   └── optimizer.pt
│           │   ├── step_10000/
│           │   └── ...
│           └── logs/
│               └── training_metrics.csv
│
├── scripts/                                   # ユーティリティスクリプト
│   ├── collect_data.py                        # データ収集スクリプト
│   ├── train_world_model.py                   # 世界モデル学習スクリプト（オプション）
│   ├── train_agents.py                        # エージェント学習スクリプト（オプション）
│   └── evaluate.py                            # 評価スクリプト
│
├── 論文/                                      # 参考論文・資料
│   ├── md/
│   │   ├── TransformerWorldModelForSampleEfficientMultiAgentReinforcementLearning.md
│   │   └── ActiveWorldModelLearningWithProgress.md
│   └── *.pdf
│
└── 中間報告/                                  # 中間報告資料
    └── 提出内容.pdf
```

---

## 主要コンポーネント

> 💡 **設計方針**: 各コンポーネントは MATWM 論文 (arXiv:2506.18537) の実証結果に基づき選定しています。

### 1. 世界モデル (World Model)

#### Encoder/Decoder (Categorical VAE)
- **役割**: ベクトル観測を潜在空間（Categorical VAE）にエンコード・デコード
- **アーキテクチャ**: MLP-based Encoder/Decoder
- **潜在表現**: 32 categorical variables × 32 classes = 1024次元離散空間

**📥 入力 (Encoder)**:
- 観測ベクトル `obs`: `[batch, obs_dim]`
  - adversary: `[batch, 16]` (self_vel[2] + self_pos[2] + landmarks[6] + other_agents[6])
  - good agent: `[batch, 14]` (self_vel[2] + self_pos[2] + landmarks[6] + other_agents[4])

**📤 出力 (Encoder)**:
- 潜在分布 `logits`: `[batch, 32, 32]` (32個の categorical 変数、各32クラス)
- サンプル `z`: `[batch, 32]` (32個の one-hot → 32個のクラスインデックス)

**📥 入力 (Decoder)**:
- 潜在状態 `z`: `[batch, 32]` (categorical インデックス)
- または `z_onehot`: `[batch, 32*32=1024]` (one-hot 展開)

**📤 出力 (Decoder)**:
- 再構成観測 `obs_recon`: `[batch, obs_dim]` (元の観測空間に復元)

- **📚 論文根拠**: 
  - MATWM Section 2.1: "discrete latent spaces often outperform continuous ones"
  - Table 1: Categorical VAE を採用（MAMBA と同様）
  - 連続空間より情報圧縮効率が高く、RL タスクで実証済み
- **Simple Tag への適用**: ベクトル観測（14/16次元）を効率的に圧縮し、予測性能を向上

#### Dynamics Model (Transformer)
- **役割**: 過去の潜在状態 $z_t$ と行動 $a_t$ から次の潜在状態 $z_{t+1}$ を予測
- **アーキテクチャ**: Vanilla Transformer with Action Mixer (4層、8ヘッド、512次元)

**📥 入力**:
- 潜在状態系列 `z_seq`: `[batch, seq_len, latent_dim]`
  - `seq_len`: 過去の時系列長（例: 64 steps）
  - `latent_dim`: 32 (categorical インデックス) or 1024 (one-hot)
- 行動系列 `action_seq`: `[batch, seq_len, 1]`
  - **Action Scaled**: agent ID によってオフセット済み
    - Agent 0: 0-4
    - Agent 1: 5-9
    - Agent 2: 10-14
    - Agent 3: 15-19

**🔄 内部処理**:
1. **Embedding**: `z_seq` と `action_seq` を埋め込み → `[batch, seq_len, 512]`
2. **Positional Encoding**: 時系列位置情報を付加
3. **Transformer Layers** (×4): 
   - Multi-Head Self-Attention (8 heads)
   - Feed-Forward Network
   - Layer Normalization & Residual Connection
4. **Output Projection**: `[batch, seq_len, 32*32]` (次状態の logits)

**📤 出力**:
- 次潜在状態分布 `logits_next`: `[batch, seq_len, 32, 32]`
  - 各時刻の次状態を予測（32 categorical × 32 classes）
- サンプル `z_next`: `[batch, seq_len, 32]`

- **📚 論文根拠**: 
  - MATWM Section 2: "transformers typically outperform RNNs due to long-range dependencies"
  - STORM ベース、RNN/GRU を使う MAMBA/MBVD を上回る性能
  - Table C.6: ハイパーパラメータ設定に準拠
- **利点**: 長期依存関係のモデリング、並列計算可能

#### Reward Predictor (Two-hot Symlog)
- **役割**: 潜在状態から報酬を予測
- **アーキテクチャ**: MLP (256次元hidden、2層)

**📥 入力**:
- 潜在状態 `z`: `[batch, latent_dim]`
  - `latent_dim`: 32 (categorical) or 1024 (one-hot)
- または時系列: `z_seq`: `[batch, seq_len, latent_dim]`

**🔄 内部処理**:
1. **MLP**: `z` → hidden(256) → output(255)
2. **Symlog 変換**: 報酬 `r` を symlog 空間に変換
   - `symlog(r) = sign(r) * log(1 + |r|)`
   - 範囲: `[-20, +20]` → 255 bins に離散化
3. **Two-hot Encoding**: 
   - 2つの隣接ビンに確率を分配
   - 例: symlog(r) = 3.7 → bin[3] と bin[4] に重み付き分配

**📤 出力**:
- 報酬分布 `logits`: `[batch, 255]` (255 bins over symlog space)
- 予測報酬値 `r_pred`: `[batch, 1]`
  - `r_pred = symexp(weighted_sum_over_bins)`
  - `symexp(x) = sign(x) * (exp(|x|) - 1)` (逆変換)

- **📚 論文根拠**: 
  - MATWM Section 3, Equation 5: symlog two-hot loss 採用
  - Dreamer V3 から継承した手法で、極端な報酬値にロバスト
- **Simple Tag への効果**: ±10の報酬を滑らかに学習可能、外れ値に強い

#### Continuation Predictor
- **役割**: エピソード終了フラグを予測（continues = 1 - done）
- **アーキテクチャ**: MLP (256次元hidden、2層)

**📥 入力**:
- 潜在状態 `z`: `[batch, latent_dim]`
  - `latent_dim`: 32 (categorical) or 1024 (one-hot)
- または時系列: `z_seq`: `[batch, seq_len, latent_dim]`

**🔄 内部処理**:
1. **MLP**: `z` → hidden(256) → output(1)
2. **Sigmoid**: logit → 確率 `[0, 1]`

**📤 出力**:
- 継続確率 `continues`: `[batch, 1]` or `[batch, seq_len, 1]`
  - `continues = 1`: エピソード継続
  - `continues = 0`: エピソード終了 (done)
- Bernoulli 分布のパラメータ

- **📚 論文根拠**: 
  - MATWM Equation 6: Binary cross-entropy loss
  - Imagination rollout の品質向上に不可欠
- **Simple Tag での重要性**: 
  - Good agent が捕まったタイミングを正確に予測
  - 想像軌道で適切に終了判定

#### Teammate Predictor ★社会的世界モデルのコア★
- **役割**: 他のエージェントの行動を予測
- **アーキテクチャ**: エージェントごとに独立した MLP (256次元hidden、2層)

**📥 入力**:
- **Focal agent** の潜在状態 `z_focal`: `[batch, latent_dim]`
  - `latent_dim`: 32 (categorical) or 1024 (one-hot)
  - 自身の観測から得られた潜在表現
- または時系列: `z_seq`: `[batch, seq_len, latent_dim]`

**🔄 内部処理**:
1. **Agent-specific MLP** (各他エージェントごと):
   - Agent 0 を予測: `MLP_0(z_focal)` → logits `[5]`
   - Agent 1 を予測: `MLP_1(z_focal)` → logits `[5]`
   - Agent 2 を予測: `MLP_2(z_focal)` → logits `[5]`
   - (Simple Tag: 3体の他エージェント)
2. **Softmax**: logits → 行動確率分布

**📤 出力**:
- 他エージェントの行動分布（各エージェント独立）:
  - Agent 0: `logits_0`: `[batch, 5]` → `probs_0`: `[batch, 5]`
  - Agent 1: `logits_1`: `[batch, 5]` → `probs_1`: `[batch, 5]`
  - Agent 2: `logits_2`: `[batch, 5]` → `probs_2`: `[batch, 5]`
- **Unscaled action space**: 0-4 (original action space)
  - Dynamics Model への入力時に再度 Action Scaling を適用

**🔄 Imagination での使用**:
```python
# Rollout 時の他エージェント行動サンプリング
z_t = current_latent_state  # [1, latent_dim]
teammate_actions = []
for agent_id in other_agents:
    action_probs = teammate_predictor[agent_id](z_t)  # [1, 5]
    action = sample(action_probs)  # 確率的サンプリング
    scaled_action = scale_action(action, agent_id)  # スケーリング
    teammate_actions.append(scaled_action)

# 次状態予測
z_next = dynamics_model(z_t, focal_action, teammate_actions)
```

- **📚 論文根拠**: 
  - MATWM Section 3.1, Equation 8: teammate predictor の定義
  - Abstract: "lightweight and effective teammate predictor module"
  - **Ablation Study (Table 5)**: Teammate Predictor なしでは性能が劇的に低下
    - 8m: 67.0 → 0.0 (完全崩壊)
    - so_many_baneling: 74.0 → 0.0 (完全崩壊)
  - Section 4.3: 協調タスクで "substantial gains"
- **効果**: 
  - **非定常性の軽減**: 他エージェントの方策変化を追跡
  - **協調行動**: Adversaries が互いの動きを予測して連携
  - **競争行動**: Good agent が Adversaries の追跡パターンを予測して逃走
  - **Imagination rollout**: 他エージェント行動をシミュレートして学習
- **Simple Tag での重要性**: 
  - **Adversaries**: 他2体の動きを予測して包囲戦術を計画
  - **Good Agent**: 3体の追跡パターンを予測して最適逃走ルート選択

### 2. エージェント (Agent)

#### Actor Network
- **役割**: 方策 $\pi(a|z)$ を学習
- **アーキテクチャ**: MLP (256次元hidden、2層)

**📥 入力**:
- 潜在状態 `z`: `[batch, latent_dim]`
  - 実環境または想像環境から得られた潜在状態
  - `latent_dim`: 32 (categorical) or 1024 (one-hot)
- または想像軌道: `z_imagination`: `[batch, horizon, latent_dim]`
  - `horizon`: 15 (想像する未来のステップ数)

**🔄 内部処理**:
1. **MLP**: `z` → hidden(256) → hidden(256) → output(5)
2. **Softmax**: logits → 行動確率分布

**📤 出力**:
- 行動分布 `logits`: `[batch, 5]` or `[batch, horizon, 5]`
  - 5つの行動クラス: {0: no_action, 1: left, 2: right, 3: down, 4: up}
- 行動確率 `probs`: `[batch, 5]`
- サンプルされた行動 `action`: `[batch, 1]` (categorical sampling)
- エントロピー `entropy`: `[batch, 1]` (探索促進用)

**損失関数**:
```
L_actor = -𝔼[advantages * log_prob(action)] - β * entropy
```
- `advantages`: Critic から計算された利得
- `β`: entropy coefficient (0.001)

- **📚 論文根拠**: 
  - MATWM Equation 10, 11: Actor の定義と損失関数
  - Entropy regularization で探索を促進

#### Critic Network (Semi-centralized)
- **役割**: 価値関数 $V(z)$ を学習
- **アーキテクチャ**: MLP (256次元hidden、2層)

**📥 入力**:
- **Primary**: Focal agent の潜在状態 `z`: `[batch, latent_dim]`
- **Optional (Semi-centralized)**: Teammate Predictor からの他エージェント行動予測
  - 訓練時: 想像上の他エージェント行動情報を暗黙的に利用
  - 実行時: `z` のみで価値推定（Decentralized Execution）

**🔄 内部処理**:
1. **MLP**: `z` → hidden(256) → hidden(256) → output(1)
2. 他エージェントの影響は `z` 自体に含まれる（観測に他エージェント位置が含まれるため）

**📤 出力**:
- 状態価値 `V(z)`: `[batch, 1]` or `[batch, horizon, 1]`
  - 現在状態から得られる期待累積報酬
- 想像軌道の場合: `V_imagination`: `[batch, horizon, 1]`

**損失関数**:
```
L_critic = 𝔼[(V(z) - target_value)²]
target_value = r + γ * continues * V(z_next)  (TD target)
または
target_value = λ-return (GAE による advantage 計算)
```

**GAE (Generalized Advantage Estimation)**:
```
A_t = Σ_{l=0}^{horizon} (γλ)^l * δ_{t+l}
δ_t = r_t + γ * continues_t * V(z_{t+1}) - V(z_t)
```
- `γ`: discount factor (0.99)
- `λ`: GAE lambda (0.95)

- **📚 論文根拠**: 
  - MATWM Table 1: "Semi-centralized" critic を採用
  - Section 3.2: "semi-centralized critic that does not require having direct access to non-focal agent information"
  - Equation 12: λ-return による advantage 計算
  - 他エージェントの**想像上の行動**を考慮（直接的な情報アクセスは不要）
- **利点**: 
  - **Centralized (MAMBA)** のようにスケーラビリティ問題なし
  - **Decentralized** より協調性が高い
  - **CTDE** (Centralized Training, Decentralized Execution) パラダイムに準拠
  - エージェント数が増えても線形にスケール

#### Imagination-based Training
- **プロセス**: 実環境の経験から想像軌道を生成し、Actor/Critic を学習

**📥 入力**:
- Replay Buffer からサンプルされた実体験:
  - 観測 `obs_real`: `[batch, obs_dim]`
  - 行動 `action_real`: `[batch, 1]`
  - 報酬 `reward_real`: `[batch, 1]`

**🔄 Imagination Rollout プロセス**:

```python
# Step 1: 実観測を潜在状態に変換
z_0 = encoder(obs_real)  # [batch, latent_dim]

# Step 2: 想像軌道を生成 (horizon=15 steps)
z_imagination = [z_0]
actions_imagination = []
rewards_imagination = []
continues_imagination = []

for t in range(horizon):  # t = 0, 1, ..., 14
    # 2.1: Actor で行動を決定
    action_t = actor(z_t)  # [batch, 1], focal agent の行動
    
    # 2.2: Teammate Predictor で他エージェント行動を予測
    teammate_actions_t = []
    for other_agent in other_agents:
        teammate_action = teammate_predictor[other_agent](z_t)
        teammate_actions_t.append(teammate_action)
    
    # 2.3: Dynamics Model で次状態を予測
    all_actions = [action_t] + teammate_actions_t  # [batch, n_agents]
    z_next = dynamics_model(z_t, all_actions)  # [batch, latent_dim]
    
    # 2.4: Reward Predictor で報酬を予測
    r_t = reward_predictor(z_next)  # [batch, 1]
    
    # 2.5: Continuation Predictor で終了判定
    continues_t = continuation_predictor(z_next)  # [batch, 1]
    
    # 記録
    z_imagination.append(z_next)
    actions_imagination.append(action_t)
    rewards_imagination.append(r_t)
    continues_imagination.append(continues_t)
    
    z_t = z_next  # 次ステップへ

# Step 3: 想像軌道から価値とAdvantageを計算
V_imagination = critic(z_imagination)  # [batch, horizon+1, 1]
advantages = compute_gae(
    rewards_imagination,
    V_imagination,
    continues_imagination,
    γ=0.99,
    λ=0.95
)  # [batch, horizon, 1]
```

**📤 出力 (学習に使用)**:
- 想像軌道の潜在状態: `z_imagination`: `[batch, horizon, latent_dim]`
- 想像軌道の行動: `actions_imagination`: `[batch, horizon, 1]`
- 想像軌道の報酬: `rewards_imagination`: `[batch, horizon, 1]`
- 想像軌道の継続フラグ: `continues_imagination`: `[batch, horizon, 1]`
- 想像軌道の価値: `V_imagination`: `[batch, horizon, 1]`
- Advantages: `advantages`: `[batch, horizon, 1]`

**学習更新**:
```python
# Actor 更新
actor_loss = -𝔼[advantages * log_prob(actions_imagination)] - β * entropy

# Critic 更新
critic_loss = 𝔼[(V_imagination - target_values)²]
```

- **📚 論文根拠**: 
  - MATWM Abstract: "imagine future trajectories" で学習
  - Section 3.1: "agents learn entirely from imagination"
  - Table C.6: Imagination horizon = 16 (我々は15を採用)
  - Equation 12: λ-return (GAE) による advantage 計算
- **サンプル効率**: 
  - 実環境**1ステップ** → 想像**15ステップ** = **15倍の学習データ**
  - 目標: **50K-100K steps で収束**（従来手法の1/10以下）
- **Simple Tag での効果**: 
  - 少ないエピソードで協調・競争戦略を学習
  - 危険な状況（Good agent が捕まる）を想像上で学習可能
  - Teammate Predictor により他エージェントの動きを考慮した計画

### 3. 訓練戦略 (Training Strategy)

#### Prioritized Replay Buffer
- **役割**: 最近の経験を優先的にサンプリング
- **実装**: Exponential decay (0.995 per step)

**📥 入力 (経験の保存)**:
- 観測 `obs`: `[obs_dim]` (14 or 16)
- 行動 `action`: `[1]` (0-4, unscaled)
- 報酬 `reward`: `[1]` (±10 or 0)
- 次観測 `next_obs`: `[obs_dim]`
- 終了フラグ `done`: `[1]` (0 or 1)
- タイムステップ `t`: エピソード内の時刻

**🔄 優先度計算**:
```python
# 保存時に優先度を付与
priority = decay_rate ** (current_step - t)
# decay_rate = 0.995
# current_step: 現在の全体ステップ数
# t: 経験が収集された時のステップ数

# 例:
# t=0 (古い経験), current_step=1000 → priority = 0.995^1000 ≈ 0.0067
# t=999 (新しい経験), current_step=1000 → priority = 0.995^1 ≈ 0.995
```

**📤 出力 (サンプリング)**:
- バッチサンプル (世界モデル学習用):
  - `obs_batch`: `[batch_size, seq_len, obs_dim]`
  - `action_batch`: `[batch_size, seq_len, 1]`
  - `reward_batch`: `[batch_size, seq_len, 1]`
  - `continues_batch`: `[batch_size, seq_len, 1]`
  - `batch_size`: 16
  - `seq_len`: 64 (時系列の長さ)

- バッチサンプル (エージェント学習用):
  - `obs_batch`: `[batch_size, obs_dim]` (単一ステップ)
  - `batch_size`: 16
  - Imagination の開始点として使用

**サンプリング確率**:
```python
p_i = priority_i / Σ_j priority_j
# 最近の経験ほど高確率でサンプルされる
```

- **📚 論文根拠**: 
  - MATWM Section 3: "prioritized replay mechanism that trains the world model on recent experiences"
  - Section 3.2: 非定常性（他エージェントの方策変化）への対応
  - Table C.6: Replay sampling priority decay = 0.9998 (我々は0.995を採用)
  - **Ablation Study (Table 5)**: PER なしでは性能低下
    - 8m: 65.0 → 52.0
    - Pistonball: 92.6 → 85.1
- **効果**: 
  - 古い経験（outdated behaviors）の影響を軽減
  - 方策が進化しても world model が追従
  - 学習の安定性向上
  - 非定常性（他エージェントの方策変化）への適応

#### Action Scaling
- **役割**: エージェントごとに行動空間をオフセットして、World Model がエージェントを識別

**📥 入力**:
- Original action `a`: `[1]` (0-4)
  - 0: no_action
  - 1: move_left
  - 2: move_right
  - 3: move_down
  - 4: move_up
- Agent ID `agent_id`: `[1]` (0, 1, 2, 3)

**🔄 スケーリング処理**:
```python
def scale_action(action, agent_id, action_space_size=5):
    """
    行動にエージェントIDベースのオフセットを追加
    """
    scaled_action = action + agent_id * action_space_size
    return scaled_action

def unscale_action(scaled_action, agent_id, action_space_size=5):
    """
    スケーリングされた行動を元に戻す
    """
    action = scaled_action - agent_id * action_space_size
    return action
```

**📤 出力**:
- Scaled action `a_scaled`:
  - **Agent 0**: 0-4 (変化なし)
  - **Agent 1**: 5-9
  - **Agent 2**: 10-14
  - **Agent 3**: 15-19

**使用フロー**:
```python
# 1. 環境から行動収集時
action_env = agent.select_action(obs)  # 0-4
action_scaled = scale_action(action_env, agent_id)  # 0-19

# 2. World Model 学習時
dynamics_input = [z_t, action_scaled]  # スケーリング済み行動を入力

# 3. Teammate Predictor 出力時
teammate_action = teammate_predictor(z_t)  # 0-4 (unscaled)
teammate_action_scaled = scale_action(teammate_action, teammate_id)

# 4. Imagination rollout 時
all_actions_scaled = [
    scale_action(focal_action, focal_id),
    scale_action(teammate_action_0, teammate_0_id),
    scale_action(teammate_action_1, teammate_1_id),
    scale_action(teammate_action_2, teammate_2_id),
]
z_next = dynamics_model(z_t, all_actions_scaled)
```

**行動空間の拡張**:
- Original: 5 actions per agent
- Scaled: 20 actions (5 × 4 agents)
- Dynamics Model の出力層: 20次元

- **📚 論文根拠**: 
  - MATWM Section 3: "action scaling mechanism to encode agent-specific information"
  - World Model が explicit ID や embedding なしでエージェントを識別可能
  - **Ablation Study (Table 5)**: Action Scaling なしでは性能低下（特に画像ベース環境）
    - Pistonball: 92.6 → 88.4
    - Externality Mushrooms: 146.8 → 135.7
- **利点**: 
  - **シンプル**: 追加パラメータ不要
  - **効率的**: 計算コスト増加なし
  - **識別可能**: Shared world model でもエージェント別の行動パターンを学習
  - **Dynamics 学習**: 「Agent 1 が右に移動」と「Agent 2 が右に移動」を区別可能

#### Decentralized Execution
- **方針**: 各エージェントは自身の観測のみで行動決定

**🎓 訓練時 (Centralized Training)**:

**📥 入力**:
- 全エージェントの経験が Replay Buffer に蓄積
- Teammate Predictor が他エージェントの行動を学習
- Imagination rollout で全エージェントの相互作用を考慮

**🔄 処理**:
```python
# 各エージェントが独立に世界モデルを学習
for agent in agents:
    # 自身の観測から潜在状態
    z = encoder(agent.obs)
    
    # Teammate Predictor で他エージェント予測
    teammate_actions = [
        teammate_predictor[other_id](z)
        for other_id in other_agents
    ]
    
    # 全エージェントを考慮した次状態予測
    z_next = dynamics_model(z, agent.action, teammate_actions)
    
    # Actor/Critic 更新（他エージェント情報を暗黙的に利用）
    actor.update(z)
    critic.update(z)  # Semi-centralized
```

**🎮 実行時 (Decentralized Execution)**:

**📥 入力**:
- 自身の観測のみ: `obs_focal`: `[obs_dim]`
- 通信不要（他エージェントの状態・行動に直接アクセスしない）

**📤 出力**:
- 行動 `action`: `[1]` (0-4)

**🔄 処理**:
```python
# 実行時は完全に独立
z = encoder(obs_focal)  # 自身の観測のみ
action = actor(z)  # 自身の方策のみ

# Teammate Predictor は使用しない（実行時）
# 他エージェントの影響は obs に含まれる位置情報で間接的に把握
```

**特徴**:
- ✅ **スケーラブル**: エージェント数 N に対して O(N) の計算量
- ✅ **部分観測に適合**: 各エージェントは自身の観測のみで行動
- ✅ **通信不要**: 分散実行可能
- ✅ **ロバスト**: 一部エージェントの故障に強い

- **📚 論文根拠**: 
  - MATWM Section 3.2: Decentralized world model approach
  - CTDE (Centralized Training, Decentralized Execution) 設計
  - Scalability: エージェント数に対して線形にスケール
- **Simple Tag での適用**: 
  - **訓練**: Teammate Predictor で他エージェントを考慮、Semi-centralized Critic
  - **実行**: 各エージェント独立に行動選択、部分観測のみ使用

---

## 実行フロー

### 1. 環境セットアップ

```bash
pip install torch numpy matplotlib tqdm h5py
pip install pettingzoo[mpe] supersuit
```

### 2. データ収集（Warm-up）

```python
# Notebookまたはスクリプトで実行
# - ランダム方策で1000ステップ分の経験を収集
# - Replay Bufferに保存
```

### 3. 世界モデル学習

```python
# 主なステップ:
# 1. Replay Bufferから64長系列をサンプリング（Prioritized Replay）
# 2. Encoder/Decoder, Dynamics, Reward, Continuation, Teammate Predictorを更新
# 3. 1エポック学習（実環境1ステップごと）
```

### 4. エージェント学習

```python
# 主なステップ:
# 1. Replay Bufferからランダムサンプリング
# 2. 世界モデルで15ステップの想像軌道を生成
# 3. Actor/Criticを想像軌道で更新
# 4. 1エポック学習（実環境1ステップごと）
```

### 5. 評価・可視化

```python
# - 累積報酬の推移
# - 学習曲線
# - Teammate Predictionの精度
# - 潜在空間の可視化
```

---

## 編集可能・不可セクション

### ✅ 編集可能（自由に変更してください）

1. **モデルアーキテクチャ**
   - 潜在次元数、Transformerの層数・ヘッド数
   - Teammate Predictorの設計
   - Actor/Criticのネットワーク構造

2. **学習ハイパーパラメータ**
   - 学習率、バッチサイズ
   - Imagination horizon
   - Prioritized Replayの減衰率

3. **拡張機能**
   - γ-progress好奇心の導入（Active World Model Learning）
   - Communication moduleの追加
   - Attention mechanismの強化

### ❌ 編集不可（基本的に維持）

1. **環境の設定**
   - simple_tag_v3の基本パラメータ
   - 観測・行動空間の定義

2. **評価指標**
   - 累積報酬による性能評価

---

## 環境要件

- **Python**: 3.9+
- **主要ライブラリ**:
  - PyTorch 2.0+
  - PettingZoo[mpe]
  - NumPy
  - Matplotlib
  - tqdm
  - h5py (オプション: データ保存)

---

## 性能向上のアイデア

### 世界モデルの改善

- [ ] **Teammate Predictor の精度向上**
  - Attention 機構の導入
  - 複数ステップ先の行動予測
  - 📚 参考: MATWM Section 4.3 で示唆されている改善方向

- [ ] **Theory of Mind 要素の組み込み**
  - 他エージェントの意図・信念の推定
  - Recursive reasoning (「相手は自分をどう予測しているか」)
  - 📚 参考: AWML Section 7 の Theory of Mind discussion

- [ ] **Communication Module の追加**
  - メッセージ passing 機構
  - Attention-based communication
  - Differentiable communication

### エージェントの改善

- [ ] **γ-Progress Curiosity の導入** ⭐ 次の重要拡張
  - Active World Model Learning (AWML 論文) の手法
  - 📚 論文根拠: AWML Equation 10, 11, 12
  - **効果** (AWML Table 1):
    - Mixture World: 7.83倍の性能向上 (vs Random)
    - Noise World: 13.79倍の性能向上
    - White Noise Problem の解決
  - **実装**:
    ```python
    # θ_old = (1-γ) Σ γ^(k-1-i) θ_i  (exponential mixture)
    # θ_old ← γ θ_old + (1-γ) θ_new
    # r = L(θ_old, x, a) - L(θ_new, x, a)
    ```
  - **Simple Tag での期待効果**:
    - 学習可能なパターン（協調追跡、効率的逃走）に注力
    - ノイズ的な行動を無視
    - 探索効率の大幅向上

- [ ] **階層的プランニング**
  - High-level goal selection
  - Low-level action execution
  - Feudal RL との統合

- [ ] **より洗練された Critic の設計**
  - Value decomposition
  - Distributional RL

- [ ] **Adversarial Training**
  - Self-play による継続学習
  - Population-based training

### サンプル効率の改善

- [ ] **Prioritized Replay の重み付け最適化**
  - 現在: Exponential decay (0.995)
  - 検討: 予測誤差ベースの優先度
  - 📚 参考: MATWM Table C.6 では 0.9998 を使用

- [ ] **模倣学習との組み合わせ**
  - Expert demonstrations からの学習
  - Behavior cloning + RL

- [ ] **Self-play による継続学習**
  - Good agent と Adversaries の相互進化
  - Curriculum learning

---

## 参考文献

### 主要論文

1. **MATWM (Multi-Agent Transformer World Model)** 🌟
   - Deihim, A., Alonso, E., & Apostolopoulou, D. (2025)
   - "Transformer World Model for Sample Efficient Multi-Agent Reinforcement Learning"
   - arXiv:2506.18537v1 [cs.LG], 23 Jun 2025
   - **本実装のベース**: 全コンポーネントがこの論文に基づく
   - **主要貢献**:
     - Teammate Predictor による社会的世界モデル
     - 50K steps で最先端性能（SMAC, PettingZoo, Melting Pot）
     - 初の画像ベース対応マルチエージェント世界モデル

2. **Active World Model Learning with Progress Curiosity** 🎯
   - Kim, K., Sano, M., De Freitas, J., Haber, N., & Yamins, D. (2020)
   - arXiv:2007.07853v1 [cs.LG], 15 Jul 2020
   - ICML 2020
   - **今後の拡張として導入予定**: γ-Progress Curiosity
   - **主要貢献**:
     - γ-Progress: 学習可能な dynamics に注意を向ける
     - White Noise Problem の解決
     - Animate attention の自然な獲得
   - **期待効果**: 探索効率の大幅向上（Table 1: 7.83倍 in Mixture World）

### 基盤技術

3. **STORM (Stochastic Transformer World Model)**
   - Zhang, W., et al. (2023)
   - NeurIPS 2023
   - **MATWM の基盤**: 単一エージェント世界モデルの最先端
   - Two-hot symlog, KL balance, free bits などを統合

4. **Dreamer V3**
   - Hafner, D., et al. (2023)
   - "Mastering Diverse Domains through World Models"
   - **貢献**: Two-hot symlog rewards, percentile return normalization
   - MATWM が多数の技術を継承

### 関連手法

5. **MAMBA (Multi-Agent Model-Based RL)**
   - Egorov, V., & Shpilman, A. (2022)
   - Centralized world model approach
   - Categorical VAE 採用

6. **MARIE (Decentralized Transformers with Centralized Aggregation)**
   - Zhang, Y., et al. (2024)
   - arXiv:2406.15836
   - 初の Transformer ベースマルチエージェント世界モデル
   - Perceiver による feature aggregation

7. **PettingZoo**
   - Terry, J. K., et al. (2021)
   - "PettingZoo: Gym for Multi-Agent Reinforcement Learning"
   - Simple Tag 環境を提供

---

## MATWM と AWML の融合戦略

### 現状: MATWM のフル実装 ✅

**実装済みコンポーネント**:
- ✅ Categorical VAE (Encoder/Decoder)
- ✅ Transformer Dynamics Model
- ✅ Two-hot Symlog Reward Predictor
- ✅ Continuation Predictor
- ✅ **Teammate Predictor** (社会的世界モデルの核心)
- ✅ Prioritized Replay Buffer
- ✅ Action Scaling
- ✅ Semi-centralized Critic
- ✅ Imagination-based Training

### 次段階: AWML γ-Progress の統合 🎯

#### 統合方法

**1. Curiosity Reward の定義**:
```python
# AWML Equation 10, 11, 12 に基づく
θ_old = exponential_mixture_of_past_models(γ=0.9)
L_old = world_model_loss(θ_old, experience)
L_new = world_model_loss(θ_current, experience)
r_curiosity = L_old - L_new  # Progress = 予測誤差の改善
```

**2. 報酬の統合**:
```python
r_total = r_extrinsic + λ_curiosity * r_curiosity
# r_extrinsic: Simple Tag の報酬 (±10)
# λ_curiosity: 好奇心の重み (0.1-1.0)
```

**3. 実装上の注意**:
- Teammate Predictor の予測誤差も Progress に含める
- γ: 0.9-0.95 が AWML で推奨
- λ_curiosity: 探索と利用のバランス調整

#### 期待される効果

- **探索効率の向上**: 学習可能なパターン（協調追跡、効率的逃走）に注力
- **White Noise Problem の回避**: ランダムな行動を無視
- **サンプル効率**: 50K steps → 30K steps 以下への短縮を期待
- **Teammate Prediction との相乗効果**: 
  - 他エージェントの学習可能な方策に注意を向ける
  - 予測不可能なランダム行動を無視

#### 実装優先度

1. **Phase 1 (現在)**: MATWM のみで Simple Tag を学習 ✅
2. **Phase 2 (次)**: γ-Progress の統合と ablation study
3. **Phase 3**: Theory of Mind 要素の追加

---

## 論文との対応表

| コンポーネント | MATWM 論文 | AWML 論文 | 実装状況 |
|--------------|-----------|-----------|---------|
| Categorical VAE | Section 2.1, Table 1 | - | ✅ |
| Transformer Dynamics | Section 2 | - | ✅ |
| Reward Predictor | Equation 5 | - | ✅ |
| Continuation | Equation 6 | - | ✅ |
| Teammate Predictor | Section 3.1, Eq 8, Table 5 | - | ✅ |
| Prioritized Replay | Section 3, 3.2, Table 5 | - | ✅ |
| Action Scaling | Section 3, Table 5 | - | ✅ |
| Semi-centralized Critic | Section 3.2, Table 1 | - | ✅ |
| Imagination Training | Abstract, Section 3.1, Eq 12 | - | ✅ |
| γ-Progress Curiosity | - | Eq 10-12, Table 1 | 🔜 Phase 2 |
| Theory of Mind | Section 4.3 (discussion) | Section 7 | 🔜 Phase 3 |

---

## 更新履歴

- 2026-01-14 (初版): プロジェクト構造初版作成
- 2026-01-14 (第2版): MATWM フル実装完了、論文根拠を全コンポーネントに追加
- 2026-01-14 (第3版): AWML γ-Progress 統合戦略を追加、論文対応表を整備

---

**Good Luck! 🏃‍♂️🏃‍♀️🎯**

