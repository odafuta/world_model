# MATWM Implementation Summary

## プロジェクト概要

**社会的世界モデル (Social World Model)** をPettingZooの`simple_tag`環境に実装しました。

- **ベースモデル**: MATWM (Multi-Agent Transformer World Model)
- **論文**: Deihim et al. (2025), "Transformer World Model for Sample Efficient Multi-Agent Reinforcement Learning", arXiv:2506.18537
- **環境**: Simple Tag (predator-prey, 3 adversaries vs 1 good agent)
- **目標**: サンプル効率的なマルチエージェント学習 (100K steps以内)

---

## 実装ファイル

### 1. プロジェクト構造

```
最終課題/
├── PROJECT_STRUCTURE.md                          # プロジェクト構造の詳細説明
├── IMPLEMENTATION_SUMMARY.md                     # 本ファイル
├── README.md                                     # プロジェクト概要・選定理由
├── simple_tag.md                                 # 環境仕様
│
├── matwm_implementation.py                       # World Modelコンポーネント ★
│   ├── MATWMConfig                               # 設定クラス
│   ├── PrioritizedReplayBuffer                   # 優先度付きリプレイバッファ
│   ├── Encoder/Decoder                           # Categorical VAE
│   ├── DynamicsModel                             # Transformer dynamics
│   ├── RewardPredictor                           # 報酬予測
│   ├── ContinuationPredictor                     # 継続予測
│   ├── TeammatePredictor ★                       # 他エージェント行動予測
│   ├── Actor/Critic                              # Actor-Critic networks
│   └── Utility functions                         # symlog, two-hot, etc.
│
├── matwm_agent.py                                # 完全なエージェント実装 ★
│   └── MATWMAgent                                # 訓練ループ含む完全実装
│
├── 2026_MATWM_simple_tag_Implementation.ipynb   # メインNotebook ★
│   ├── セットアップ
│   ├── 環境確認
│   ├── 訓練ループ
│   ├── 可視化
│   └── 評価
│
├── data/                                         # データ保存先
│   └── replay_buffers/
│
└── results/                                      # 学習結果保存先
    ├── checkpoints/
    ├── training_curves.png
    └── logs/
```

---

## コア実装: 社会的世界モデル

### Teammate Predictor ★

**最も重要なコンポーネント**: 他のエージェントの行動を予測

```python
class TeammatePredictor(nn.Module):
    """
    Predict other agents' actions from focal agent's latent state
    
    ★ This is the CORE component for social world modeling ★
    
    It enables the focal agent to anticipate behaviors of other agents,
    which is crucial for coordination and competition.
    """
    
    def __init__(self, latent_dim=32, num_classes=32, action_dim=5, 
                 num_agents=4, hidden_dim=256):
        super().__init__()
        self.num_agents = num_agents
        self.action_dim = action_dim
        
        # Separate predictor for each other agent
        self.predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim * num_classes, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
            for _ in range(num_agents - 1)
        ])
    
    def forward(self, z, focal_agent_idx):
        """Predict actions of all other agents"""
        z_flat = z.reshape(*z.shape[:-2], -1)
        
        teammate_logits = {}
        predictor_idx = 0
        for agent_idx in range(self.num_agents):
            if agent_idx != focal_agent_idx:
                logits = self.predictors[predictor_idx](z_flat)
                teammate_logits[agent_idx] = logits
                predictor_idx += 1
        
        return teammate_logits
```

**効果**:
- 非定常性の軽減: 他エージェントの方策変化に対応
- 協調行動の促進: adversaries同士の連携
- 競争行動の改善: good agentがadversariesの動きを予測して逃げる

---

## アーキテクチャ詳細

### World Model

#### 1. Encoder/Decoder (Categorical VAE)

- **入力**: ベクトル観測 (14次元 or 16次元)
- **潜在空間**: 32 categorical variables × 32 classes = 1024次元離散空間
- **特徴**: Gumbel-Softmax による微分可能なサンプリング

#### 2. Dynamics Model (Transformer)

- **アーキテクチャ**: 4層, 8ヘッド, 512次元
- **入力**: 潜在状態系列 + 行動系列 (action scaled by agent ID)
- **出力**: 次時刻の潜在状態分布

#### 3. Reward Predictor

- **エンコーディング**: Two-hot symlog (Dreamer V3)
- **利点**: 極端な報酬値にもロバスト

#### 4. Continuation Predictor

- **出力**: Bernoulli分布 (エピソード継続確率)

### Agent (Actor-Critic)

#### Actor

- **方策**: Categorical分布
- **学習**: Policy gradient with imagined advantages

#### Critic

- **価値関数**: V(z) - 潜在状態から価値を推定
- **特徴**: Semi-centralized - 他エージェントの想像行動を考慮

---

## 訓練戦略

### 1. Prioritized Replay Buffer

- **優先度**: 最近の経験ほど高い重み (exponential decay)
- **理由**: 他エージェントの方策変化に追従

```python
# Priority decay per step
self.priorities = deque([p * 0.995 for p in self.priorities])
```

### 2. Action Scaling

- **Agent 0**: actions 0-4
- **Agent 1**: actions 5-9
- **Agent 2**: actions 10-14
- **Agent 3**: actions 15-19

これにより、世界モデルがどのエージェントの行動かを識別可能。

### 3. Imagination-based Training

- **実環境**: 1ステップ → **想像**: 15ステップ分の学習
- **サンプル効率**: 15倍の学習データを生成

---

## 訓練フロー

### Phase 1: Warmup (1000 steps)

- ランダム方策でReplay Bufferを埋める
- 学習は行わない

### Phase 2: Joint Training (残りのsteps)

各環境ステップごとに:

1. **環境との相互作用**
   - 各エージェントがActorで行動選択
   - 環境からreward, next_obs, doneを取得
   - Replay Bufferに保存

2. **World Model訓練** (各エージェント)
   - Prioritized ReplayからSequence sampling
   - 6つの損失関数を最小化:
     - Reconstruction loss
     - Dynamics loss
     - Reward loss
     - Continuation loss
     - **Teammate prediction loss** ★
     - KL divergence (with free nats)

3. **Agent訓練** (各エージェント)
   - Random sampling (uniform)
   - 想像ロールアウト (horizon=15)
   - Actor loss (policy gradient)
   - Critic loss (TD error)

### Phase 3: Evaluation

- Deterministic policy
- 複数エピソードで平均報酬を計算

---

## 損失関数

### World Model Total Loss

```
L_total = L_recon + L_dynamics + L_reward + L_cont + 0.5 * L_teammate + L_kl
```

#### 1. Reconstruction Loss

```python
L_recon = MSE(decoder(z), obs)
```

#### 2. Dynamics Loss

```python
L_dynamics = CrossEntropy(z_next_pred, z_next_target)
```

#### 3. Reward Loss (Two-hot Symlog)

```python
reward_symlog = symlog(reward)
reward_target = two_hot_encode(reward_symlog)
L_reward = CrossEntropy(reward_pred, reward_target)
```

#### 4. Continuation Loss

```python
L_cont = BCE(continuation_pred, 1 - done)
```

#### 5. Teammate Prediction Loss ★

```python
L_teammate = mean([
    CrossEntropy(teammate_pred[agent_i], actual_action[agent_i])
    for agent_i in other_agents
])
```

#### 6. KL Divergence (with Free Nats)

```python
L_kl = max(KL(z_posterior || z_prior), free_nats)
```

### Agent Losses

#### Actor Loss

```python
L_actor = -mean(log_prob(action) * advantage)
```

#### Critic Loss

```python
L_critic = MSE(V(z), returns)
```

---

## ハイパーパラメータ

### モデル

- Latent dim: 32 × 32 = 1024
- Hidden dim: 512
- Transformer layers: 4
- Attention heads: 8

### 訓練

- Batch size: 16
- Sequence length: 64
- Imagination horizon: 15
- Learning rate: 3e-4
- γ (discount): 0.99
- λ (GAE): 0.95

### Buffer

- Capacity: 100,000
- Priority decay: 0.995

### 実行

- Total steps: 100,000 (フル訓練)
- Warmup: 1,000
- Save interval: 5,000

---

## 使用方法

### 1. 環境セットアップ

```bash
pip install torch numpy matplotlib tqdm
pip install pettingzoo[mpe] supersuit
```

### 2. 訓練の実行

#### Notebook実行

```bash
jupyter notebook 2026_MATWM_simple_tag_Implementation.ipynb
```

#### スクリプト実行

```python
from matwm_implementation import MATWMConfig
from matwm_agent import MATWMAgent
from pettingzoo.mpe import simple_tag_v3

# Configuration
config = MATWMConfig(total_steps=100000)

# Create environment
env = simple_tag_v3.parallel_env(...)

# Create agents
agents = {name: MATWMAgent(config, name, idx, device) 
          for idx, name in enumerate(env.agents)}

# Train (see notebook for full loop)
```

### 3. 評価

```python
def evaluate_agents(agents, num_episodes=20):
    # ... (see notebook)
    pass

eval_rewards = evaluate_agents(agents)
```

---

## 期待される結果

### サンプル効率

- **目標**: 50K-100K steps で収束
- **比較**: 従来のmodel-free手法は1M+ steps必要

### 性能

#### Adversaries (predators)

- 初期: ランダムに動く
- 学習後: 協調してgood agentを追跡・包囲
- Teammate Predictorにより他のadversariesの動きを予測

#### Good Agent (prey)

- 初期: 逃げられない
- 学習後: adversariesの動きを予測して効率的に逃げる
- 障害物を利用した戦略

### 学習曲線

- **Adversaries**: 報酬が徐々に上昇 (0 → +10付近)
- **Good Agent**: 報酬が改善 (-10 → -5付近)
- **Teammate Loss**: 徐々に減少 → 他エージェント予測の精度向上

---

## 実装のポイント

### 1. Categorical VAE

- Gumbel-Softmax trick for differentiability
- One-hot encoding in forward pass
- Soft probabilities in backward pass

### 2. Action Scaling

```python
scaled_action = action + agent_idx * action_dim
```

これにより各エージェントのaction spaceが重複しない

### 3. Prioritized Replay

- World Model訓練: Prioritized (recent重視)
- Agent訓練: Uniform (diverse experiences)

### 4. Imagination Rollout

- Detach after each step to prevent long gradient chains
- Use world model in eval mode during imagination

### 5. Two-hot Encoding

- 連続値を2つのビンに分散
- よりsmooth な学習

---

## トラブルシューティング

### 問題1: 学習が進まない

**原因**: Replay Bufferが小さすぎる / Warmupが短い

**解決策**:
```python
config.buffer_size = 100000
config.warmup_steps = 2000
```

### 問題2: Teammate Lossが下がらない

**原因**: 他エージェントの方策がまだランダム / 学習率が高すぎる

**解決策**:
- Warmup期間を延ばす
- Teammate weightを調整: `config.teammate_weight = 0.3`

### 問題3: メモリ不足

**原因**: Sequence lengthが長すぎる / Batch sizeが大きすぎる

**解決策**:
```python
config.sequence_length = 32  # 64 → 32
config.batch_size = 8  # 16 → 8
```

### 問題4: 学習が不安定

**原因**: Learning rateが高すぎる / Gradient explosion

**解決策**:
```python
config.learning_rate = 1e-4  # 3e-4 → 1e-4
# Gradient clippingは既に実装済み (max_norm=100)
```

---

## 今後の拡張

### 1. γ-Progress Curiosity

Active World Model Learningの手法:

```python
class ProgressCuriosity:
    def compute_intrinsic_reward(self, z_curr, z_pred, z_actual):
        error_new = F.mse_loss(z_pred, z_actual)
        progress = self.error_old - error_new
        self.error_old = error_new
        return torch.clamp(progress, 0, 1)
```

### 2. Theory of Mind

より高度な社会的推論:

- 他エージェントの信念・意図の推定
- Recursive reasoning
- Mental state tracking

### 3. Communication

```python
class CommunicationModule(nn.Module):
    def forward(self, z_focal, z_others):
        # Attention-based message passing
        messages = self.attention(z_focal, z_others)
        z_augmented = torch.cat([z_focal, messages], dim=-1)
        return z_augmented
```

### 4. 階層的プランニング

- High-level: Goal selection
- Low-level: Action execution

### 5. Self-Play

- Population-based training
- Best response dynamics

---

## 参考文献

1. **MATWM**: Deihim, A., Alonso, E., & Apostolopoulou, D. (2025). *Transformer World Model for Sample Efficient Multi-Agent Reinforcement Learning*. arXiv:2506.18537.

2. **Dreamer V3**: Hafner, D., Rusu, A., Veness, J., Duan, J., Hadsell, R., & Wayne, G. (2023). *Mastering Diverse Domains through World Models*.

3. **STORM**: [Single-agent Transformer World Model]

4. **Categorical VAE**: Jang, E., Gu, S., & Poole, B. (2016). *Categorical Reparameterization with Gumbel-Softmax*.

5. **PettingZoo**: Terry, J. K., et al. (2021). *PettingZoo: Gym for Multi-Agent Reinforcement Learning*.

---

## ライセンス

This implementation is for educational purposes.

---

## 連絡先

プロジェクト: 世界モデルと知能 2025 最終課題

---

**実装完了!** 🎉

社会的世界モデルを用いたマルチエージェント強化学習の完全な実装が完了しました。


