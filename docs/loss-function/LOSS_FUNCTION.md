# Loss Function Updates - 論文への忠実な実装

## 📋 更新概要

MATWM論文 (arXiv:2506.18537) の損失関数定義（Equations 3-14）に完全に準拠するように、World ModelとActor-Criticの損失関数を修正しました。

---

## 🔬 World Model損失関数

### 論文の定義 (Equation 3)

```
L(φ) = 1/(BT) Σ[L_rec + L_rew + L_con + L_team + β₁L_dyn + β₂L_rep]
```

where:
- β₁ = 0.5 (kl_weight)
- β₂ = 0.1 (representation_weight)

## 🎯 Actor-Critic損失関数

### 論文の定義 (Equations 11-14)

**Actor Loss (Equation 11):**
```
L(θ) = -sg((G_t^λ - V)/max(1,S)) ln π - η H(π)
```

**Critic Loss (Equation 11):**
```
L(ψ) = (V - sg(G_t^λ))² + (V - sg(V_EMA))²
```

**EMA Update (Equation 14):**
```
ψ_t+1^EMA = σψ_t^EMA + (1-σ)ψ_t
```
---

## 🔧 Config更新

### `matwm_implementation.py`

**更新内容:**
```python
# Loss Weights (Paper Equation 3, Table C.6)
kl_weight: float = 0.5              # β₁: Dynamics loss weight
representation_weight: float = 0.1   # β₂: Representation loss weight
free_nats: float = 1.0              # Free bits for KL losses

# RL Parameters (Paper Equations 11-14)
gamma: float = 0.99
lambda_gae: float = 0.95
entropy_coef: float = 0.01          # η in Equation 11
critic_ema_decay: float = 0.98      # σ in Equation 14 (NEW)
```

---

## ✅ 検証項目

### World Model

- [x] L_rec: MSE (Equation 4) ✅
- [x] L_rew: Symlog two-hot (Equation 5) ✅
- [x] L_con: BCE (Equation 6) ✅
- [x] L_team: Cross-entropy with stop-gradient (Equation 8) ✅
- [x] L_dyn: KL divergence with sg(target) (Equation 9a) ✅
- [x] L_rep: KL divergence with sg(prediction) (Equation 9b) ✅
- [x] Total loss weight: β₁=0.5, β₂=0.1 (Equation 3) ✅

### Actor-Critic

- [x] Actor: Percentile normalization + entropy (Equation 11, 13) ✅
- [x] Critic: Two MSE terms (λ-return + EMA) (Equation 11) ✅
- [x] EMA update: σ=0.98 (Equation 14) ✅
- [x] λ-return: GAE implementation (Equation 12) ✅

---

## 🚀 期待される効果

### World Model

1. **L_dyn (Dynamics Loss)**
   - KL divergenceによる分布のソフトマッチング
   - Dynamics modelの予測精度向上

2. **L_rep (Representation Loss)**
   - Encoderが予測しやすい潜在表現を学習
   - Dynamics modelとEncoderの協調学習

3. **Teammate Predictor stop-gradient**
   - Encoderの学習安定化
   - ノイズ耐性の向上

### Actor-Critic

1. **Critic EMA正則化**
   - Value関数の学習安定化
   - 過学習の抑制

2. **Percentile正規化**
   - 外れ値に対する頑健性
   - より安定した方策更新

3. **2項Critic Loss**
   - λ-returnとEMAの双方向正則化
   - より高品質な価値推定

---

## 🔬 論文との対応表

| 論文の記号 | コード変数名 | 説明 |
|-----------|------------|------|
| L_rec | `recon_loss` | Reconstruction loss (Eq. 4) |
| L_rew | `reward_loss` | Reward loss (Eq. 5) |
| L_con | `cont_loss` | Continuation loss (Eq. 6) |
| L_team | `teammate_loss` | Teammate loss (Eq. 8) |
| L_dyn | `dynamics_loss` | Dynamics loss (Eq. 9a) |
| L_rep | `representation_loss` | Representation loss (Eq. 9b) |
| β₁ | `config.kl_weight` | 0.5 (Table C.6) |
| β₂ | `config.representation_weight` | 0.1 (Table C.6) |
| G_t^λ | `lambda_returns` | λ-return (Eq. 12) |
| S | `S` | Normalization factor (Eq. 13) |
| V_EMA | `self.critic_ema` | EMA critic (Eq. 14) |
| σ | `config.critic_ema_decay` | 0.98 (EMA decay) |
| η | `config.entropy_coef` | 0.01 (Entropy coef) |

---

## 📚 参考文献

Deihim et al. (2025). "Transformer World Model for Sample Efficient Multi-Agent Reinforcement Learning". arXiv:2506.18537

重要なセクション:
- Section 3.1: World Model Loss (Equations 3-9)
- Section 3.2: Training Structure
- Equations 11-14: Actor-Critic Loss
- Table C.6: Hyperparameter Settings
- Line 140: Teammate predictor stop-gradient

---

## 更新日時

2026-02-11
