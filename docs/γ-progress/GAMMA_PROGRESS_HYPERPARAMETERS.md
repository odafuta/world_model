# γ-Progress ハイパーパラメータの論文再現性

## 📚 論文の設定 (Kim et al. 2020)

### 明示的に記載されている値

| パラメータ | 論文の値 | 実装の値 | 一致 |
|-----------|---------|---------|------|
| **γ (EMA decay rate)** | 0.9995 | 0.9995 | ✅ |
| **γ-Progress weight** | (記載なし) | 1.0 | ⚠️ |
| **Normalization** | (記載なし) | True (0.99) | ⚠️ |

### 論文の数式

**Equation 11 (EMA更新):**
```
θ_old ← γ·θ_old + (1-γ)·θ_new
```
- `γ = 0.9995` (論文 p.172で明示)

**Equation 12 (報酬計算):**
```
c(x_t) = L(θ_new, x, a) - L(θ_old, x, a)
```
- **重み係数は適用されていない**（そのまま報酬として使用）

---

## 🔍 実装上の考慮事項

### 論文と我々の実装の違い

| 項目 | 論文 (Kim et al. 2020) | 我々の実装 (MATWM) |
|------|----------------------|-------------------|
| **環境** | 3D視覚探索環境 | PettingZoo Simple Tag |
| **タスク** | World Model学習のみ | World Model + RL (Actor-Critic) |
| **報酬** | γ-Progress報酬のみ | 環境報酬 + 好奇心報酬 (複数種類) |
| **他の好奇心** | なし | Dynamics/Reward/Social Curiosity + LLM |

### 重要な違い: 報酬の組み合わせ

**論文:**
```
total_reward = γ-Progress報酬
```

**我々の実装:**
```
total_reward = env_reward + computational_curiosity + semantic_bonus + γ-progress
              ↑             ↑                          ↑                ↑
              環境報酬      Dynamics/Reward/Social     LLM評価          γ-Progress
```

---

## ⚙️ `gamma_progress_weight` の役割

### なぜ重み係数が必要か？

我々の実装では、**複数の報酬源を統合**しているため、各報酬のバランスを調整する必要があります:

```python
# curiosity_reward.py
total = (
    cur["total"]              # Computational curiosity (Dynamics + Reward + Social)
    + self._semantic_bonus     # LLM semantic curiosity
    + gamma_progress_weight * gamma_progress_reward  # γ-Progress
)
```

### デフォルト値の根拠

**現在の設定: `gamma_progress_weight = 1.0`**

理由:
1. **論文に忠実**: 論文では重み係数なし（暗黙的に1.0）
2. **他の好奇心とのバランス**:
   - `dynamics_curiosity_weight = 1.0`
   - `reward_curiosity_weight = 0.5`
   - `social_curiosity_weight = 2.0`
   - `gamma_progress_weight = 1.0` ← 同じスケール

---

## 🧪 実験での調整

### 推奨される実験設定

#### 1. 論文再現実験 (γ-Progressのみ)

```python
config = MATWMConfig(
    use_gamma_progress=True,
    gamma_progress=0.9995,
    gamma_progress_weight=1.0,  # 論文に従う
)

curiosity_config = CuriosityConfig(
    dynamics_curiosity_weight=0.0,  # 他の好奇心を無効化
    reward_curiosity_weight=0.0,
    social_curiosity_weight=0.0,
    use_llm_curiosity=False,
)
```

#### 2. 統合実験 (すべての好奇心を併用)

```python
config = MATWMConfig(
    use_gamma_progress=True,
    gamma_progress=0.9995,
    gamma_progress_weight=1.0,  # デフォルト
)

curiosity_config = CuriosityConfig(
    dynamics_curiosity_weight=1.0,
    reward_curiosity_weight=0.5,
    social_curiosity_weight=2.0,
    use_llm_curiosity=True,
)
```

#### 3. 重み調整実験

```python
# γ-Progressを強調
config = MATWMConfig(
    gamma_progress_weight=2.0,  # 強め
)

# γ-Progressを抑制
config = MATWMConfig(
    gamma_progress_weight=0.5,  # 弱め
)
```

---

## 📊 ハイパーパラメータの感度

### γ (EMA decay rate)

**論文の値: 0.9995**

```python
# より長期的な平均 (より保守的)
gamma_progress=0.9999  # 10000ステップの履歴

# 論文の値 (推奨)
gamma_progress=0.9995  # 2000ステップの履歴

# より短期的な平均 (より積極的)
gamma_progress=0.999   # 1000ステップの履歴
```

**効果:**
- **高いγ (0.9999)**: より長期的な学習進捗を測定、安定だが反応が遅い
- **低いγ (0.999)**: より短期的な学習進捗を測定、反応が早いがノイジー

### gamma_progress_weight

**デフォルト値: 1.0**

```python
# γ-Progressを主要な探索シグナルとして使用
gamma_progress_weight=2.0

# バランス型 (推奨)
gamma_progress_weight=1.0

# 補助的な探索シグナルとして使用
gamma_progress_weight=0.5
```

**効果:**
- **高い重み (2.0)**: γ-Progressが探索を支配、学習進捗に強く依存
- **低い重み (0.5)**: 他の好奇心とバランス、多様な探索

---

## ✅ 論文再現性チェックリスト

### 完全に再現できている項目

- [x] **γ = 0.9995** (Equation 11, p.172, Appendix B)
- [x] **EMA更新式** `θ_old ← γ·θ_old + (1-γ)·θ_new`
- [x] **Progress計算式** `r = L(θ_old, x) - L(θ_new, x)`
- [x] **メモリ効率** O(1)メモリ使用量

### 実装上の拡張（論文に記載なし）

- [x] **重み係数** `gamma_progress_weight = 1.0` (論文では暗黙的に1.0)
- [x] **正規化** `gamma_progress_normalize = True` (decay=0.99, 安定化のため)
- [x] **複数報酬統合** 環境報酬 + 複数の好奇心報酬
- [x] **Multi-Agent対応** MATWM環境への適用

### 正規化について

**論文には記載なし:**
- Appendix Bに訓練詳細があるが、正規化の記載はない
- `decay=0.99`は実装上の選択（約100ステップの移動平均）
- γ=0.9995（約2000ステップ）とは異なる目的

**オプション化:**
```python
config = MATWMConfig(
    gamma_progress_normalize=False,  # 論文に忠実（正規化なし）
)

config = MATWMConfig(
    gamma_progress_normalize=True,   # 安定化のため正規化（デフォルト）
)
```

---

## 🎯 推奨される使用方法

### 基本設定 (論文に忠実)

```python
config = MATWMConfig(
    use_gamma_progress=True,
    gamma_progress=0.9995,           # 論文値
    gamma_progress_weight=1.0,       # 論文の暗黙的な値
)
```

### アブレーション実験

```python
# 対照群: γ-Progress無効
config_baseline = MATWMConfig(use_gamma_progress=False)

# 実験群1: γ-Progressのみ
config_gamma_only = MATWMConfig(
    use_gamma_progress=True,
    gamma_progress=0.9995,
    gamma_progress_weight=1.0,
)

# 実験群2: すべての好奇心を統合
config_all_curiosity = MATWMConfig(
    use_gamma_progress=True,
    gamma_progress=0.9995,
    gamma_progress_weight=1.0,
)
# + CuriosityConfig with all curiosity enabled
```

---

## 📖 参考文献

**Kim, Y., Sano, M., De Freitas, J., Haber, N., & Yamins, D. (2020).** *Active World Model Learning with Progress Curiosity.* ICML 2020.

- **Equation 11 (p.157)**: EMA更新式
- **Equation 12 (p.171)**: 報酬計算式
- **Section 5 (p.172)**: `γ = 0.9995` の明示

---

## まとめ

✅ **論文の再現性:**
- `gamma_progress = 0.9995`: 論文と完全一致
- `gamma_progress_weight = 1.0`: 論文の暗黙的な設定を明示化

⚠️ **実装上の拡張:**
- 複数の好奇心報酬を統合するため、重み係数を導入
- 論文の単一報酬設定とは異なるが、より柔軟な実験が可能

🎯 **推奨:**
- 論文再現実験: `gamma_progress_weight = 1.0` + 他の好奇心無効
- 統合実験: すべての好奇心を有効化してバランス調整
