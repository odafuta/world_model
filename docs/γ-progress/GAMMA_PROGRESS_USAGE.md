# γ-Progress 使用方法ガイド

## 🎯 重要な原則

### 1. γ-Progressの制御

**`config.use_gamma_progress` 1つのフラグで制御:**

```python
# γ-Progress有効
config = MATWMConfig(use_gamma_progress=True)

# γ-Progress無効
config = MATWMConfig(use_gamma_progress=False)
```

### 2. Computational Curiosity は常に有効

**`config.use_gamma_progress` の値に関係なく、常に動作:**

```python
# use_gamma_progress=False でも
curiosity_managers[name].compute_intrinsic_reward(...)
# → Computational Curiosity (Dynamics/Reward/Social) は計算される
# → γ-Progress は計算されない (0.0)
```

## 🧪 実験設定例

### 実験1: ベースライン（γ-Progress無効）

```python
config = MATWMConfig(
    use_gamma_progress=False,  # ← γ-Progress無効
    total_steps=50000,
    warmup_steps=1000,
)

curiosity_config = CuriosityConfig(
    dynamics_curiosity_weight=1.0,
    reward_curiosity_weight=0.5,
    social_curiosity_weight=2.0,
    use_llm_curiosity=True,
)

# 内発的報酬 = Computational + LLM
# total = cur["total"] + semantic_bonus + 0.0
```

### 実験2: γ-Progress有効

```python
config = MATWMConfig(
    use_gamma_progress=True,   # ← γ-Progress有効
    gamma_progress=0.9995,
    gamma_progress_weight=1.0,
    total_steps=50000,
    warmup_steps=1000,
)

curiosity_config = CuriosityConfig(
    dynamics_curiosity_weight=1.0,
    reward_curiosity_weight=0.5,
    social_curiosity_weight=2.0,
    use_llm_curiosity=True,
)

# 内発的報酬 = Computational + LLM + γ-Progress
# total = cur["total"] + semantic_bonus + gamma_progress_reward
```

### 実験3: γ-Progressのみ（論文再現）

```python
config = MATWMConfig(
    use_gamma_progress=True,
    gamma_progress=0.9995,
    gamma_progress_weight=1.0,
    total_steps=50000,
    warmup_steps=1000,
)

curiosity_config = CuriosityConfig(
    dynamics_curiosity_weight=0.0,  # ← 無効化
    reward_curiosity_weight=0.0,    # ← 無効化
    social_curiosity_weight=0.0,    # ← 無効化
    use_llm_curiosity=False,        # ← 無効化
)

# 内発的報酬 = γ-Progressのみ
# total = 0.0 + 0.0 + gamma_progress_reward
```