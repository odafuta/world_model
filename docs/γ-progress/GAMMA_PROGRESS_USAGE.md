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

---

## 📋 完全な使用例

### ノートブックでの使用方法

```python
# ========================================
# Step 1: Config設定
# ========================================

# γ-Progress有効
config = MATWMConfig(
    use_gamma_progress=True,      # ← このフラグだけ！
    gamma_progress=0.9995,         # EMA decay rate
    gamma_progress_weight=1.0,     # 報酬の重み
    gamma_progress_normalize=True, # 正規化（オプション）
    # ... その他のパラメータ
)

# Curiosity Config（γ-Progressとは独立）
curiosity_config = CuriosityConfig(
    dynamics_curiosity_weight=1.0,
    reward_curiosity_weight=0.5,
    social_curiosity_weight=2.0,
    use_llm_curiosity=True,
)

# ========================================
# Step 2: World Model作成（自動分岐）
# ========================================

if config.use_gamma_progress:
    # γ-Progress有効 → EMAも作成
    shared_wm, shared_wm_ema, shared_wm_opt = MATWMAgent.create_shared_world_model_with_ema(config, device)
    print('✓ γ-Progress enabled')
else:
    # γ-Progress無効 → EMAなし
    shared_wm, shared_wm_opt = MATWMAgent.create_shared_world_model(config, device)
    shared_wm_ema = None
    print('✓ γ-Progress disabled')

# ========================================
# Step 3: CuriosityManager作成
# ========================================

curiosity_managers = create_curiosity_managers(
    agent_names,
    curiosity_config,
    matwm_config=config,           # ← use_gamma_progressを含む
    world_model=shared_wm,
    world_model_ema=shared_wm_ema,  # ← Trueなら有効、Falseなら None
    device=device,
)

# 内部で自動判定:
# if config.use_gamma_progress and world_model_ema is not None:
#     self.gamma_progress = GammaProgressReward(...)  # 有効化
# else:
#     self.gamma_progress = None  # 無効化

# ========================================
# Step 4: トレーニングループ
# ========================================

for step in range(total_steps):
    # ... 環境とのインタラクション ...
    
    # 内発的報酬計算
    intrinsic_r = curiosity_managers[name].compute_intrinsic_reward(
        shared_wm, obs_padded, actions[name],
        env_r, next_obs_padded, other_acts, device,
        done=done[name],
    )
    
    # 内部処理:
    # - Computational Curiosity: 常に計算される
    # - γ-Progress: use_gamma_progress=True の時だけ計算される
    # - total = computational + semantic + gamma_progress
    
    total_r = env_r + intrinsic_r
    
    # ... Replay bufferに保存 ...
    
    # World Model訓練
    if step >= warmup_steps:
        wm_metrics = MATWMAgent.train_world_model_shared(...)
        
        # γ-Progress: EMA更新（自動分岐）
        if config.use_gamma_progress and shared_wm_ema is not None:
            MATWMAgent.update_shared_world_model_ema(
                shared_wm, shared_wm_ema, config.gamma_progress
            )
```

---

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

---

## 🔍 動作確認方法

### ノートブック内で確認

```python
# セルを追加して確認
print("=== Curiosity Configuration ===")
for name, mgr in curiosity_managers.items():
    print(f"\n{name}:")
    print(f"  Computational Curiosity: {mgr.curiosity is not None}")
    print(f"  γ-Progress: {mgr.gamma_progress is not None}")
    print(f"  LLM Curiosity: {mgr.llm_evaluator is not None}")
    
    if mgr.gamma_progress is not None:
        print(f"    γ = {config.gamma_progress}")
        print(f"    weight = {config.gamma_progress_weight}")
```

### 期待される出力

#### use_gamma_progress=False の場合

```
=== Curiosity Configuration ===

adversary_0:
  Computational Curiosity: True  ✅
  γ-Progress: False              ✅
  LLM Curiosity: True            ✅

adversary_1:
  Computational Curiosity: True  ✅
  γ-Progress: False              ✅
  LLM Curiosity: True            ✅
```

#### use_gamma_progress=True の場合

```
=== Curiosity Configuration ===

adversary_0:
  Computational Curiosity: True  ✅
  γ-Progress: True               ✅
  LLM Curiosity: True            ✅
    γ = 0.9995
    weight = 1.0

adversary_1:
  Computational Curiosity: True  ✅
  γ-Progress: True               ✅
  LLM Curiosity: True            ✅
    γ = 0.9995
    weight = 1.0
```

---

## ✅ チェックリスト

実装が正しく動作するための確認項目:

- [x] `create_curiosity_managers()` が更新されている
- [x] ノートブックで `create_curiosity_managers()` を使用
- [x] `matwm_config`, `world_model`, `world_model_ema`, `device` を渡している
- [x] `config.use_gamma_progress=True/False` で制御可能
- [x] Computational Curiosity は常に有効
- [x] γ-Progress は条件付きで有効

---

## 🎯 まとめ

### 質問への回答

1. **消した内容は何？**
   - `create_curiosity_managers()` の呼び出しを一旦削除
   - しかし、**正しく修正して復活させました**

2. **`use_gamma_progress=True` だけで調整できる？**
   - ✅ **はい！** このフラグ1つで制御可能

3. **`curiosity_reward` は常に組み込まれてる？**
   - ✅ **はい！** `use_gamma_progress` に関係なく常に有効

### 現在の実装状態

```python
# ノートブック内
curiosity_managers = create_curiosity_managers(
    agent_names,
    curiosity_config,
    matwm_config=config,           # ✅ 渡している
    world_model=shared_wm,          # ✅ 渡している
    world_model_ema=shared_wm_ema,  # ✅ 渡している（Noneの可能性あり）
    device=device,                  # ✅ 渡している
)

# 内部で自動判定
# if config.use_gamma_progress and world_model_ema is not None:
#     self.gamma_progress = GammaProgressReward(...)  # 有効化
# else:
#     self.gamma_progress = None  # 無効化
```

**結論:** 正しく動作します！🎉
