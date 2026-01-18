# matwm_utils.py API Documentation

## 📚 命名規則 (Naming Convention)

### PUBLIC API (外部から呼び出す関数)
通常の名前で定義。Notebookや他のモジュールから直接使用。

### INTERNAL HELPER (内部ヘルパー関数)
`_`（アンダースコア）で始まる名前。PUBLIC API内部でのみ使用。外部からは呼び出さない。

---

## 🌟 PUBLIC API Functions

### 1. Weight Initialization

#### `initialize_matwm_weights(world_model, actor, critic)`
全MATWM コンポーネントの重みを初期化。

**使用例:**
```python
from matwm_utils import initialize_matwm_weights

initialize_matwm_weights(shared_world_model, dummy_agent.actor, dummy_agent.critic)
```

**使用箇所 in Notebook:**
- Cell 8: モデル作成後に呼び出し ✅

---

### 2. Model Saving/Loading

#### `save_full_checkpoint(agents, shared_world_model, shared_wm_optimizer, episode_rewards, training_metrics, global_step, path)`
完全な訓練状態を保存。

**使用例:**
```python
from matwm_utils import save_full_checkpoint

save_full_checkpoint(
    agents, shared_world_model, shared_wm_optimizer,
    episode_rewards, training_metrics, global_step,
    'checkpoints/final.pt'
)
```

**使用箇所 in Notebook:**
- 現在未使用 ❌
- **推奨**: Cell 10 (訓練ループ) に追加すべき

#### `load_full_checkpoint(agents, shared_world_model, shared_wm_optimizer, path, device)`
完全な訓練状態を復元。

**使用例:**
```python
from matwm_utils import load_full_checkpoint

episode_rewards, training_metrics, global_step = load_full_checkpoint(
    agents, shared_world_model, shared_wm_optimizer,
    'checkpoints/final.pt', device
)
```

**使用箇所 in Notebook:**
- 現在未使用 ❌
- **推奨**: 訓練再開時に使用

---

### 3. Visualization

#### `plot_training_progress(episode_rewards, training_metrics, save_path='training_curves.png')`
9パネルの詳細な訓練進捗可視化。

**生成されるグラフ:**
1. Episode Rewards (Moving Average)
2. World Model Total Loss
3. ★ Teammate Prediction Loss ★ (社会的世界モデル)
4. World Model Component Losses
5. Actor Loss (Per Agent)
6. Critic Loss (Per Agent)
7. Mean Imagined Reward (Rollout)
8. Mean Value Estimate
9. Cumulative Rewards

**使用例:**
```python
from matwm_utils import plot_training_progress

plot_training_progress(
    episode_rewards, 
    training_metrics, 
    save_path='results/training_curves_enhanced.png'
)
```

**使用箇所 in Notebook:**
- Cell 11: 訓練後の可視化 ✅

---

### 4. Architecture Inspection

#### `inspect_matwm_architecture(world_model, actor, critic, config, device)`
MATWMアーキテクチャの総合的な検証。

**検証項目:**
1. **Dummy Input Test**: 全コンポーネントの動作確認
2. **Layer Count**: Linear/Conv層の集計
3. **Parameter Count**: 詳細なパラメータ数分析
4. **Detailed Summary**: torchinfo による詳細表示

**使用例:**
```python
from matwm_utils import inspect_matwm_architecture

inspect_matwm_architecture(
    shared_world_model, 
    dummy_agent.actor, 
    dummy_agent.critic, 
    config, 
    device
)
```

**使用箇所 in Notebook:**
- Cell 8: モデル作成直後に呼び出し ✅

---

### 5. GPU Environment Information

#### `print_gpu_info()`
GPU環境の完全表示。

**表示内容:**
- CUDA/PyTorch バージョン
- 利用可能な全GPU（A100, L4等）
- 現在使用中のGPU
- メモリ情報（Total/Allocated/Reserved）
- GPU種類の自動判定

**使用例:**
```python
from matwm_utils import print_gpu_info

gpu_info = print_gpu_info()
```

**使用箇所 in Notebook:**
- Cell 7: 訓練開始前に呼び出し ✅

#### `setup_matwm_training(config, device)`
MATWM訓練の完全セットアップ（GPU情報表示 + 次ステップガイド）。

**使用例:**
```python
from matwm_utils import setup_matwm_training

setup_info = setup_matwm_training(config, device)
```

**使用箇所 in Notebook:**
- Cell 7: `print_gpu_info()` の直後に呼び出し ✅

---

## 🔒 INTERNAL HELPER Functions

これらの関数は外部から直接呼び出さない。PUBLIC API内部で使用。

### Weight Initialization
- `_init_weights(module)`: 個別モジュールの重み初期化

### Architecture Inspection
- `_count_layers(model, layer_types)`: 層数カウント
- `_count_parameters(model, trainable_only)`: パラメータ数カウント

### GPU Information
- `_get_gpu_info()`: GPU情報の取得
- `_identify_gpu_type(gpu_name)`: GPU種類の判定

---

## 📊 Notebook活用状況

| 機能 | PUBLIC API | Cell | 使用状況 |
|------|-----------|------|----------|
| **重み初期化** | `initialize_matwm_weights()` | 8 | ✅ 使用中 |
| **保存** | `save_full_checkpoint()` | - | ❌ 未使用 |
| **読み込み** | `load_full_checkpoint()` | - | ❌ 未使用 |
| **可視化** | `plot_training_progress()` | 11 | ✅ 使用中 |
| **アーキテクチャ検証** | `inspect_matwm_architecture()` | 8 | ✅ 使用中 |
| **GPU情報** | `print_gpu_info()` | 7 | ✅ 使用中 |
| **セットアップ** | `setup_matwm_training()` | 7 | ✅ 使用中 |

### ✅ 活用できている機能 (5/7)
1. `initialize_matwm_weights()` - モデル作成直後に使用
2. `plot_training_progress()` - 訓練後の詳細可視化
3. `inspect_matwm_architecture()` - アーキテクチャ検証
4. `print_gpu_info()` - GPU環境確認
5. `setup_matwm_training()` - 訓練セットアップ

### ❌ 未活用の機能 (2/7)
1. `save_full_checkpoint()` - **推奨**: 訓練ループに追加
2. `load_full_checkpoint()` - **推奨**: 訓練再開用セル追加

---

## 🎯 推奨改善

### Notebookへの追加

#### 訓練ループでのCheckpoint保存
```python
# Cell 10 (訓練関数内) に追加
if global_step % config.save_interval == 0 and global_step >= config.warmup_steps:
    # 既存の個別保存に加えて
    save_full_checkpoint(
        agents, shared_world_model, shared_wm_optimizer,
        episode_rewards, training_metrics, global_step,
        os.path.join(checkpoint_dir, 'full_checkpoint.pt')
    )
```

#### 訓練再開用セル
```python
# 新しいセル: 訓練再開時に使用
if os.path.exists('checkpoints/full_checkpoint.pt'):
    episode_rewards, training_metrics, start_step = load_full_checkpoint(
        agents, shared_world_model, shared_wm_optimizer,
        'checkpoints/full_checkpoint.pt', device
    )
    print(f"Resuming from step {start_step}")
```

---

## 📝 まとめ

### 可読性改善完了 ✅
- PUBLIC API: 通常の名前（7関数）
- INTERNAL HELPER: `_`プレフィックス（5関数）
- 明確な役割分担で保守性向上

### Notebook活用状況 ✅
- **5/7の機能が活用されている**
- 重要な機能（初期化、可視化、検証、GPU確認）は全て使用中
- 保存/読み込みは訓練ループへの統合を推奨

**可読性と活用度が大幅に向上しました！** 🎉
