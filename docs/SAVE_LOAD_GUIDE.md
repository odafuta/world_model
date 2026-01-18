# Save/Load機能 完全実装ガイド

## ✅ 実装完了内容

### 1. インポートの最適化

**修正前（冗長）:**
```python
from matwm_implementation import (
    MATWMConfig,
    PrioritizedReplayBuffer,  # ❌ 間接使用のみ
    Encoder, Decoder,         # ❌ 間接使用のみ
    DynamicsModel,            # ❌ 間接使用のみ
    ...
)
```

**修正後（最小限）:**
```python
from matwm_implementation import MATWMConfig  # ✅ 直接使用
# その他はMATWMAgent内部で使用
```

### 2. Save機能の完全実装

#### 訓練中の自動保存
- **頻度**: 5,000ステップごと（`config.save_interval`）
- **保存内容**:
  - ✅ 共有World Model
  - ✅ 共有World Model Optimizer
  - ✅ 全エージェントのActor/Critic
  - ✅ 全エージェントのOptimizer
  - ✅ Episode Rewards履歴
  - ✅ Training Metrics履歴
  - ✅ 現在のステップ数

#### 保存場所
```
results/matwm_2026_01_18_15_30_00/
├── checkpoint_5000/
│   ├── full_checkpoint.pt      # 完全な訓練状態
│   ├── adversary_0.pt          # 個別保存（後方互換性）
│   ├── adversary_1.pt
│   ├── adversary_2.pt
│   └── agent_0.pt
├── checkpoint_10000/
│   └── ...
└── final/
    └── full_checkpoint.pt      # 最終チェックポイント
```

### 3. Load機能の完全実装

#### 使用方法

**新規訓練:**
```python
agents, episode_rewards, training_metrics = train_matwm(config)
```

**訓練再開:**
```python
checkpoint_path = 'results/.../checkpoint_25000/full_checkpoint.pt'
agents, episode_rewards, training_metrics = train_matwm(
    config,
    resume_from=checkpoint_path
)
```

#### 再開時に復元されるもの
1. ✅ World Model パラメータ
2. ✅ World Model Optimizer状態（学習率、momentum等）
3. ✅ 全エージェントのActor/Critic パラメータ
4. ✅ 全エージェントのOptimizer状態
5. ✅ Episode Rewards履歴（可視化の継続）
6. ✅ Training Metrics履歴（学習曲線の継続）
7. ✅ 訓練ステップ数（正確な再開位置）

---

## 📊 使用例

### 例1: 長時間訓練（分割実行）

```python
# Day 1: 25,000ステップ訓練
config = MATWMConfig(total_steps=25000)
agents, rewards, metrics = train_matwm(config)
# → checkpoint_25000/full_checkpoint.pt に保存

# Day 2: 残り25,000ステップを続行（合計50,000）
config = MATWMConfig(total_steps=50000)  # 総合目標
agents, rewards, metrics = train_matwm(
    config,
    resume_from='results/.../checkpoint_25000/full_checkpoint.pt'
)
# → 25,000ステップから再開、50,000まで訓練
```

### 例2: 事故からの回復

```python
# 訓練中にクラッシュした場合
config = MATWMConfig(total_steps=50000)

# 最後のチェックポイントから再開
agents, rewards, metrics = train_matwm(
    config,
    resume_from='results/.../checkpoint_40000/full_checkpoint.pt'
)
# → 40,000ステップから再開
```

### 例3: ハイパラ調整後の継続

```python
# まずベースライン訓練
config = MATWMConfig(total_steps=10000)
agents, rewards, metrics = train_matwm(config)

# 学習率を調整して継続
config_tuned = MATWMConfig(
    total_steps=50000,
    agent_learning_rate=1e-4  # 変更
)
agents, rewards, metrics = train_matwm(
    config_tuned,
    resume_from='results/.../checkpoint_10000/full_checkpoint.pt'
)
```

---

## 🔍 実装の詳細

### save_full_checkpoint() の内部構造

```python
checkpoint = {
    'global_step': 25000,  # 現在のステップ
    'shared_world_model': state_dict,  # WMパラメータ
    'shared_wm_optimizer': state_dict,  # WM Optimizer
    'episode_rewards': {
        'adversary_0': [10.5, 12.3, ...],  # 全履歴
        'adversary_1': [...],
        ...
    },
    'training_metrics': {
        'shared_wm_total_loss': [0.52, 0.48, ...],
        'adversary_0_actor_loss': [...],
        ...
    },
    'agents': {
        'adversary_0': {
            'actor': state_dict,
            'critic': state_dict,
            'actor_optimizer': state_dict,
            'critic_optimizer': state_dict,
        },
        ...
    }
}
```

### load_full_checkpoint() の復元プロセス

1. **チェックポイント読み込み**
   ```python
   checkpoint = torch.load(path, map_location=device)
   ```

2. **World Model復元**
   ```python
   shared_world_model.load_state_dict(checkpoint['shared_world_model'])
   shared_wm_optimizer.load_state_dict(checkpoint['shared_wm_optimizer'])
   ```

3. **全エージェント復元**
   ```python
   for name, agent in agents.items():
       agent.actor.load_state_dict(checkpoint['agents'][name]['actor'])
       agent.critic.load_state_dict(checkpoint['agents'][name]['critic'])
       agent.actor_optimizer.load_state_dict(...)
       agent.critic_optimizer.load_state_dict(...)
   ```

4. **メトリクス復元**
   ```python
   episode_rewards = checkpoint['episode_rewards']
   training_metrics = checkpoint['training_metrics']
   start_step = checkpoint['global_step']
   ```

---

## 🎯 論文との対応

### MATWM論文の訓練設定

| 環境 | Total Steps | Checkpoint間隔（推奨） |
|------|------------|---------------------|
| Simple Tag (4 agents) | 50K | 5K (10回保存) |
| SMAC Easy Maps | 50K | 5K |
| SMAC Hard Maps | 200K | 10K (20回保存) |
| Image-based | 50K | 5K |

### 本実装の設定

```python
config = MATWMConfig(
    total_steps=50000,     # 論文準拠
    save_interval=5000,    # 10回保存
)
```

**保存頻度の推奨:**
- ✅ 5,000ステップ: 適度（デフォルト）
- ⚠️ 1,000ステップ: 頻繁すぎ（ディスクI/O過多）
- ❌ 10,000ステップ: 粗すぎ（クラッシュ時の損失大）

---

## 💾 ディスク使用量

### 1チェックポイントのサイズ（概算）

```
checkpoint_5000/
├── full_checkpoint.pt      # ~50-100MB（メイン）
│   ├── World Model         # ~20-40MB
│   ├── Optimizers          # ~20-40MB
│   ├── Agents (4)          # ~10-20MB
│   └── Metrics             # ~1-5MB
├── adversary_0.pt          # ~5MB（個別）
├── adversary_1.pt          # ~5MB
├── adversary_2.pt          # ~5MB
└── agent_0.pt              # ~5MB
Total: ~70-120MB/checkpoint
```

### 完全訓練でのディスク使用量

**50K steps, 5K間隔:**
- Checkpoints: 10個
- 合計: ~700MB - 1.2GB

---

## 🚀 ベストプラクティス

### 1. 定期的なバックアップ
```python
# 重要なチェックポイントを別ディレクトリにコピー
import shutil
shutil.copy(
    'results/.../checkpoint_25000/full_checkpoint.pt',
    'backups/milestone_25k.pt'
)
```

### 2. 古いチェックポイントの削除
```python
# 最新3つだけ残す（ディスク節約）
# 訓練関数に追加可能
```

### 3. クラウドへの自動アップロード
```python
# Google Drive, AWS S3等へ自動バックアップ
# 訓練関数のsave後に追加
```

---

## ✅ まとめ

| 機能 | 実装状況 | 使用頻度 |
|------|----------|----------|
| **自動保存** | ✅ 完全実装 | 5,000ステップごと |
| **訓練再開** | ✅ 完全実装 | 必要時に手動 |
| **メトリクス保存** | ✅ 完全実装 | 自動 |
| **最終保存** | ✅ 完全実装 | 訓練終了時 |

**これで論文準拠の完全なSave/Load機能が実装されました！** 🎉

長時間訓練でも安心して中断・再開できます。
