# カリキュラム学習型好奇心の使い方

## 概要

3段階のカリキュラム学習を実装しました：

1. **フェーズ1（序盤）**: ゲームの勝ち負けにこだわって学習（低い好奇心重み）
2. **フェーズ2（中盤）**: 好奇心駆動で色々なチャレンジを試す（高い好奇心重み）
3. **フェーズ3（終盤）**: 再び勝ちにこだわって成長（低い好奇心重み）

## 使用方法

### 1. 基本設定

```python
from curiosity_reward import CuriosityConfig, create_curiosity_managers

# カリキュラム学習モードの設定
config = CuriosityConfig(
    # 基本設定
    dynamics_curiosity_weight=1.0,
    reward_curiosity_weight=0.5,
    social_curiosity_weight=0.01,

    # カリキュラム学習を有効化
    curiosity_decay_method="curriculum",

    # 全体のエピソード数
    curriculum_total_episodes=1000,

    # 各フェーズの境界（全体の割合）
    curriculum_phase1_end=0.3,   # 0-30%: フェーズ1
    curriculum_phase2_end=0.7,   # 30-70%: フェーズ2
                                 # 70-100%: フェーズ3

    # 各フェーズの好奇心重み
    curriculum_phase1_weight=0.0001,  # フェーズ1: 低い（勝ち負け重視）
    curriculum_phase2_weight=0.005,   # フェーズ2: 高い（好奇心駆動）
    curriculum_phase3_weight=0.0001,  # フェーズ3: 低い（勝ち負け重視）
)

# CuriosityManagerの作成
curiosity_managers = create_curiosity_managers(
    agent_names=['adversary_0', 'adversary_1', 'adversary_2', 'agent_0'],
    config=config,
)
```

### 2. 訓練ループでの使用

```python
for episode in range(config.curriculum_total_episodes):
    # エピソード開始時にリセット
    for agent_name, manager in curiosity_managers.items():
        manager.reset_episode(episode=episode)

    # エピソード実行
    for step in range(max_steps):
        # ... 環境との相互作用 ...

        # 内発的報酬の計算
        intrinsic_reward = manager.compute_intrinsic_reward(
            world_model=world_model,
            obs=obs,
            action=action,
            reward=env_reward,
            next_obs=next_obs,
            other_actions=other_actions,
            device=device,
        )

        # 合計報酬 = 環境報酬 + 内発的報酬
        total_reward = env_reward + intrinsic_reward

    # エピソード終了時
    for agent_name, manager in curiosity_managers.items():
        summary = manager.get_episode_summary()
        print(f"{agent_name}: Phase {summary.get('curriculum_phase', '?')}, "
              f"Weight: {summary.get('curiosity_weight', 0):.4f}")
```

## パラメータの調整ガイド

### フェーズの境界調整

```python
# より長い探索フェーズが必要な場合
curriculum_phase1_end=0.2   # 0-20%: 基礎学習
curriculum_phase2_end=0.8   # 20-80%: 探索期間（長め）

# より早く活用フェーズに移行したい場合
curriculum_phase1_end=0.4   # 0-40%: 基礎学習（長め）
curriculum_phase2_end=0.6   # 40-60%: 探索期間（短め）
```

### 好奇心重みの調整

```python
# より強い好奇心駆動が必要な場合
curriculum_phase2_weight=0.01   # より高い値に設定

# 勝ち負けを完全に無視したくない場合
curriculum_phase1_weight=0.0005  # 完全にゼロにしない
curriculum_phase3_weight=0.0005
```

### 全体のエピソード数の調整

```python
# 環境の複雑さに応じて調整
curriculum_total_episodes=500   # シンプルな環境
curriculum_total_episodes=2000  # 複雑な環境
```

## 学習の進捗確認

```python
# 統計情報の表示
for agent_name, manager in curiosity_managers.items():
    manager.print_stats()

# 出力例:
# [adversary_0] Curiosity: dynamics=0.5234  social=0.3421  reward=0.2156  weight=0.0050  phase=2  computed=1250
```

## カリキュラム学習の効果

### フェーズ1（序盤、0-30%）
- **目的**: 基本的な戦略の学習
- **好奇心重み**: 0.0001（低い）
- **期待される振る舞い**: 環境報酬に基づいて勝つための基本戦略を学習

### フェーズ2（中盤、30-70%）
- **目的**: 探索的チャレンジ
- **好奇心重み**: 0.005（高い）
- **期待される振る舞い**:
  - 新しい協調パターンの発見
  - 未踏の状態空間の探索
  - 予想外の戦術の試行

### フェーズ3（終盤、70-100%）
- **目的**: 最適化と勝利への集中
- **好奇心重み**: 0.0001（低い）
- **期待される振る舞い**:
  - フェーズ2で発見した有効な戦術を洗練
  - 勝率の最大化に集中

## トラブルシューティング

### 問題: フェーズ2で探索が不十分

**解決策**:
```python
curriculum_phase2_weight=0.01  # 重みを増やす
curriculum_phase2_end=0.8      # フェーズ2を長くする
```

### 問題: フェーズ3で勝率が下がる

**解決策**:
```python
curriculum_phase2_end=0.6      # 早めに活用フェーズに移行
curriculum_phase3_weight=0.00005  # フェーズ3の重みをさらに下げる
```

### 問題: フェーズ間の遷移が急激すぎる

**解決策**: コード内で遷移期間（現在は各フェーズの最初の20%）を調整
```python
# curiosity_reward.py の _update_weight メソッド内
if phase_progress < 0.3:  # 20% → 30% に変更
    blend = phase_progress / 0.3
```

## その他の減衰モード

カリキュラム学習以外にも3つのモードが利用可能：

```python
# 固定減衰（線形）
curiosity_decay_method="fixed"

# 訪問カウントベース
curiosity_decay_method="count"

# World Modelの学習進捗に連動
curiosity_decay_method="adaptive"
```
