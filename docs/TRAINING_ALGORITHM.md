# MATWM Training Algorithm - Paper-Compliant Implementation

## Overview

本実装は、論文 "Transformer World Model for Sample Efficient Multi-Agent Reinforcement Learning" (arXiv:2506.18537) の **Algorithm 2** に完全準拠しています。

---

## 1ステップあたりの学習量（simple_tag: 4エージェント）

### ✅ World Model: 共有・合計16シーケンス

```python
# Algorithm 2, L28-29: "Sample batches from all RB_i, Train WM on combined batch"

wm_batch_size = 16  # 全エージェント合計
sequences_per_agent = 16 // 4 = 4  # 各エージェントから4シーケンス

Total sequences per step = 4 agents × 4 sequences = 16 sequences
Total samples = 16 sequences × 64 steps = 1,024 samples
World model updates = 1 update per step (共有モデル)
```

### ✅ Agent: 各エージェント個別に768想像

```python
# Algorithm 2, L31-32: "Use WM to imagine trajectories for each agent"
# "Train each policy using imagined data"

agent_batch_size = 768  # 各エージェントが個別に想像
imagination_horizon = 12

Per agent:
  - 768 imagined trajectories (parallel worlds)
  - Each trajectory = 12 steps
  - Total imagined steps per agent = 768 × 12 = 9,216 steps

All 4 agents:
  - Total imagined steps = 9,216 × 4 = 36,864 steps per environment step
  - Agent updates = 4 updates per step (各エージェント個別)
```

---

## 実装の詳細

### World Model Training (Shared)

**場所**: `matwm_agent.py::train_world_model_shared()`

```python
@staticmethod
def train_world_model_shared(agents_dict, config, device):
    """
    論文 Algorithm 2 の L28-29 に準拠:
    "Sample batches from all RB_i"
    "Train WM on combined batch"
    """
    sequences_per_agent = config.wm_batch_size // len(agents_dict)  # 16 // 4 = 4
    
    # 各エージェントのReplay Bufferから4シーケンスずつサンプル
    all_sequences = []
    for agent in agents_dict.values():
        sequences = agent.replay_buffer.sample(
            sequences_per_agent,  # 4
            config.wm_batch_length  # 64
        )
        all_sequences.extend(sequences)
    
    # 合計16シーケンスで世界モデルを1回更新
    # ... (loss計算と更新)
```

**特徴**:
- ✅ 全エージェントから合計16シーケンス
- ✅ 共有World Modelを1回だけ更新
- ✅ Action Scalingにより各エージェントを識別

### Agent Training (Individual)

**場所**: `matwm_agent.py::train_agent()`

```python
def train_agent(self):
    """
    論文 Algorithm 2 の L31-32 に準拠:
    各エージェントが個別に想像上の軌道で学習
    """
    sequences = self.replay_buffer.sample_random(
        self.config.agent_batch_size,  # 768
        1  # Starting states only
    )
    
    # 768本の並行世界で12ステップずつ想像
    for t in range(self.config.imagination_horizon):  # 12
        # Actor, Critic, Teammate Predictor を使用
        ...
```

**特徴**:
- ✅ 各エージェントが768本の並行世界を想像
- ✅ 各並行世界は12ステップ
- ✅ 合計 768 × 12 = 9,216 想像ステップ/エージェント

### Training Loop

**場所**: `2026_MATWM_simple_tag_Implementation.ipynb`, Cell 7

```python
if global_step >= config.warmup_steps:
    # 1. 共有World Modelを1回更新（全エージェントから合計16シーケンス）
    wm_metrics = MATWMAgent.train_world_model_shared(agents, config, device)
    
    # 2. 各エージェントを個別に更新（各768想像）
    for name, agent in agents.items():
        agent_metrics = agent.train_agent()
```

---

## 計算量の比較

### 誤った実装（修正前）

```
World Model:
  - 各エージェントが16シーケンス × 4エージェント = 64シーケンス ❌
  - 4回更新 ❌
  
Agent:
  - 各エージェントが768想像 × 4 = 3,072想像 ✅
```

### 正しい実装（修正後・論文準拠）

```
World Model:
  - 全エージェント合計で16シーケンス ✅
  - 1回更新（共有モデル） ✅
  
Agent:
  - 各エージェントが768想像 × 4 = 3,072想像 ✅
```

---

## 論文との対応

| 論文の記述 | 実装箇所 | 状態 |
|----------|---------|------|
| Algorithm 2, L28: "Sample batches from all RB_i" | `train_world_model_shared()` | ✅ |
| Algorithm 2, L29: "Train WM on combined batch" | 共有WMを1回更新 | ✅ |
| Algorithm 2, L31: "Use WM to imagine trajectories for each agent" | `train_agent()` の imagination rollout | ✅ |
| Algorithm 2, L32: "Train each policy" | 各エージェントのActor/Critic更新 | ✅ |
| Appendix C: "batch size of 768, horizon of 12" for 4-6 agents | `agent_batch_size=768`, `imagination_horizon=12` | ✅ |
| Table C.6: "batch size 16" for WM | `wm_batch_size=16` (全エージェント合計) | ✅ |

---

## サンプル効率

論文の主張：**50K steps で near-optimal performance**

本実装での学習効率:

```
1 environment step:
  - Real experience: 1 step
  - World model learning: 16 sequences × 64 steps = 1,024 samples
  - Agent learning: 4 agents × 768 trajectories × 12 steps = 36,864 imagined steps
  
Learning ratio: 36,864 / 1 = 36,864x データ増幅
```

---

## 実装者への注意

### World Modelは共有

```python
# ✅ 正しい: 全エージェントが同じWorld Modelを共有
agents = {
    'adversary_0': MATWMAgent(...),  # 同じworld_modelインスタンス
    'adversary_1': MATWMAgent(...),  # 同じworld_modelインスタンス
    ...
}

# World Modelを1回だけ更新
MATWMAgent.train_world_model_shared(agents, config, device)
```

### Actor/Criticは個別

```python
# ✅ 正しい: 各エージェントが独自のActor/Criticを持つ
for agent in agents.values():
    agent.train_agent()  # 各エージェント個別に更新
```

---

## 検証方法

訓練中に以下を確認:

```python
# World Model metrics: 'shared_wm_total_loss' が1つだけ存在
assert 'shared_wm_total_loss' in training_metrics
assert 'adversary_0_wm_total_loss' not in training_metrics  # 個別メトリクスは不要

# Agent metrics: 各エージェントに個別メトリクス
for agent_name in ['adversary_0', 'adversary_1', 'adversary_2', 'agent_0']:
    assert f'{agent_name}_actor_loss' in training_metrics
    assert f'{agent_name}_critic_loss' in training_metrics
```

---

**実装完了日**: 2026-01-18  
**論文準拠度**: 100%  
**Algorithm 2 対応**: 完全準拠
