# 重み初期化の説明

## 質問1: dummy_agentだけ初期化しても意味がないのでは？

**その通りです！** 現在のCell 7（アーキテクチャ確認セル）では、`dummy_agent`のみを初期化していますが、これは**確認用のダミーエージェント**であり、実際の訓練では使用されません。

### 現状の問題

```python
# Cell 7: アーキテクチャ確認（確認用のみ）
dummy_agent = MATWMAgent(config, 'adversary_0', 0, device, shared_world_model=shared_world_model)
initialize_matwm_weights(shared_world_model, dummy_agent.actor, dummy_agent.critic)
# ↑ このdummy_agentは訓練では使われない！
```

### 正しい実装

**訓練関数内で重み初期化を行う必要があります。**

#### 修正箇所

`train_matwm()` 関数内で、エージェント作成後、訓練開始前に初期化：

```python
def train_matwm(config, device, run_dir='./runs', resume_from=None):
    # ...
    
    # Create shared world model (論文通り: 全エージェントで1つを共有)
    shared_world_model, shared_wm_optimizer = MATWMAgent.create_shared_world_model(config, device)
    print(f'Created shared world model with {sum(p.numel() for p in shared_world_model.parameters())} parameters')
    
    # Create agents with shared world model
    agents = {}
    for idx, name in enumerate(agent_names):
        agents[name] = MATWMAgent(config, name, idx, device, shared_world_model=shared_world_model)
    
    # ★★★ 重み初期化をここに追加 ★★★
    # Resume from checkpointの前に実行（新規訓練の場合のみ）
    if resume_from is None:
        print('\n=== Initializing Weights ===')
        # 共有World Modelを初期化
        shared_world_model.apply(_init_weights)  # または initialize_matwm_weights の内部処理
        # 各エージェントのActor/Criticを初期化
        for name, agent in agents.items():
            agent.actor.apply(_init_weights)
            agent.critic.apply(_init_weights)
        print('✓ Weight initialization complete for all agents')
    
    # Training metrics
    episode_rewards = {name: [] for name in agent_names}
    training_metrics = defaultdict(list)
    start_step = 0
    
    # Resume from checkpoint if provided
    if resume_from is not None and os.path.exists(resume_from):
        print(f'\n=== Resuming from checkpoint: {resume_from} ===')
        # チェックポイントから読み込み（重みを上書き）
        episode_rewards, training_metrics, start_step = load_full_checkpoint(
            agents, shared_world_model, shared_wm_optimizer, resume_from, device
        )
        print(f'✓ Resumed from step {start_step}')
    
    # ... 訓練ループ ...
```

### なぜこの順序が重要か

1. **新規訓練**: `resume_from=None`
   - エージェント作成 → **重み初期化** → 訓練開始
   
2. **訓練再開**: `resume_from`が指定されている
   - エージェント作成 → (初期化スキップ) → **チェックポイントから重み読み込み** → 訓練再開

### `_init_weights`関数の定義

`matwm_utils.py`から`_init_weights`をインポートする必要があります：

```python
# Notebookのインポートセクション
from matwm_utils import (
    initialize_matwm_weights,
    _init_weights,  # ← これを追加
    ...
)
```

または、訓練関数内で直接使用：

```python
from matwm_utils import _init_weights

# 訓練関数内
if resume_from is None:
    print('\n=== Initializing Weights ===')
    shared_world_model.apply(_init_weights)
    for name, agent in agents.items():
        agent.actor.apply(_init_weights)
        agent.critic.apply(_init_weights)
    print('✓ Weight initialization complete')
```

---

## 質問2: torchinfo の summary が表示されないのは正常？

**いいえ、正常ではありません。** `torchinfo`がインストールされていないか、エラーが発生しています。

### 原因

Cell 7の冒頭に以下があります：

```python
# 必要に応じてインストール
%pip install torchinfo
```

しかし、これは**コメントアウトされている**ため、実際にはインストールされていません。

### 解決策

#### Option 1: Cell 7を修正（推奨）

```python
# torchinfo のインストール
%pip install torchinfo

# 以降のコードは変更なし...
```

#### Option 2: 別セルで事前インストール

新しいセルを作成：

```python
%pip install torchinfo
```

その後、カーネルを再起動してCell 7を再実行。

### 期待される出力（torchinfo正常時）

```
[4] Detailed Model Summary (torchinfo)
----------------------------------------------------------------------

--- World Model Encoder ---
==========================================================================================
Layer (type:depth-idx)                   Input Shape          Output Shape         Param #
==========================================================================================
Encoder                                  [1, 16]              [1, 32, 32]          --
├─Sequential: 1-1                        [1, 16]              [1, 1024]            --
│    └─Linear: 2-1                       [1, 16]              [1, 512]             8,704
│    └─LayerNorm: 2-2                    [1, 512]             [1, 512]             1,024
│    └─ReLU: 2-3                         [1, 512]             [1, 512]             --
│    └─Linear: 2-4                       [1, 512]             [1, 512]             262,656
│    └─LayerNorm: 2-5                    [1, 512]             [1, 512]             1,024
│    └─ReLU: 2-6                         [1, 512]             [1, 512]             --
│    └─Linear: 2-7                       [1, 512]             [1, 512]             262,656
│    └─LayerNorm: 2-8                    [1, 512]             [1, 512]             1,024
│    └─ReLU: 2-9                         [1, 512]             [1, 512]             --
│    └─Linear: 2-10                      [1, 512]             [1, 1024]            525,312
==========================================================================================
Total params: 1,062,400
Trainable params: 1,062,400
Non-trainable params: 0
Total mult-adds (M): 1.06
==========================================================================================

--- World Model Dynamics (Transformer) ---
(Transformer の詳細な summary)

--- Actor Network ---
(Actor の詳細な summary)

--- Critic Network ---
(Critic の詳細な summary)
```

---

## まとめ

### 修正が必要な箇所

1. **✓ 修正済み**: `matwm_utils.py` の `inspect_matwm_architecture()`
   - `config.obs_dims['adversary_0']` → `config.max_obs_dim`

2. **要修正**: 訓練関数 (`train_matwm()`)
   - エージェント作成後、訓練開始前に重み初期化を追加
   - `if resume_from is None:` の条件下で実行

3. **要修正**: Cell 7
   - `%pip install torchinfo` のコメントアウトを解除
   - または、コメントを明確にして、torchinfo なしでも動作することを示す

### 次のステップ

1. **Cell 2を修正して torchinfo をインストール**（すべてのセルの前に実行）
   ```python
   %pip install torchinfo
   ```

2. **訓練関数に重み初期化を追加**（コード修正が必要）

3. **Notebook全体を再実行**して動作確認

これで、正しく重み初期化が行われ、詳細なアーキテクチャサマリーも表示されるようになります。
