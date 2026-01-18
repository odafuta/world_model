# Quick Start Guide - MATWM for Simple Tag

このガイドでは、実装の実行方法を簡潔に説明します。

---

## 📋 前提条件

- Python 3.9+
- PyTorch 2.0+
- GPU推奨 (CPUでも動作可能だが遅い)

---

## 🚀 セットアップ (5分)

### 1. 依存関係のインストール

```bash
# PyTorchのインストール (GPUの場合)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# その他の依存関係
pip install numpy matplotlib tqdm pettingzoo[mpe] supersuit
```

### 2. ファイルの確認

以下のファイルが揃っていることを確認:

```
最終課題/
├── matwm_implementation.py       # World Model components
├── matwm_agent.py                 # Agent implementation
├── 2026_MATWM_simple_tag_Implementation.ipynb  # Main notebook
└── (その他のドキュメント)
```

---

## 🎯 実行方法

### 方法1: Jupyter Notebook (推奨)

```bash
# Notebookを開く
jupyter notebook 2026_MATWM_simple_tag_Implementation.ipynb

# セルを順番に実行
# Shift + Enter で実行
```

#### Notebookの構成

1. **セル1-3**: セットアップ・環境確認
2. **セル4-5**: MATWM実装の読み込み
3. **セル6-7**: 訓練ループの定義
4. **セル8-9**: 訓練の実行 ★ここで時間がかかります★
5. **セル10-11**: 可視化
6. **セル12-13**: 評価
7. **セル14**: まとめ

### 方法2: Pythonスクリプト

```python
# train_matwm.py として保存して実行

import torch
from pettingzoo.mpe import simple_tag_v3
from matwm_implementation import MATWMConfig
from matwm_agent import MATWMAgent

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 設定
config = MATWMConfig(
    total_steps=10000,  # テスト用に短く設定
    warmup_steps=1000,
)

# 環境作成
env = simple_tag_v3.parallel_env(
    num_good=1,
    num_adversaries=3,
    num_obstacles=2,
    max_cycles=25,
)
env.reset(seed=42)

# エージェント作成
agents = {}
for idx, name in enumerate(env.agents):
    agents[name] = MATWMAgent(config, name, idx, device)

print("Setup complete! Ready to train.")

# 訓練ループは Notebook を参照
```

---

## ⚙️ 設定のカスタマイズ

### クイックテスト (10-30分)

```python
config = MATWMConfig(
    total_steps=5000,      # 短め
    warmup_steps=500,
    batch_size=8,          # 小さめ
    sequence_length=32,    # 短め
)
```

### 標準訓練 (1-3時間, GPU)

```python
config = MATWMConfig(
    total_steps=50000,
    warmup_steps=1000,
    batch_size=16,
    sequence_length=64,
)
```

### フル訓練 (3-6時間, GPU)

```python
config = MATWMConfig(
    total_steps=100000,
    warmup_steps=1000,
    batch_size=16,
    sequence_length=64,
)
```

---

## 📊 結果の確認

### 訓練中

Progress barで進捗を確認:

```
Training: 45%|████▌    | 45000/100000 [01:23<01:41, 542.34it/s]
Step 45000: adversary_0=-2.35 adversary_1=-1.87 adversary_2=-2.12 agent_0=-8.43
```

### 訓練後

1. **学習曲線**: `results/training_curves.png`
2. **チェックポイント**: `results/matwm_YYYY_MM_DD_HH_MM_SS/checkpoint_*/`
3. **評価結果**: Notebookのセル13で出力

---

## 🔍 主要な可視化

### Episode Rewards

各エージェントの累積報酬の推移:

- **Adversaries** (predators): 徐々に上昇 (0 → +5〜+10)
- **Good Agent** (prey): 徐々に改善 (-10 → -5〜0)

### World Model Loss

- **Total Loss**: 全体の学習損失
- **Teammate Loss** ★: 他エージェント予測の精度

### Actor Loss

方策の学習損失

---

## 💡 トラブルシューティング

### 問題: Out of Memory

**解決策**:

```python
config.batch_size = 8           # 16 → 8
config.sequence_length = 32     # 64 → 32
config.imagination_horizon = 10 # 15 → 10
```

### 問題: 学習が遅い

**解決策**:

- GPUを使用しているか確認: `torch.cuda.is_available()`
- Batch sizeを増やす (メモリに余裕がある場合)
- Warmup期間を短くする

### 問題: 学習が進まない

**解決策**:

- Warmup期間を延ばす: `config.warmup_steps = 2000`
- Learning rateを下げる: `config.learning_rate = 1e-4`
- Teammate weightを調整: `config.teammate_weight = 0.3`

### 問題: Teammate Loss が下がらない

**原因**: 他エージェントの方策がまだ安定していない

**解決策**:

- より長く訓練する
- Teammate weightを下げる
- Warmup期間を延ばす

---

## 📈 期待される性能

### 初期 (0-10K steps)

- ランダムな行動
- Adversaries: 報酬 ~0
- Good Agent: 報酬 ~-10

### 中期 (10K-50K steps)

- 基本的な戦略の学習
- Adversaries: 報酬 +2〜+5
- Good Agent: 報酬 -8〜-5

### 後期 (50K-100K steps)

- 洗練された戦略
- Adversaries: 協調して追跡
- Good Agent: 効率的に逃げる
- Teammate Prediction: 精度向上

---

## 🎓 学習のポイント

### 観察すべき指標

1. **Episode Rewards**: エージェントの性能
2. **Teammate Loss**: 社会的世界モデルの精度
3. **World Model Loss**: 環境モデルの精度
4. **Actor/Critic Loss**: 方策・価値関数の学習

### 重要な概念

1. **Imagination-based Training**: 
   - 実環境1ステップ → 想像15ステップ
   - サンプル効率の鍵

2. **Teammate Predictor**:
   - 社会的世界モデルのコア
   - 他エージェント行動の予測

3. **Prioritized Replay**:
   - 最近の経験を重視
   - 方策変化への追従

---

## 📚 次のステップ

### 1. ベースライン実装の完成 ✅

現在の実装で動作確認

### 2. 性能評価

- ランダム方策との比較
- Model-free手法との比較

### 3. 拡張機能の実装

- γ-Progress Curiosity
- Theory of Mind
- Communication Module

### 4. 論文執筆

- 実装の説明
- 実験結果
- 考察

---

## 📞 サポート

### ドキュメント

- **README.md**: プロジェクト概要
- **PROJECT_STRUCTURE.md**: 構造の詳細
- **IMPLEMENTATION_SUMMARY.md**: 完全な実装ドキュメント

### デバッグ

```python
# デバッグモード
import logging
logging.basicConfig(level=logging.DEBUG)

# 小さいスケールでテスト
config = MATWMConfig(total_steps=1000, warmup_steps=100)
```

---

## ✅ チェックリスト

実行前に確認:

- [ ] Python 3.9+がインストールされている
- [ ] PyTorchがインストールされている
- [ ] PettingZooがインストールされている
- [ ] GPUが利用可能 (オプション)
- [ ] 必要なファイルが揃っている
- [ ] ディスク容量が十分にある (チェックポイント保存用)

実行後に確認:

- [ ] 訓練が完了した
- [ ] 学習曲線が保存された
- [ ] チェックポイントが保存された
- [ ] 評価結果が表示された

---

**準備完了!** 🚀

Notebookを開いて、セルを順番に実行してください。

Good luck with your implementation! 🎉


