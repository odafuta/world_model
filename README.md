# 最終課題: 社会的世界モデルの実装

## テーマ
**「他のエージェントの行動」を予測する社会的世界モデル**

マルチエージェント環境において、他のエージェントの行動を予測することで、より効率的な協調・競争を実現する世界モデルを実装します。

## 外部環境
**PettingZoo の Simple Tag**

- Predator-Prey タスク
- 3 adversaries (predators) vs 1 good agent (prey)
- 部分観測可能環境
- 離散行動空間

## ベースモデル

### 1. MATWM: Multi-Agent Transformer World Model ★メイン★

**論文**: Deihim et al. (2025), "Transformer World Model for Sample Efficient Multi-Agent Reinforcement Learning", arXiv:2506.18537

#### MATWMとは

単一エージェントで高い性能を示したSTORMモデルを基盤としつつ、複数のエージェントが協調・競争する複雑な環境において、**「想像（Imagination）」**を通じて学習を劇的に効率化した**マルチエージェント強化学習（MARL）**における**サンプル効率**を劇的に改善するための新しい**Transformerベースのワールドモデル**
    
#### 主要コンポーネント

1. **Categorical VAE** (Encoder/Decoder)
   - 観測を離散潜在空間にエンコード
   
2. **Transformer-based Dynamics Model**
   - 次状態の予測
   
3. **Teammate Predictor ★社会的世界モデルのコア★**
   - 他エージェントの行動を予測
   - 非定常性の軽減
   - 協調・競争行動の改善
   
4. **Prioritized Replay Buffer**
   - 最近の経験を重視
   - 方策変化に追従

#### Simple Tag環境への適用性

**適切な理由**:

1. **マルチエージェント**: 4エージェント間の相互作用が重要
2. **部分観測**: 他エージェントの予測が価値を持つ
3. **協調・競争**: Adversaries同士の協調、Good agentとの競争
4. **サンプル効率**: 環境がシンプルなため、MATWMの効果を明確に評価可能

**懸念点への対応**:

- **懸念点1** (被引用数少ない): 
  - 2025年6月投稿で最新の研究
  - STORM, Dreamer V3など確立された手法を基盤としている
  - 教育目的での実装には十分な価値がある
  
- **懸念点2** (Simple Tagへの適用):
  - MATWMは元論文でPettingZoo環境でも評価されている
  - Simple Tagは典型的なPettingZoo タスク
  - 実装を通じて適用性を検証することが本課題の目的

### 2. Active World Model Learning with Progress Curiosity (拡張案)

**論文**: [Active World Model Learning with Progress Curiosity]

好奇心駆動型探索により、世界モデルの学習を効率化:

- **γ-progress**: 予測誤差の改善度を内発的報酬として利用
- 探索の改善
- より早い収束

**実装方針**: まずMATWMのベースライン実装を完成させ、その後γ-progressを追加拡張として実装

---

## 実装済みファイル

### コアファイル

1. **PROJECT_STRUCTURE.md**: プロジェクト構造の詳細説明
2. **IMPLEMENTATION_SUMMARY.md**: 実装の完全なドキュメント
3. **matwm_implementation.py**: World Modelコンポーネント
4. **matwm_agent.py**: 完全なエージェント実装
5. **curiosity_reward.py**: 好奇心駆動型報酬とγ-Progress実装

### 実験用Notebook

実験は `experiments/` ディレクトリに配置され、各メソッドごとに分類されています：

- **experiments/llm_curiosity_and_γ-progress/**: LLM意味的好奇心 + γ-Progress
- **experiments/only_llm_curiosity/**: LLM意味的好奇心のみ
- **experiments/only_γ_progress/**: γ-Progressのみ

各ディレクトリには以下のNotebookがあります：
- フル訓練版（50,000ステップ）: `2026_MATWM_simple_tag_Implementation_*.ipynb`
- テスト版（2,000ステップ）: `*_test.ipynb`

詳細は `experiments/README.md` を参照してください。

### 参考資料

- **simple_tag.md**: Simple Tag環境の詳細仕様
- **論文/md/**: 参考論文のマークダウン版
- **docs/**: 詳細なドキュメント群

---

## 実装の特徴

### 1. Teammate Predictor (社会的世界モデルのコア)

他エージェントの行動を予測するモジュール:

```python
class TeammatePredictor(nn.Module):
    """Predict other agents' actions from focal agent's latent state"""
    # 各エージェント用の独立した予測器
    # 協調・競争行動の予測が可能
```

### 2. Prioritized Replay Buffer

最近の経験を重視:

```python
# Exponential decay (0.995 per step)
# 方策変化に追従
```

### 3. Imagination-based Training

実環境1ステップ → 想像15ステップ:

```python
# サンプル効率: 15倍の学習データ
# 目標: 100K steps で収束
```

### 4. Action Scaling

エージェント識別:

```python
scaled_action = action + agent_idx * action_dim
# Agent 0: 0-4, Agent 1: 5-9, ...
```

---

## 使用方法

### セットアップ

```bash
pip install torch numpy matplotlib tqdm
pip install pettingzoo[mpe] supersuit
pip install jupyter jupyterlab ipykernel
```

### 訓練の実行

#### ローカル環境での実行

```bash
# Jupyter Labを起動
jupyter lab

# ブラウザで experiments/ ディレクトリを開き、実行したい実験のNotebookを選択
# 例:
# - experiments/llm_curiosity_and_γ-progress/2026_MATWM_simple_tag_Implementation_gamma_true.ipynb
# - experiments/only_llm_curiosity/2026_MATWM_simple_tag_Implementation_gamma_false.ipynb
# - experiments/only_γ_progress/2026_MATWM_simple_tag_Implementation_oly_gamma_true.ipynb
```

#### GCP H100での実行（推奨）

大規模な訓練には、GCP H100 GPUの使用を推奨します：

1. **セットアップガイド**: `docs/gcp_execution/GCP_EXECUTION_GUIDE.md`
2. **クイックスタート**: `docs/gcp_execution/QUICK_START_GCP.md`

詳細な手順とコスト見積もりについては、上記ドキュメントを参照してください。

#### 出力ディレクトリ構造

Notebook実行時、結果は以下のディレクトリに保存されます：

```
llm_logs/{method}/{timestamp}/
results/{method}/{timestamp}/
```

- `{method}`: 実験メソッド名（`llm_and_gamma`, `only_llm`, `only_gamma`）
- `{timestamp}`: 実行開始時刻（例: `20260211_100030`）

これにより、複数の実行結果が混同せず、系統的に管理できます。

---

## 期待される結果

- **サンプル効率**: 50K-100K steps で収束 (従来の1/10)
- **協調行動**: Adversaries が協調して good agent を追跡
- **競争行動**: Good agent が adversaries の動きを予測して逃げる
- **Teammate Prediction**: 他エージェント行動予測の精度向上

---

## 実装済み機能

1. ✅ MATWM ベースライン実装
2. ✅ γ-Progress Curiosity の導入
3. ✅ LLM意味的好奇心の実装
4. ✅ 計算型好奇心（Dynamics, Reward, Social）
5. ✅ 複数実験メソッドのNotebook実装

## 今後の拡張（オプション）

1. ⬜ Theory of Mind の強化
2. ⬜ Communication Module
3. ⬜ 階層的プランニング

---

## 参考文献

1. Deihim, A., Alonso, E., & Apostolopoulou, D. (2025). Transformer World Model for Sample Efficient Multi-Agent Reinforcement Learning. arXiv:2506.18537.

2. Hafner, D., et al. (2023). Mastering Diverse Domains through World Models (Dreamer V3).

3. Terry, J. K., et al. (2021). PettingZoo: Gym for Multi-Agent Reinforcement Learning.

---

**実装完了!** 🎉
