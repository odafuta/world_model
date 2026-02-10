# 実行ファイル一覧

## ✅ 実行可能なファイル（修正済み）

### 本番用（論文再現設定: total_steps=50000）

1. **`train_gamma_true.py`** ✅
   - use_gamma_progress=True
   - total_steps=50000
   - warmup_steps=5000
   - 実行時間: 約11-13時間（H100使用時）
   - 保存先: `results_gamma_true/`

2. **`train_gamma_false.py`** ✅
   - use_gamma_progress=False
   - total_steps=50000
   - warmup_steps=5000
   - 実行時間: 約11-13時間（H100使用時）
   - 保存先: `results_gamma_false/`

### テスト用（短縮版: total_steps=2000）

3. **`train_gamma_true_test.py`** ✅
   - use_gamma_progress=True
   - total_steps=2000
   - warmup_steps=200
   - 実行時間: 約30-40分（H100使用時）
   - 保存先: `results_gamma_true_test/`

4. **`train_gamma_false_test.py`** ✅
   - use_gamma_progress=False
   - total_steps=2000
   - warmup_steps=200
   - 実行時間: 約30-40分（H100使用時）
   - 保存先: `results_gamma_false_test/`

---

## 🔧 修正内容

### 削除したJupyter Notebook特有のコード

1. **`get_ipython().run_line_magic()`**
   - Jupyter特有のマジックコマンド
   - 削除して通常のimport文に変更

2. **`# In[4]:`**
   - セル番号のコメント
   - すべて削除

3. **不要な環境テストコード**
   - 環境仕様の確認コード
   - GPU確認コード
   - アーキテクチャ検証コード
   - 削除して本番実行に最適化

### 追加した機能

1. **matplotlib設定**
   ```python
   matplotlib.use('Agg')  # GUIなし環境用
   ```

2. **明確な設定表示**
   ```python
   print('=' * 70)
   print('MATWM + Curiosity-Driven Training (γ-Progress=TRUE)')
   print('=' * 70)
   ```

3. **`if __name__ == '__main__':`**
   - モジュールとしてインポートされた場合の実行を防止

---

## 📊 ファイル比較

| ファイル | total_steps | warmup_steps | use_gamma_progress | 実行時間 |
|---------|-------------|--------------|-------------------|---------|
| train_gamma_true.py | 50000 | 5000 | True | 11-13時間 |
| train_gamma_false.py | 50000 | 5000 | False | 11-13時間 |
| train_gamma_true_test.py | 2000 | 200 | True | 30-40分 |
| train_gamma_false_test.py | 2000 | 200 | False | 30-40分 |

---

## 🚀 実行方法

### ローカルでテスト実行

```bash
# テスト版（短時間）
python train_gamma_true_test.py
python train_gamma_false_test.py
```

### GCPで本番実行

```bash
# 本番版（論文再現）
python3 train_gamma_true.py 2>&1 | tee train_true.log
python3 train_gamma_false.py 2>&1 | tee train_false.log
```

---

## 📦 依存関係

すべてのファイルは `requirements.txt` に記載されたパッケージに依存:

```bash
pip install -r requirements.txt
```

必要なパッケージ:
- torch>=2.0.0
- numpy>=1.24.0
- pettingzoo>=1.24.0
- gymnasium>=0.29.0
- supersuit>=3.9.0
- matplotlib>=3.7.0
- tqdm>=4.65.0
- pygame>=2.5.0
- requests>=2.31.0
- torchinfo>=1.8.0

---

## ✅ 実行前の確認

### ローカルテスト
```bash
# 構文チェック
python -m py_compile train_gamma_true.py
python -m py_compile train_gamma_false.py

# インポートチェック
python -c "from train_gamma_true import *"
```

### GCP実行前
```bash
# GPU確認
nvidia-smi

# パッケージ確認
pip list | grep -E "torch|pettingzoo|gymnasium"

# ファイル確認
ls -lh train_gamma_*.py
```

---

## 🎯 推奨実行順序

### 1. ローカルでテスト（オプション）
```bash
python train_gamma_true_test.py
```
→ 動作確認（30-40分）

### 2. GCPでテスト実行
```bash
python3 train_gamma_true_test.py 2>&1 | tee test.log
```
→ GCP環境での動作確認（30-40分）

### 3. GCPで本番実行
```bash
# インスタンス1
tmux new -s matwm_true
python3 train_gamma_true.py 2>&1 | tee train_true.log

# インスタンス2
tmux new -s matwm_false
python3 train_gamma_false.py 2>&1 | tee train_false.log
```
→ 論文再現実験（11-13時間）

---

## 📝 出力ファイル

### 実行中
- `train_true.log` - 実行ログ
- `train_false.log` - 実行ログ

### 実行後
```
results_gamma_true/
└── run_20260211_100000/
    ├── checkpoint_5000/
    │   ├── adversary_0.pt
    │   ├── adversary_1.pt
    │   ├── adversary_2.pt
    │   ├── agent_0.pt
    │   └── full_checkpoint.pt
    ├── checkpoint_10000/
    ├── ...
    └── final/
        └── full_checkpoint.pt
```

---

## ⚠️ トラブルシューティング

### エラー: `ModuleNotFoundError`
```bash
# 解決策
pip install -r requirements.txt
```

### エラー: `CUDA out of memory`
```python
# train_*.py の設定を調整
config = MATWMConfig(
    wm_batch_size=16,  # デフォルト32から削減
    ac_batch_size=128,  # デフォルト256から削減
)
```

### エラー: `get_ipython() not defined`
→ ファイルが正しく修正されていません。このREADMEに記載のファイルを使用してください。

---

## 📞 サポート

詳細な実行手順は以下を参照:
- **[GCP_EXECUTION_GUIDE.md](./docs/gcp_execution/GCP_EXECUTION_GUIDE.md)** - 完全ガイド
- **[QUICK_START_GCP.md](./docs/gcp_execution/QUICK_START_GCP.md)** - クイックスタート
- **[PRE_EXECUTION_CHECKLIST.md](./docs/gcp_execution/PRE_EXECUTION_CHECKLIST.md)** - 実行前チェックリスト

---

**すべてのファイルは実行可能な状態です。Good luck! 🚀**
