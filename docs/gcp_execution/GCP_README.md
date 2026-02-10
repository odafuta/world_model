# GCP H100 実行ガイド - 完全版

## 📚 ドキュメント一覧

### 🚀 実行ガイド（優先順位順）

1. **[PRE_EXECUTION_CHECKLIST.md](./PRE_EXECUTION_CHECKLIST.md)** ⭐ 最初に読む
   - 実行前のチェックリスト
   - 必要な準備の確認
   - コストと時間の見積もり

2. **[QUICK_START_GCP.md](./QUICK_START_GCP.md)** ⭐⭐ クイックスタート
   - 5ステップで実行開始
   - 最小限の説明で素早く開始
   - 推奨: 時間がない場合

3. **[GCP_EXECUTION_GUIDE.md](./GCP_EXECUTION_GUIDE.md)** ⭐⭐⭐ 詳細ガイド
   - 完全なステップバイステップガイド
   - トラブルシューティング付き
   - 推奨: 初めてGCPを使う場合

### 📦 必要なファイル

- `requirements.txt` - 依存パッケージ一覧
- `train_gamma_true.py` - use_gamma_progress=True版
- `train_gamma_false.py` - use_gamma_progress=False版
- `matwm_implementation.py` - MATWM実装
- `matwm_agent.py` - エージェント実装
- `matwm_utils.py` - ユーティリティ関数
- `curiosity_reward.py` - 好奇心報酬計算

---

## ⏱️ 実行時間とコスト（H100使用）

### 構成
- **GPU**: NVIDIA H100 80GB × 1
- **CPU**: n1-standard-16 (16 vCPU, 60GB RAM)
- **訓練**: total_steps=50000（論文再現）

### 予想
| 項目 | 値 |
|------|-----|
| **実行時間** | **11-13時間/台** |
| **時間単価** | **$5.92/時** |
| **1台コスト** | **$71** (約10,650円) |
| **2台同時** | **$142** (約21,300円) |

---

## 🎯 実行フロー

```
1. 準備 (30分)
   ├─ GCP設定
   ├─ ファイルアップロード
   └─ インスタンス作成

2. セットアップ (10分)
   ├─ SSH接続
   ├─ ファイルダウンロード
   └─ パッケージインストール

3. 実行 (12時間)
   ├─ tmuxセッション開始
   ├─ トレーニング実行
   └─ 定期的な進捗確認

4. 結果取得 (30分)
   ├─ GCSにアップロード
   ├─ ローカルにダウンロード
   └─ インスタンス削除

合計: 約13時間
```

---

## 🚀 クイックスタート（5ステップ）

### 1. ファイルアップロード
```cmd
gsutil -m cp *.py requirements.txt gs://matwm-training-bucket/
```

### 2. VMインスタンス作成
- GCPコンソールで H100 × n1-standard-16 を2台作成

### 3. セットアップ
```bash
mkdir -p ~/matwm_project && cd ~/matwm_project
gsutil -m cp gs://matwm-training-bucket/*.py .
gsutil cp gs://matwm-training-bucket/requirements.txt .
pip install -r requirements.txt
```

### 4. 実行
```bash
tmux new -s matwm_true
python3 train_gamma_true.py 2>&1 | tee train_true.log
# Ctrl+B → D
```

### 5. 結果取得
```bash
gsutil -m cp -r results_gamma_true gs://matwm-training-bucket/
```

---

## 📊 実行結果の構成

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

## ⚠️ 重要な注意事項

### 実行前
- ✅ H100 GPU割り当てクォータを確認（最低2個必要）
- ✅ 予算確保: 約$150 (22,500円)
- ✅ 時間確保: 約13時間

### 実行中
- ✅ tmux使用でSSH切断しても継続
- ✅ GPU使用率90%以上を確認
- ✅ 1-2時間ごとに進捗確認

### 完了後
- ✅ **必ず結果をGCSにアップロード**
- ✅ **ローカルにダウンロード後、VM削除**
- ✅ 削除しないと課金継続（$5.92/時）

---

## 🔧 トラブルシューティング

### GPU割り当てクォータ不足
```
GCPコンソール → IAMと管理 → 割り当て → "NVIDIA H100" で検索
→ 割り当てを編集 → 新しい上限: 2
```

### SSH切断後の再接続
```bash
gcloud compute ssh matwm-gamma-true-h100 --zone=us-central1-a
tmux attach -t matwm_true
```

### CUDA Out of Memory
```python
# train_gamma_true.py で調整
config = MATWMConfig(
    wm_batch_size=16,  # デフォルト32から削減
    ac_batch_size=128,  # デフォルト256から削減
)
```

---

## 📞 サポート

### ドキュメント
- [GCP Compute Engine](https://cloud.google.com/compute/docs)
- [NVIDIA H100 仕様](https://www.nvidia.com/en-us/data-center/h100/)
- [PyTorch ドキュメント](https://pytorch.org/docs/stable/index.html)

### 確認コマンド
```bash
# GPU確認
nvidia-smi

# 進捗確認
tail -f train_true.log

# ディスク確認
df -h

# メモリ確認
free -h
```

---

## ✅ 実行チェックリスト

- [ ] PRE_EXECUTION_CHECKLIST.md を確認
- [ ] ファイルをGCSにアップロード
- [ ] VMインスタンス2台作成
- [ ] セットアップ完了
- [ ] トレーニング実行開始
- [ ] 定期的に進捗確認
- [ ] 結果をGCSにアップロード
- [ ] ローカルにダウンロード
- [ ] VMインスタンス削除

---

## 🎯 次のステップ

1. **[PRE_EXECUTION_CHECKLIST.md](./PRE_EXECUTION_CHECKLIST.md)** を開く
2. すべての項目を確認
3. **[QUICK_START_GCP.md](./QUICK_START_GCP.md)** または **[GCP_EXECUTION_GUIDE.md](./GCP_EXECUTION_GUIDE.md)** に従って実行

**Good luck! 🚀**
