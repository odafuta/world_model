# GCP実行前チェックリスト

## ✅ 実行前に必ず確認

### 1. GCP準備
- [ ] GCPアカウント作成済み
- [ ] プロジェクト作成済み
- [ ] 支払い方法設定済み
- [ ] Compute Engine API有効化済み
- [ ] H100 GPU割り当てクォータ確認（最低2個必要）

### 2. ローカルPC準備
- [ ] Google Cloud SDK インストール済み
- [ ] `gcloud auth login` 実行済み
- [ ] プロジェクトID設定済み (`gcloud config set project YOUR_PROJECT_ID`)
- [ ] 必要なファイルが揃っている:
  - [ ] `train_gamma_true.py`
  - [ ] `train_gamma_false.py`
  - [ ] `matwm_implementation.py`
  - [ ] `matwm_agent.py`
  - [ ] `matwm_utils.py`
  - [ ] `curiosity_reward.py`
  - [ ] `requirements.txt`

### 3. Cloud Storage準備
- [ ] バケット作成済み (`gsutil mb gs://matwm-training-bucket`)
- [ ] ファイルアップロード完了
- [ ] アップロード確認済み (`gsutil ls gs://matwm-training-bucket/`)

### 4. 予算確認
- [ ] 予算: 約$150 (22,500円) 確保済み
- [ ] 実行時間: 約13時間（準備+実行+結果取得）確保済み

---

## 💰 コスト見積もり確認

| 項目 | 金額 |
|------|------|
| インスタンス1 (gamma_true) | $71 |
| インスタンス2 (gamma_false) | $71 |
| Cloud Storage (1ヶ月) | $5 |
| **合計** | **$147** (約22,050円) |

---

## ⏱️ タイムライン

| フェーズ | 時間 |
|---------|------|
| 準備（ファイルアップロード、インスタンス作成） | 30分 |
| セットアップ（パッケージインストール） | 10分 |
| 実行（トレーニング） | 12時間 |
| 結果取得（ダウンロード） | 30分 |
| **合計** | **約13時間** |

---

## 🚨 重要な注意事項

### 実行中
- [ ] SSH接続が切れてもトレーニングは継続（tmux使用）
- [ ] 定期的に進捗確認（1-2時間ごと推奨）
- [ ] GPU使用率が90%以上であることを確認

### 完了後
- [ ] **必ず結果をCloud Storageにアップロード**
- [ ] **ローカルPCにダウンロード完了後、VMインスタンス削除**
- [ ] 削除しないと課金が継続（$5.92/時）

### トラブル時
- [ ] ログファイルを確認 (`tail -f train_true.log`)
- [ ] GPU状態を確認 (`nvidia-smi`)
- [ ] tmuxセッションに再接続 (`tmux attach -t matwm_true`)

---

## 📞 緊急時の連絡先

- GCPサポート: https://cloud.google.com/support
- 課金確認: https://console.cloud.google.com/billing

---

## 🎯 実行開始前の最終確認

すべてのチェックボックスにチェックが入っていますか？

- [ ] はい、すべて確認済み → **実行開始！**
- [ ] いいえ → 不足している項目を完了してください

---

**準備が整ったら `QUICK_START_GCP.md` または `GCP_EXECUTION_GUIDE.md` に従って実行を開始してください。**
