# 実行前の最終確認チェックリスト

## ✅ 実行可能な状態の確認

### 1. ディレクトリ保存について

**結論: 変更不要！このまま実行できます。**

```python
# 現在の設定（相対パス）
save_dir='results_gamma_true'  # ✅ これで問題なし
```

**実行時の動作:**
```
~/matwm_project/
├── train_gamma_true.py
├── results_gamma_true/      # ← 自動作成される
│   └── run_20260211_100000/
│       ├── checkpoint_5000/
│       └── final/
```

**確認不要:** ディレクトリは自動的に作成されます。

---

### 2. APIキーについて

**結論: APIキーなしで実行可能！**

| 機能 | APIキー | 状態 |
|------|---------|------|
| 計算型好奇心（Dynamics/Reward/Social） | ❌ 不要 | ✅ 常に有効 |
| LLM意味的好奇心 | ✅ 必要 | ⭕ オプション |

**推奨: APIキーなしで実行**
- 計算型好奇心で十分な性能
- 論文再現に必要な機能はすべて動作
- 追加設定不要

---

## 🚀 実行手順（3ステップ）

### ステップ1: ファイルアップロード

**ローカルPC（コマンドプロンプト）:**

```cmd
cd "C:\Users\0622d\OneDrive - OUMail (Osaka University)\M1_秋冬\松尾研究室\WorldModel\最終課題"

REM GCS バケット作成（初回のみ）
gsutil mb gs://matwm-training-bucket

REM ファイルアップロード
gsutil -m cp *.py requirements.txt gs://matwm-training-bucket/
```

### ステップ2: GCP VMセットアップ

**GCP VMにSSH接続後:**

```bash
# 作業ディレクトリ作成
mkdir -p ~/matwm_project
cd ~/matwm_project

# ファイルダウンロード
gsutil -m cp gs://matwm-training-bucket/*.py .
gsutil cp gs://matwm-training-bucket/requirements.txt .

# パッケージインストール
pip install -r requirements.txt

# GPU確認
nvidia-smi
```

### ステップ3: 実行開始

```bash
# tmuxセッション開始
tmux new -s matwm_true

# 実行（APIキーなし）
python3 train_gamma_true.py 2>&1 | tee train_true.log

# デタッチ: Ctrl+B → D
```

**別のVMで同様に:**
```bash
tmux new -s matwm_false
python3 train_gamma_false.py 2>&1 | tee train_false.log
```

---

## 📋 実行前チェックリスト

### 必須項目
- [ ] GCP VMインスタンス作成済み（H100 × 2台）
- [ ] ファイルをGCSにアップロード済み
- [ ] VMにSSH接続成功
- [ ] GPU認識確認（`nvidia-smi`）
- [ ] ファイルダウンロード完了
- [ ] パッケージインストール完了（`pip install -r requirements.txt`）

### オプション項目（APIキー使用時のみ）
- [ ] OpenRouterアカウント作成
- [ ] APIキー取得
- [ ] 環境変数設定（`export OPENROUTER_API_KEY="..."`）

---

## 🎯 実行パターン

### パターンA: APIキーなし（推奨）

```bash
# 何もしない → そのまま実行
cd ~/matwm_project
python3 train_gamma_true.py 2>&1 | tee train_true.log
```

**出力例:**
```
======================================================================
MATWM + Curiosity-Driven Training (γ-Progress=TRUE)
======================================================================
Device: cuda
  GPU: NVIDIA H100 80GB HBM3
  VRAM: 81.6 GB

use_gamma_progress: True
Total steps: 50000
Warmup steps: 5000

=== Curiosity Configuration ===
  Social Curiosity Weight: 2.0 ★
  Dynamics Curiosity Weight: 1.0
  Decay method: adaptive
  LLM enabled: False  # ← APIキーなし
  Log dir: llm_logs_gamma_true

Shared World Model: 11466783 params (γ-Progress enabled)

=== Initializing Weights ===
✓ Weight initialization complete

=== Starting MATWM + Curiosity Training ===
Save directory: ./results_gamma_true/run_20260211_100000
Total steps: 50000
Warmup steps: 5000
Social curiosity weight: 2.0 ★

Training:   0%|          | 0/50000 [00:00<?, ?it/s]
```

### パターンB: APIキーあり（オプション）

```bash
# 環境変数設定
export OPENROUTER_API_KEY="sk-or-v1-YOUR_API_KEY_HERE"

# 実行
python3 train_gamma_true.py 2>&1 | tee train_true.log
```

**出力例:**
```
=== Curiosity Configuration ===
  Social Curiosity Weight: 2.0 ★
  Dynamics Curiosity Weight: 1.0
  Decay method: adaptive
  LLM enabled: True  # ← APIキーあり
  LLM model: google/gemma-3-4b-it:free
```

---

## 🔍 実行中の確認

### GPU使用率の確認

```bash
# リアルタイムモニタリング
watch -n 1 nvidia-smi

# 期待される出力
+-----------------------------------------------------------------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA H100 80GB    Off  | 00000000:00:04.0 Off |                    0 |
| N/A   65C    P0   450W / 700W |  35000MiB / 81559MiB |     95%      Default |
+-------------------------------+----------------------+----------------------+
```

✅ **GPU使用率が90%以上であれば正常**

### ログの確認

```bash
# リアルタイムでログを表示
tail -f train_true.log

# 最新の進捗を確認
tail -n 50 train_true.log
```

### tmuxセッションへの再接続

```bash
# セッション一覧
tmux ls

# セッションにアタッチ
tmux attach -t matwm_true

# デタッチ: Ctrl+B → D
```

---

## 📊 予想される実行時間とコスト

### H100使用時

| 項目 | 値 |
|------|-----|
| **実行時間** | **11-13時間/台** |
| **時間単価** | **$5.92/時** |
| **1台コスト** | **$71** (約10,650円) |
| **2台同時** | **$142** (約21,300円) |

### 進捗の目安

```
Step 5000/50000 (10%完了) - 約1.2時間経過
Step 10000/50000 (20%完了) - 約2.4時間経過
Step 25000/50000 (50%完了) - 約6時間経過
Step 40000/50000 (80%完了) - 約9.6時間経過
Step 50000/50000 (100%完了) - 約12時間経過
```

---

## 🛑 完了後の処理

### 結果の保存

```bash
# VM上で
cd ~/matwm_project

# GCSにアップロード
gsutil -m cp -r results_gamma_true gs://matwm-training-bucket/
gsutil -m cp -r results_gamma_false gs://matwm-training-bucket/
gsutil -m cp *.log gs://matwm-training-bucket/logs/
```

### ローカルPCにダウンロード

```cmd
REM ローカルPCで
cd "C:\Users\0622d\OneDrive - OUMail (Osaka University)\M1_秋冬\松尾研究室\WorldModel\最終課題"

mkdir gcp_results

gsutil -m cp -r gs://matwm-training-bucket/results_gamma_true gcp_results\
gsutil -m cp -r gs://matwm-training-bucket/results_gamma_false gcp_results\
gsutil -m cp -r gs://matwm-training-bucket/logs gcp_results\
```

### VMインスタンスの削除

```cmd
REM 課金停止
gcloud compute instances delete matwm-gamma-true-h100 --zone=us-central1-a
gcloud compute instances delete matwm-gamma-false-h100 --zone=us-central1-a
```

---

## ⚠️ よくある質問

### Q1: ディレクトリは変更する必要がある？
**A:** いいえ、変更不要です。相対パス（`results_gamma_true/`）で自動作成されます。

### Q2: APIキーは必須？
**A:** いいえ、オプションです。APIキーなしでも完全に動作します。

### Q3: 実行時間はどのくらい？
**A:** H100使用時、約11-13時間/台です。

### Q4: コストはいくら？
**A:** 2台同時実行で約$142（21,300円）です。

### Q5: SSH切断したらどうなる？
**A:** tmuxを使用しているため、実行は継続されます。再接続して`tmux attach -t matwm_true`で確認できます。

---

## 🎯 最終確認

### すべて準備完了？

- [x] ディレクトリ保存: 変更不要（自動作成）
- [x] APIキー: 不要（オプション）
- [ ] GCP VMインスタンス作成
- [ ] ファイルアップロード
- [ ] パッケージインストール
- [ ] 実行開始

**すべて確認できたら実行開始！🚀**

---

## 📚 関連ドキュメント

- **[DIRECTORY_AND_API_SETUP.md](./DIRECTORY_AND_API_SETUP.md)** - ディレクトリとAPIキーの詳細
- **[GCP_EXECUTION_GUIDE.md](./docs/gcp_execution/GCP_EXECUTION_GUIDE.md)** - 完全な実行ガイド
- **[QUICK_START_GCP.md](./docs/gcp_execution/QUICK_START_GCP.md)** - クイックスタート
- **[EXECUTION_FILES_README.md](./EXECUTION_FILES_README.md)** - 実行ファイルの説明

---

**準備完了！実行を開始してください。Good luck! 🚀**
