# GCP H100 クイックスタートガイド

## ⏱️ 実行時間とコスト（H100使用）

| 項目 | 値 |
|------|-----|
| **実行時間** | **11-13時間/台** |
| **コスト** | **$71/台** (約10,650円) |
| **2台同時** | **$142** (約21,300円) |

---

## 🚀 5ステップで実行開始

### ステップ1: ファイルアップロード（ローカルPC）

```cmd
cd "C:\Users\0622d\OneDrive - OUMail (Osaka University)\M1_秋冬\松尾研究室\WorldModel\最終課題"

gcloud auth login
gsutil mb gs://matwm-training-bucket
gsutil -m cp *.py requirements.txt gs://matwm-training-bucket/
```

### ステップ2: VMインスタンス作成（GCPコンソール）

1. https://console.cloud.google.com/ → **Compute Engine** → **CREATE INSTANCE**

2. 設定:
   ```
   名前: matwm-gamma-true-h100
   リージョン: us-central1-a
   マシン: n1-standard-16 (16 vCPU, 60GB RAM)
   GPU: NVIDIA H100 80GB × 1
   OS: Deep Learning on Linux (Debian 11, CUDA 11.8)
   ディスク: 200GB
   ```

3. **作成** をクリック

4. 同じ手順で2台目も作成: `matwm-gamma-false-h100`

### ステップ3: セットアップ（VM上）

SSH接続後:

```bash
# 作業ディレクトリ作成
mkdir -p ~/matwm_project && cd ~/matwm_project

# ファイルダウンロード
gsutil -m cp gs://matwm-training-bucket/*.py .
gsutil cp gs://matwm-training-bucket/requirements.txt .

# パッケージインストール
pip install -r requirements.txt

# GPU確認
nvidia-smi
```

### ステップ4: 実行開始（VM上）

```bash
# tmuxセッション開始
tmux new -s matwm_true

# 実行（インスタンス1: gamma_true）
python3 train_gamma_true.py 2>&1 | tee train_true.log

# デタッチ: Ctrl+B → D
```

インスタンス2でも同様に:
```bash
tmux new -s matwm_false
python3 train_gamma_false.py 2>&1 | tee train_false.log
# Ctrl+B → D
```

### ステップ5: 結果取得（12時間後）

VM上:
```bash
# 結果をGCSにアップロード
gsutil -m cp -r results_gamma_true gs://matwm-training-bucket/
gsutil -m cp -r results_gamma_false gs://matwm-training-bucket/
```

ローカルPC:
```cmd
mkdir gcp_results
gsutil -m cp -r gs://matwm-training-bucket/results_* gcp_results\
```

---

## 📊 進捗確認

```bash
# セッションに再接続
tmux attach -t matwm_true

# ログ確認
tail -f train_true.log

# GPU使用率確認
watch -n 1 nvidia-smi
```

---

## 🛑 完了後の停止

```cmd
# インスタンス削除（課金停止）
gcloud compute instances delete matwm-gamma-true-h100 --zone=us-central1-a
gcloud compute instances delete matwm-gamma-false-h100 --zone=us-central1-a
```

---

## ⚠️ よくある問題

### GPU割り当てクォータ不足
GCPコンソール → **IAMと管理** → **割り当て** → 「NVIDIA H100」で検索 → 割り当てを編集

### SSH切断後の再接続
```bash
gcloud compute ssh matwm-gamma-true-h100 --zone=us-central1-a
tmux attach -t matwm_true
```

---

**詳細は `GCP_EXECUTION_GUIDE.md` を参照してください。**
