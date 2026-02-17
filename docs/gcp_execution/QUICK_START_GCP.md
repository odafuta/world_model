# GCP H100 クイックスタートガイド（Jupyter Notebook版）

## ⏱️ 実行時間とコスト（H100使用）

| 項目 | 値 |
|------|-----|
| **実行時間** | **11-13時間/台** |
| **コスト** | **$71/台** (約10,650円) |
| **複数実験同時** | **$71 × 実験数** |

---

## 🚀 5ステップで実行開始

### ステップ1: ファイルアップロード（ローカルPC）

```powershell
cd "C:\Users\0622d\OneDrive - OUMail (Osaka University)\M1_秋冬\松尾研究室\WorldModel\最終課題"

gcloud auth login
gsutil mb gs://matwm-training-bucket

# Pythonファイルと実験Notebookをアップロード
gsutil -m cp *.py requirements.txt gs://matwm-training-bucket/
gsutil -m rsync -r experiments/ gs://matwm-training-bucket/experiments/
```

### ステップ2: VMインスタンス作成（GCPコンソール）

1. https://console.cloud.google.com/ → **Compute Engine** → **CREATE INSTANCE**

2. 設定:
   ```
   名前: matwm-h100 (実験ごとに異なる名前も可)
   リージョン: us-central1-a
   マシン: n1-standard-16 (16 vCPU, 60GB RAM)
   GPU: NVIDIA H100 80GB × 1
   OS: Deep Learning on Linux (Debian 11, CUDA 11.8)
   ディスク: 200GB
   ```

3. **作成** をクリック

4. 複数実験を並列実行する場合は、必要な台数分作成

### ステップ3: セットアップ（VM上）

SSH接続後:

```bash
# 作業ディレクトリ作成
mkdir -p ~/matwm_project && cd ~/matwm_project

# ファイルダウンロード
gsutil -m cp gs://matwm-training-bucket/*.py .
gsutil cp gs://matwm-training-bucket/requirements.txt .
gsutil -m rsync -r gs://matwm-training-bucket/experiments/ experiments/

# Jupyter & パッケージインストール
pip install jupyter jupyterlab ipykernel
pip install -r requirements.txt
python3 -m ipykernel install --user --name matwm --display-name "Python (MATWM)"

# GPU確認
nvidia-smi
```

### ステップ4: Jupyter Lab起動とNotebook実行（VM上）

```bash
# tmuxセッション開始
tmux new -s jupyter

# JupyterLab起動
jupyter lab --no-browser --port=8888 --ip=0.0.0.0 --allow-root

# トークンをコピーしておく（出力に表示される）
# 例: http://localhost:8888/lab?token=abcd1234...

# デタッチ: Ctrl+B → D
```

**ローカルPCで（別のターミナル）:**
```powershell
# SSHポートフォワーディング
gcloud compute ssh matwm-h100 --zone=us-central1-a -- -L 8888:localhost:8888

# ブラウザで http://localhost:8888 を開いてトークンを入力
# experiments/ 内の実験Notebookを選択して実行
```

### ステップ5: 結果取得（12時間後）

VM上（SSH接続）:
```bash
cd ~/matwm_project

# 結果をGCSにアップロード
gsutil -m rsync -r llm_logs/ gs://matwm-training-bucket/llm_logs/
gsutil -m rsync -r results/ gs://matwm-training-bucket/results/
```

ローカルPC:
```powershell
mkdir gcp_results -Force

# 全実験結果をダウンロード
gsutil -m rsync -r gs://matwm-training-bucket/llm_logs/ gcp_results/llm_logs/
gsutil -m rsync -r gs://matwm-training-bucket/results/ gcp_results/results/
```

**ダウンロードされる構造:**
```
gcp_results/
├── llm_logs/
│   ├── llm_and_gamma/
│   ├── only_llm/
│   └── only_gamma/
└── results/
    ├── llm_and_gamma/
    ├── only_llm/
    └── only_gamma/
```

---

## 📊 進捗確認

```bash
# JupyterLabセッションに再接続
tmux attach -t jupyter

# 結果ディレクトリの監視（別ウィンドウで）
watch -n 10 'ls -lRh results/'

# GPU使用率確認
watch -n 1 nvidia-smi
```

**ブラウザで:** http://localhost:8888 でNotebookの実行状況を直接確認

---

## 🛑 完了後の停止

```powershell
# インスタンス削除（課金停止）
gcloud compute instances delete matwm-h100 --zone=us-central1-a

# 複数インスタンスがある場合
gcloud compute instances list
gcloud compute instances delete [INSTANCE_NAME] --zone=us-central1-a
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
