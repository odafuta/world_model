# GCP H100 実行ガイド - MATWM Training (Jupyter Notebook版)

## 📊 実行時間とコスト予想（H100使用時）

### 構成
- **GPU**: NVIDIA H100 80GB × 1
- **CPU**: n1-standard-16 (16 vCPU, 60GB RAM)
- **訓練設定**: total_steps=50000（論文再現）

### 時間とコスト
| 項目 | 予想値 |
|------|--------|
| **実行時間** | **11-13時間/台** |
| **時間単価** | $5.92/時 (VM $0.76 + H100 $5.12 + Disk $0.04) |
| **1台あたりコスト** | **$65-77** (約9,750-11,550円) |
| **2台同時実行** | **$130-154** (約19,500-23,100円) |

### 最速構成の推奨
- **H100 80GB**: 最速のGPU（A100の1.6倍高速）
- **n1-standard-16**: 16 vCPU でデータ処理を高速化
- **us-central1リージョン**: H100が利用可能

---

## 🚀 ステップ1: ローカルPCでの準備

### 1-1. GCP CLIのインストール（初回のみ）

Windows PowerShell または コマンドプロンプトで:

```cmd
# Google Cloud SDK をダウンロードしてインストール
# https://cloud.google.com/sdk/docs/install からインストーラーをダウンロード

# インストール後、認証
gcloud auth login

# プロジェクトを設定
gcloud config set project YOUR_PROJECT_ID

# 現在の設定を確認
gcloud config list
```

### 1-2. Cloud Storageバケットの作成

```cmd
# バケット作成（初回のみ）
gsutil mb -l us-central1 gs://matwm-training-bucket

# バケットの確認
gsutil ls
```

### 1-3. 必要なファイルをアップロード

```cmd
# プロジェクトディレクトリに移動
cd "C:\Users\0622d\OneDrive - OUMail (Osaka University)\M1_秋冬\松尾研究室\WorldModel\最終課題"

# 必要なファイルをアップロード（Pythonファイル、Notebook、設定ファイル）
gsutil -m cp *.py gs://matwm-training-bucket/
gsutil cp requirements.txt gs://matwm-training-bucket/
gsutil cp *.md gs://matwm-training-bucket/

# 実験ディレクトリをアップロード
gsutil -m rsync -r experiments/ gs://matwm-training-bucket/experiments/
```

**アップロード完了を確認:**
```cmd
gsutil ls gs://matwm-training-bucket/
gsutil ls gs://matwm-training-bucket/experiments/
```

出力例:
```
gs://matwm-training-bucket/curiosity_reward.py
gs://matwm-training-bucket/matwm_agent.py
gs://matwm-training-bucket/matwm_implementation.py
gs://matwm-training-bucket/matwm_utils.py
gs://matwm-training-bucket/requirements.txt
gs://matwm-training-bucket/experiments/
gs://matwm-training-bucket/experiments/llm_curiosity_and_γ-progress/
gs://matwm-training-bucket/experiments/only_llm_curiosity/
gs://matwm-training-bucket/experiments/only_γ_progress/
```

---

## 🚀 ステップ2: GCP VMインスタンスの作成

### 2-1. GCPコンソールにアクセス

1. ブラウザで https://console.cloud.google.com/ を開く
2. 左上のメニュー → **Compute Engine** → **VM instances**
3. **CREATE INSTANCE** をクリック

### 2-2. インスタンス1の作成（use_gamma_progress=True用）

#### 基本設定
```
名前: matwm-gamma-true-h100
リージョン: us-central1
ゾーン: us-central1-a
```

#### マシン構成
```
マシンファミリー: 汎用
シリーズ: N1
マシンタイプ: n1-standard-16
  - vCPU: 16
  - メモリ: 60 GB
```

#### GPU設定
```
GPU タイプ: NVIDIA H100 80GB
GPU 数: 1

※ 注意: GPU割り当てクォータが必要です
※ クォータが不足している場合は、「割り当てを増やす」をクリック
```

#### ブートディスク
```
オペレーティングシステム: Deep Learning on Linux
バージョン: Debian 11 based Deep Learning VM with CUDA 11.8 M126
ディスクタイプ: 標準永続ディスク
サイズ: 200 GB
```

#### ファイアウォール（オプション）
```
□ HTTPトラフィックを許可
□ HTTPSトラフィックを許可
```

**作成** をクリック

⏱️ **作成時間**: 約2-3分

### 2-3. インスタンス2の作成（use_gamma_progress=False用）

同じ手順で2台目を作成:
```
名前: matwm-gamma-false-h100
（他の設定は同じ）
```

### 2-4. インスタンスの起動確認

VMインスタンス一覧で、両方のインスタンスが「実行中」（緑のチェックマーク）になっていることを確認。

---

## 🚀 ステップ3: SSH接続とセットアップ（インスタンス1）

### 3-1. SSH接続

GCPコンソールのVMインスタンス一覧で:
1. **matwm-gamma-true-h100** の行を探す
2. **SSH** ボタンをクリック
3. 新しいウィンドウでターミナルが開く

### 3-2. GPU動作確認

```bash
# GPU情報を確認
nvidia-smi
```

**期待される出力:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.0   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA H100 80GB    Off  | 00000000:00:04.0 Off |                    0 |
| N/A   32C    P0    70W / 700W |      0MiB / 81559MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

✅ H100が認識されていることを確認

### 3-3. Python環境の確認

```bash
# Pythonバージョン確認
python3 --version

# PyTorch確認
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

**期待される出力:**
```
Python 3.10.x
PyTorch: 2.x.x
CUDA available: True
```

### 3-4. 作業ディレクトリの作成

```bash
# ホームディレクトリに作業フォルダ作成
mkdir -p ~/matwm_project
cd ~/matwm_project

# 現在のディレクトリを確認
pwd
```

出力: `/home/YOUR_USERNAME/matwm_project`

### 3-5. ファイルのダウンロード

```bash
# Cloud Storageからファイルをダウンロード
gsutil -m cp gs://matwm-training-bucket/*.py .
gsutil cp gs://matwm-training-bucket/requirements.txt .

# 実験ディレクトリもダウンロード
gsutil -m rsync -r gs://matwm-training-bucket/experiments/ experiments/

# ダウンロード確認
ls -lh
ls -R experiments/
```

**期待される出力:**
```
-rw-r--r-- 1 user user  50K Feb 11 10:00 curiosity_reward.py
-rw-r--r-- 1 user user  30K Feb 11 10:00 matwm_agent.py
-rw-r--r-- 1 user user  25K Feb 11 10:00 matwm_implementation.py
-rw-r--r-- 1 user user  20K Feb 11 10:00 matwm_utils.py
-rw-r--r-- 1 user user 500B Feb 11 10:00 requirements.txt
drwxr-xr-x 5 user user 4.0K Feb 11 10:00 experiments/

experiments/:
llm_curiosity_and_γ-progress/
only_llm_curiosity/
only_γ_progress/
README.md
```

### 3-6. Jupyter Notebookのインストール

```bash
# Jupyter関連パッケージをインストール
pip install jupyter jupyterlab ipykernel

# カーネルの登録
python3 -m ipykernel install --user --name matwm --display-name "Python (MATWM)"

# インストール確認
jupyter --version
```

### 3-7. 依存パッケージのインストール

```bash
# requirements.txtから一括インストール
pip install -r requirements.txt

# インストール完了確認
pip list | grep -E "torch|pettingzoo|gymnasium"
```

⏱️ **インストール時間**: 約3-5分

**期待される出力:**
```
gymnasium         0.29.x
pettingzoo        1.24.x
torch             2.x.x
```

### 3-8. 動作テスト

```bash
# 簡単な動作確認
python3 -c "
import torch
import pettingzoo
from matwm_implementation import MATWMConfig
print('✓ All imports successful')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
"
```

**期待される出力:**
```
✓ All imports successful
✓ CUDA available: True
✓ GPU: NVIDIA H100 80GB HBM3
```

---

## 🚀 ステップ4: Jupyter Notebookの実行

### 4-1. JupyterLabの起動

```bash
# tmuxセッションを開始（切断しても実行継続）
tmux new -s jupyter

# JupyterLabを起動（ポート8888で起動）
cd ~/matwm_project
jupyter lab --no-browser --port=8888 --ip=0.0.0.0 --allow-root
```

**出力例:**
```
[I 2026-02-11 10:00:00.000 ServerApp] Jupyter Server 2.x.x is running at:
[I 2026-02-11 10:00:00.000 ServerApp] http://localhost:8888/lab?token=abcd1234...
[I 2026-02-11 10:00:00.000 ServerApp]  or http://127.0.0.1:8888/lab?token=abcd1234...
```

トークン（`token=...`）をコピーしておいてください。

### 4-2. SSHポートフォワーディングの設定

**ローカルPC（別のターミナル）で実行:**

```cmd
# GCP VMにSSHポートフォワーディングで接続
gcloud compute ssh matwm-gamma-true-h100 \
  --zone=us-central1-a \
  -- -L 8888:localhost:8888
```

これで、ローカルPCのブラウザから `http://localhost:8888` でJupyterLabにアクセスできます。

### 4-3. Notebookの実行

1. ブラウザで `http://localhost:8888` を開く
2. 先ほどコピーしたトークンを入力
3. `experiments/` ディレクトリを開く
4. 実行したい実験メソッドのディレクトリを選択：
   - `llm_curiosity_and_γ-progress/` - LLM好奇心 + γ-Progress
   - `only_llm_curiosity/` - LLM好奇心のみ
   - `only_γ_progress/` - γ-Progressのみ
5. Notebookファイル（`.ipynb`）を開く：
   - フル実行版（50,000ステップ）: `2026_MATWM_simple_tag_Implementation_*.ipynb`
   - テスト版（2,000ステップ）: `*_test.ipynb`
6. カーネルを「Python (MATWM)」に設定
7. セルを順番に実行（Shift + Enter）

### 4-4. 出力ディレクトリの確認

Notebookを実行すると、以下のディレクトリに結果が保存されます：

```
~/matwm_project/
├── llm_logs/
│   ├── llm_and_gamma/
│   │   └── 20260211_100030/  # タイムスタンプ付き
│   ├── only_llm/
│   │   └── 20260211_110045/
│   └── only_gamma/
│       └── 20260211_120100/
└── results/
    ├── llm_and_gamma/
    │   └── 20260211_100030/
    │       ├── checkpoint_5000/
    │       ├── checkpoint_10000/
    │       └── training_curves.png
    ├── only_llm/
    └── only_gamma/
```

### 4-5. 実行の監視

```bash
# 別のtmuxウィンドウを開く（Ctrl+B → C）
# または別のSSHセッションで接続

# ログディレクトリを監視
watch -n 10 'ls -lRh llm_logs/'
watch -n 10 'ls -lRh results/'

# GPUの使用状況を監視
watch -n 1 nvidia-smi
```

### 4-6. tmuxセッションのデタッチ

JupyterLabが起動したら、必要に応じてセッションから離脱します:

```
Ctrl+B を押してから D を押す
```

元のターミナルに戻ります。SSH接続を切断してもJupyterLabは継続されます。

---

## 🚀 ステップ5: 複数実験の同時実行（オプション）

複数のGCP VMインスタンスで異なる実験を並列実行できます。

### 5-1. 追加インスタンスの作成

ステップ2と同様に、追加のVMインスタンスを作成します：
- `matwm-only-llm-h100` (Only LLM Curiosity実験用)
- `matwm-only-gamma-h100` (Only γ-Progress実験用)

### 5-2. セットアップと実行

各インスタンスで、ステップ3-4を繰り返します：
1. SSH接続
2. 環境セットアップ
3. JupyterLabの起動（異なるポートを使用、例: 8889, 8890）
4. 対応する実験Notebookを実行

**ポートフォワーディングの例（複数インスタンス）:**
```cmd
# インスタンス1（llm_and_gamma）
gcloud compute ssh matwm-gamma-true-h100 --zone=us-central1-a -- -L 8888:localhost:8888

# インスタンス2（only_llm）
gcloud compute ssh matwm-only-llm-h100 --zone=us-central1-a -- -L 8889:localhost:8889

# インスタンス3（only_gamma）
gcloud compute ssh matwm-only-gamma-h100 --zone=us-central1-a -- -L 8890:localhost:8890
```

---

## 🚀 ステップ6: 進捗確認

### 6-1. tmuxセッションへの再接続

いつでもSSH接続して進捗を確認できます:

```bash
# セッション一覧を確認
tmux ls

# セッションにアタッチ
tmux attach -t matwm_true

# デタッチ: Ctrl+B → D
```

### 6-2. Jupyter Notebookの出力確認

Notebookの各セルの出力をブラウザで直接確認できます。
また、保存された結果ファイルも確認できます：

```bash
# LLMログの確認
ls -lh llm_logs/*/

# 結果ディレクトリの確認
ls -lRh results/*/

# チェックポイントの確認
find results/ -name "checkpoint_*" -type d
```

### 6-3. GPU使用状況の確認

```bash
# リアルタイムモニタリング
watch -n 1 nvidia-smi

# 停止: Ctrl+C
```

**期待される出力:**
```
+-----------------------------------------------------------------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA H100 80GB    Off  | 00000000:00:04.0 Off |                    0 |
| N/A   65C    P0   450W / 700W |  35000MiB / 81559MiB |     95%      Default |
+-------------------------------+----------------------+----------------------+
```

✅ GPU使用率が90%以上であれば正常に動作中

### 6-4. 進捗の目安

```
Step 5000/50000 (10%完了) - 約1.2時間経過
Step 10000/50000 (20%完了) - 約2.4時間経過
Step 25000/50000 (50%完了) - 約6時間経過
Step 40000/50000 (80%完了) - 約9.6時間経過
Step 50000/50000 (100%完了) - 約12時間経過
```

---

## 🚀 ステップ7: 結果の取得

### 7-1. トレーニング完了の確認

Notebookの最終セルが実行完了すると、訓練が終了します。
ブラウザでNotebookの出力を確認してください。

**完了時の出力例（Notebook内）:**
```
Training: 100%|██████████| 50000/50000 [12:15:32<00:00, 1.13it/s]

======================================================================
Training Complete!
======================================================================
Total episodes: 2000
Final checkpoint saved
Training curves saved to: results/llm_and_gamma/20260211_100030/training_curves.png
```

### 7-2. 結果をCloud Storageにアップロード

```bash
# SSH接続して、結果をGCSにアップロード
cd ~/matwm_project

# 全実験結果をアップロード
gsutil -m rsync -r llm_logs/ gs://matwm-training-bucket/llm_logs/
gsutil -m rsync -r results/ gs://matwm-training-bucket/results/

# アップロード確認
gsutil ls -r gs://matwm-training-bucket/llm_logs/
gsutil ls -r gs://matwm-training-bucket/results/
```

### 7-3. ローカルPCにダウンロード

ローカルPC（Windows PowerShell）で:

```powershell
cd "C:\Users\0622d\OneDrive - OUMail (Osaka University)\M1_秋冬\松尾研究室\WorldModel\最終課題"

# 結果ディレクトリを作成
mkdir gcp_results -Force

# ダウンロード（全実験結果とLLMログ）
gsutil -m rsync -r gs://matwm-training-bucket/llm_logs/ gcp_results/llm_logs/
gsutil -m rsync -r gs://matwm-training-bucket/results/ gcp_results/results/

# ダウンロード確認
ls -R gcp_results
```

**ダウンロードされる構造:**
```
gcp_results/
├── llm_logs/
│   ├── llm_and_gamma/
│   │   └── 20260211_100030/
│   ├── only_llm/
│   └── only_gamma/
└── results/
    ├── llm_and_gamma/
    │   └── 20260211_100030/
    │       ├── checkpoint_5000/
    │       ├── checkpoint_10000/
    │       └── training_curves.png
    ├── only_llm/
    └── only_gamma/
```

⏱️ **ダウンロード時間**: 約5-10分（結果ファイルのサイズによる）

---

## 🚀 ステップ8: VMインスタンスの停止・削除

### 8-1. インスタンスの停止（再利用する場合）

ローカルPCで:

```cmd
REM 停止（課金が大幅に減る）
gcloud compute instances stop matwm-gamma-true-h100 --zone=us-central1-a
gcloud compute instances stop matwm-gamma-false-h100 --zone=us-central1-a

REM 停止確認
gcloud compute instances list
```

**停止時の課金:**
- GPU: 課金停止
- CPU: 課金停止
- ディスク: 継続（約$0.04/時 → $0.96/日）

### 8-2. インスタンスの削除（完全に終了する場合）

```cmd
REM 削除（課金完全停止）
gcloud compute instances delete matwm-gamma-true-h100 --zone=us-central1-a
gcloud compute instances delete matwm-gamma-false-h100 --zone=us-central1-a

REM 確認メッセージで "Y" を入力
```

⚠️ **注意**: 削除するとVMの全データが失われます。結果をGCSにアップロード済みか確認してください。

---

## 📊 コスト詳細

### 実行コスト（1台あたり）

| 項目 | 単価 | 時間 | 合計 |
|------|------|------|------|
| n1-standard-16 | $0.76/時 | 12時間 | $9.12 |
| H100 80GB | $5.12/時 | 12時間 | $61.44 |
| 標準永続ディスク 200GB | $0.04/時 | 12時間 | $0.48 |
| **合計** | **$5.92/時** | **12時間** | **$71.04** |

### 2台同時実行の総コスト

```
$71.04 × 2台 = $142.08 (約21,312円)
```

### コスト削減のヒント

1. **プリエンプティブルVM**: 最大80%割引（ただし24時間以内に停止される可能性）
2. **Spot VM**: 最大91%割引（同上）
3. **リージョン選択**: us-central1が最安
4. **実行時間の最適化**: warmup_stepsを調整

---

## ⚠️ トラブルシューティング

### 問題1: GPU割り当てクォータ不足

**エラー:**
```
Quota 'NVIDIA_H100_GPUS' exceeded. Limit: 0.0 in region us-central1.
```

**解決策:**
1. GCPコンソール → **IAMと管理** → **割り当て**
2. 「NVIDIA H100」で検索
3. 「割り当てを編集」をクリック
4. 新しい上限: 2（2台分）
5. リクエストを送信（承認まで数時間～1日）

### 問題2: SSH接続が切れた

**解決策:**
```bash
# 再接続してtmuxセッションにアタッチ
gcloud compute ssh matwm-gamma-true-h100 --zone=us-central1-a
tmux attach -t matwm_true
```

### 問題3: CUDA Out of Memory

**解決策:**
```python
# train_gamma_true.py の設定を調整
config = MATWMConfig(
    wm_batch_size=16,  # デフォルト32から削減
    ac_batch_size=128,  # デフォルト256から削減
)
```

### 問題4: パッケージインポートエラー

**解決策:**
```bash
# 再インストール
pip install --upgrade -r requirements.txt

# キャッシュクリア
pip cache purge
pip install --no-cache-dir -r requirements.txt
```

---

## ✅ チェックリスト

### 実行前
- [ ] GCP CLIインストール済み
- [ ] プロジェクト設定完了
- [ ] Cloud Storageバケット作成
- [ ] ファイルアップロード完了
- [ ] H100 GPU割り当てクォータ確認

### インスタンス作成
- [ ] インスタンス1作成（gamma_true）
- [ ] インスタンス2作成（gamma_false）
- [ ] 両方のインスタンスが実行中

### セットアップ
- [ ] SSH接続成功
- [ ] GPU認識確認（nvidia-smi）
- [ ] ファイルダウンロード完了
- [ ] パッケージインストール完了
- [ ] 動作テスト成功

### 実行
- [ ] tmuxセッション開始
- [ ] トレーニング実行開始
- [ ] GPU使用率確認（90%以上）
- [ ] 両方のインスタンスで実行中

### 完了後
- [ ] トレーニング完了確認
- [ ] 結果をGCSにアップロード
- [ ] ローカルPCにダウンロード
- [ ] VMインスタンス停止/削除

---

## 📞 サポート

### 問題が発生した場合

1. **ログを確認**: `tail -f train_true_*.log`
2. **GPU状態を確認**: `nvidia-smi`
3. **ディスク容量を確認**: `df -h`
4. **メモリ使用量を確認**: `free -h`

### 参考リンク

- [GCP Compute Engine ドキュメント](https://cloud.google.com/compute/docs)
- [H100 GPU 仕様](https://www.nvidia.com/en-us/data-center/h100/)
- [PyTorch ドキュメント](https://pytorch.org/docs/stable/index.html)

---

## 🎯 まとめ

### 最速実行のポイント

1. ✅ **H100 80GB**: 最速GPU（A100の1.6倍）
2. ✅ **n1-standard-16**: 16 vCPUでボトルネック解消
3. ✅ **2台同時実行**: 並列実行で時間短縮
4. ✅ **us-central1**: 最安リージョン

### 予想実行時間

```
準備: 30分
実行: 12時間
結果取得: 30分
─────────────
合計: 約13時間
```

### 予想コスト

```
2台 × 12時間 × $5.92/時 = $142 (約21,300円)
```

**最速で確実に実行を完了させるための完全ガイドです。Good luck! 🚀**
