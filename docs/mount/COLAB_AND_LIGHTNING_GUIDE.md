# Google Colab & Lightning AI 実行ガイド

## 📊 プラットフォーム比較

| 項目 | Google Colab Pro+ | Lightning AI | GCP (参考) |
|------|------------------|--------------|-----------|
| **GPU** | A100 / H100 | H100 | H100 |
| **実行時間** | 最大12時間/セッション | 制限なし | 制限なし |
| **月額料金** | $49.99 | 従量課金 | 従量課金 |
| **時間単価** | - | $2-3/時 | $5.92/時 |
| **総コスト（50000 steps）** | $49.99（月額） | $22-39 | $71 |
| **切断リスク** | あり（12時間） | なし | なし |
| **推奨度** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### 推奨順位

1. **Lightning AI** ⭐⭐⭐⭐⭐
   - 最もコスパが良い
   - 切断リスクなし
   - H100が使える

2. **Google Colab Pro+** ⭐⭐⭐
   - 月額固定で使いやすい
   - 12時間制限に注意
   - チェックポイントから再開必要

3. **GCP** ⭐⭐⭐⭐
   - Free-tier制限で使えない場合あり
   - 最も安定

---

## 🚀 方法1: Google Colab（H100/A100）

### ステップ1: Colab Notebookの準備

#### 1-1. ファイルの準備

**ローカルPCで:**

```cmd
cd "C:\Users\0622d\OneDrive - OUMail (Osaka University)\M1_秋冬\松尾研究室\WorldModel\最終課題"

REM 必要なファイルをZIP圧縮
powershell Compress-Archive -Path train_gamma_true.py,train_gamma_false.py,matwm_implementation.py,matwm_agent.py,matwm_utils.py,curiosity_reward.py,requirements.txt -DestinationPath MATWM_Project.zip
```

#### 1-2. Google Driveにアップロード

1. Google Drive (https://drive.google.com/) を開く
2. `MATWM_Project.zip` をアップロード
3. アップロード完了を確認

#### 1-3. Colab Notebookを開く

1. Google Colab (https://colab.research.google.com/) を開く
2. **ファイル** → **ノートブックを開く**
3. **アップロード** タブ
4. `colab_setup.ipynb` をアップロード

または、新しいノートブックを作成して以下のコードをコピー

### ステップ2: GPUランタイムの設定

1. **ランタイム** → **ランタイムのタイプを変更**
2. **ハードウェアアクセラレータ**: GPU
3. **GPU タイプ**: 
   - Colab Pro: **A100**
   - Colab Pro+: **H100** (利用可能な場合)
4. **保存**

### ステップ3: セットアップの実行

`colab_setup.ipynb` の各セルを順番に実行:

1. **Google Drive マウント** → 認証
2. **ファイル解凍** → ZIPを解凍
3. **GPU確認** → H100/A100を確認
4. **パッケージインストール** → 依存関係をインストール
5. **動作確認** → インポートテスト

### ステップ4: トレーニング実行

#### use_gamma_progress=True

```python
%cd /content/matwm_project
!python train_gamma_true.py 2>&1 | tee train_true.log
```

#### use_gamma_progress=False

```python
%cd /content/matwm_project
!python train_gamma_false.py 2>&1 | tee train_false.log
```

### ステップ5: 定期的なバックアップ（重要！）

**Colabは12時間で切断されるため、定期的にバックアップが必要**

```python
import shutil
import os
from datetime import datetime

# バックアップ先
backup_dir = '/content/drive/MyDrive/MATWM_Results'
os.makedirs(backup_dir, exist_ok=True)

# タイムスタンプ
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# 結果をコピー
if os.path.exists('/content/matwm_project/results_gamma_true'):
    dest = os.path.join(backup_dir, f'results_gamma_true_{timestamp}')
    shutil.copytree('/content/matwm_project/results_gamma_true', dest)
    print(f'✓ Backed up to: {dest}')
```

**このセルを2-3時間ごとに実行してください**

### ステップ6: チェックポイントから再開

切断された場合:

1. 新しいColabセッションを開始
2. セットアップセルを再実行
3. 最新のチェックポイントを確認:

```python
import glob

checkpoint_pattern = '/content/drive/MyDrive/MATWM_Results/results_gamma_true_*/*/checkpoint_*/full_checkpoint.pt'
checkpoints = sorted(glob.glob(checkpoint_pattern))

if checkpoints:
    latest = checkpoints[-1]
    print(f'Latest checkpoint: {latest}')
```

4. チェックポイントから再開（実装が必要）

### Colabのコストと制限

| プラン | 月額 | GPU | 実行時間 | 推奨 |
|--------|------|-----|---------|------|
| Free | $0 | T4 | 短時間 | ❌ 不十分 |
| Pro | $9.99 | A100 | 最大12時間 | ⭐⭐ |
| Pro+ | $49.99 | H100 | 最大12時間 | ⭐⭐⭐ |

**注意:**
- 12時間制限があるため、50000 stepsは複数セッション必要
- チェックポイントから再開する仕組みが必要

---

## 🚀 方法2: Lightning AI（H100）- 推奨

### ステップ1: Lightning AIアカウント作成

1. https://lightning.ai/ にアクセス
2. **Sign Up** をクリック
3. GitHubまたはGoogleアカウントで登録
4. メール認証を完了

### ステップ2: 新しいStudioを作成

1. ダッシュボードで **New Studio** をクリック
2. **Studio名**: `MATWM Training`
3. **Hardware**: 
   - GPU: **H100 (80GB)**
   - CPU: 8-16 cores
   - RAM: 32-64 GB
4. **Create** をクリック

⏱️ 起動時間: 約2-3分

### ステップ3: ファイルのアップロード

#### 方法A: Web UI経由（推奨）

1. Lightning AI Studio が開いたら、左サイドバーの **Files** をクリック
2. **Upload** ボタンをクリック
3. 以下のファイルを選択してアップロード:
   - `train_gamma_true.py`
   - `train_gamma_false.py`
   - `matwm_implementation.py`
   - `matwm_agent.py`
   - `matwm_utils.py`
   - `curiosity_reward.py`
   - `requirements.txt`
   - `lightning_ai_setup.py`

#### 方法B: Git経由

Lightning AI Studio のターミナルで:

```bash
cd ~
git clone YOUR_GITHUB_REPO_URL matwm_project
cd matwm_project
```

#### 方法C: ZIP経由

1. ローカルPCでZIP作成（上記参照）
2. Lightning AI Studio の Files → Upload
3. `MATWM_Project.zip` をアップロード
4. ターミナルで解凍:

```bash
cd ~
unzip MATWM_Project.zip -d matwm_project
cd matwm_project
```

### ステップ4: セットアップスクリプトの実行

Lightning AI Studio のターミナルで:

```bash
cd ~/matwm_project
python lightning_ai_setup.py
```

**出力例:**
```
======================================================================
Lightning AI H100 Setup for MATWM Training
======================================================================

[1/7] Checking environment...
✓ Running on Lightning AI
  Project ID: abc123...
  
PyTorch version: 2.x.x
CUDA available: True
GPU: NVIDIA H100 80GB HBM3
VRAM: 81.6 GB

[2/7] Creating working directory...
✓ Working directory: /home/user/matwm_project

[3/7] File upload instructions:
...

[4/7] Verifying files...
✓ train_gamma_true.py
✓ train_gamma_false.py
✓ matwm_implementation.py
✓ matwm_agent.py
✓ matwm_utils.py
✓ curiosity_reward.py
✓ requirements.txt

✓ All required files found

[5/7] Installing dependencies...
✓ Dependencies installed successfully

[6/7] Testing imports...
✓ matwm_implementation
✓ matwm_agent
✓ curiosity_reward
✓ pettingzoo

[7/7] Setup complete!
```

### ステップ5: トレーニング実行

#### バックグラウンドで実行（推奨）

```bash
cd ~/matwm_project

# use_gamma_progress=True
nohup python train_gamma_true.py > train_true.log 2>&1 &

# プロセスIDを確認
echo $!

# use_gamma_progress=False（別のターミナルで）
nohup python train_gamma_false.py > train_false.log 2>&1 &
```

#### フォアグラウンドで実行

```bash
cd ~/matwm_project
python train_gamma_true.py 2>&1 | tee train_true.log
```

### ステップ6: 進捗確認

```bash
# ログをリアルタイム表示
tail -f ~/matwm_project/train_true.log

# GPU使用率確認
nvidia-smi

# プロセス確認
ps aux | grep python
```

### ステップ7: 結果のダウンロード

#### 方法A: Web UI経由

1. Lightning AI Studio の Files
2. `results_gamma_true/` を右クリック
3. **Download** を選択

#### 方法B: ZIP圧縮してダウンロード

```bash
cd ~/matwm_project
zip -r results_gamma_true.zip results_gamma_true/
zip -r results_gamma_false.zip results_gamma_false/
```

Lightning AI Studio の Files から ZIP をダウンロード

#### 方法C: Lightning AI CLI（ローカルPC）

```bash
# ローカルPCで
lightning download ~/matwm_project/results_gamma_true/
```

### Lightning AIのコスト

| リソース | 時間単価 | 12時間 | 備考 |
|---------|---------|--------|------|
| H100 80GB | $2.00-3.00/時 | $24-36 | 最速 |
| A100 40GB | $1.00-1.50/時 | $12-18 | コスパ良 |

**50000 steps（約12時間）の総コスト:**
- H100: **$24-36**
- A100: **$12-18**（実行時間は1.6倍）

### Lightning AIの利点

✅ **切断リスクなし** - 長時間実行可能
✅ **コスパ良好** - GCPより安い
✅ **簡単なセットアップ** - Web UIで完結
✅ **柔軟な課金** - 使った分だけ支払い

---

## 📊 実行時間とコスト比較

### 50000 steps（論文再現）の場合

| プラットフォーム | GPU | 実行時間 | コスト | 切断リスク | 推奨度 |
|----------------|-----|---------|--------|-----------|--------|
| **Lightning AI** | H100 | 11-13時間 | $24-36 | なし | ⭐⭐⭐⭐⭐ |
| **Colab Pro+** | H100 | 11-13時間 | $49.99/月 | あり | ⭐⭐⭐ |
| **Colab Pro** | A100 | 18-22時間 | $9.99/月 | あり | ⭐⭐ |
| **GCP** | H100 | 11-13時間 | $71 | なし | ⭐⭐⭐⭐ |

### 推奨

1. **最もコスパが良い**: Lightning AI（H100）
2. **月額固定が良い**: Colab Pro+（ただし12時間制限）
3. **最も安定**: GCP（Free-tier制限に注意）

---

## 🎯 実行パターン別の推奨

### パターン1: 時間がない、確実に完了させたい

**推奨: Lightning AI（H100）**

```bash
# セットアップ: 10分
python lightning_ai_setup.py

# 実行: 11-13時間（切断なし）
nohup python train_gamma_true.py > train_true.log 2>&1 &
```

**メリット:**
- ✅ 切断リスクなし
- ✅ 1回で完了
- ✅ コスト: $24-36

### パターン2: 月額固定が良い

**推奨: Google Colab Pro+（H100）**

```python
# セットアップ: 10分
# 実行: 12時間 × 2セッション

# セッション1（12時間）
!python train_gamma_true.py

# バックアップ → 再開
# セッション2（残り時間）
```

**メリット:**
- ✅ 月額固定（$49.99）
- ✅ 使い慣れたColab環境

**デメリット:**
- ⚠️ 12時間制限
- ⚠️ チェックポイントから再開が必要

### パターン3: 最も安く（時間がかかっても良い）

**推奨: Google Colab Pro（A100）**

```python
# 実行: 18-22時間（2-3セッション必要）
```

**メリット:**
- ✅ 月額$9.99（最安）

**デメリット:**
- ⚠️ 実行時間が長い
- ⚠️ 複数セッション必要

---

## 📋 チェックリスト

### Google Colab
- [ ] Colab Pro/Pro+アカウント作成
- [ ] ファイルをZIP圧縮
- [ ] Google Driveにアップロード
- [ ] `colab_setup.ipynb` を開く
- [ ] GPUランタイム設定（H100/A100）
- [ ] セットアップセル実行
- [ ] トレーニング開始
- [ ] 定期的にバックアップ

### Lightning AI
- [ ] Lightning AIアカウント作成
- [ ] 新しいStudio作成（H100）
- [ ] ファイルアップロード
- [ ] `lightning_ai_setup.py` 実行
- [ ] トレーニング開始（nohup推奨）
- [ ] 進捗確認
- [ ] 結果ダウンロード

---

## 💡 Tips

### Colab切断対策

1. **定期的なバックアップ**（2-3時間ごと）
2. **ブラウザを開いたまま**
3. **チェックポイント自動保存**（5000 steps毎）

### Lightning AI最適化

1. **nohupでバックグラウンド実行**
2. **複数実験を並列実行**（リソースが許せば）
3. **結果は定期的にダウンロード**

---

**Lightning AIが最もおすすめです！切断リスクなし、コスパ良好、簡単セットアップ。🚀**
