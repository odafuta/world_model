# クイックスタート: Colab & Lightning AI

## 🎯 最速で実行開始する方法

### 推奨: Lightning AI（H100）

**実行時間**: 11-13時間  
**コスト**: $24-36  
**切断リスク**: なし

---

## 🚀 Lightning AI - 5ステップ

### 1. アカウント作成
https://lightning.ai/ → Sign Up

### 2. Studio作成
New Studio → GPU: **H100 (80GB)** → Create

### 3. ファイルアップロード
Files → Upload → 以下をアップロード:
- `train_gamma_true.py`
- `train_gamma_false.py`
- `matwm_implementation.py`
- `matwm_agent.py`
- `matwm_utils.py`
- `curiosity_reward.py`
- `requirements.txt`

### 4. セットアップ
ターミナルで:
```bash
cd ~
mkdir matwm_project
cd matwm_project
# (アップロードしたファイルをここに移動)

pip install -r requirements.txt
```

### 5. 実行
```bash
nohup python train_gamma_true.py > train_true.log 2>&1 &
tail -f train_true.log
```

**完了！11-13時間後に結果が得られます。**

---

## 🚀 Google Colab - 5ステップ

### 1. ファイル準備
ローカルPCで:
```cmd
cd "C:\Users\0622d\OneDrive - OUMail (Osaka University)\M1_秋冬\松尾研究室\WorldModel\最終課題"
powershell Compress-Archive -Path *.py,requirements.txt -DestinationPath MATWM_Project.zip
```

### 2. Google Driveにアップロード
`MATWM_Project.zip` を Google Drive にアップロード

### 3. Colab Notebook作成
https://colab.research.google.com/ → 新しいノートブック

### 4. GPU設定
ランタイム → ランタイムのタイプを変更 → GPU: **H100** (Pro+) または **A100** (Pro)

### 5. 実行
```python
# Google Drive マウント
from google.colab import drive
drive.mount('/content/drive')

# 解凍
!unzip /content/drive/MyDrive/MATWM_Project.zip -d /content/matwm_project

# インストール
%cd /content/matwm_project
!pip install -r requirements.txt

# 実行
!python train_gamma_true.py 2>&1 | tee train_true.log
```

**注意**: 12時間で切断されるため、バックアップが必要

---

## 📊 比較

| 項目 | Lightning AI | Colab Pro+ |
|------|-------------|-----------|
| GPU | H100 | H100/A100 |
| 実行時間 | 11-13時間 | 11-13時間 |
| コスト | $24-36 | $49.99/月 |
| 切断リスク | ❌ なし | ⚠️ あり（12時間） |
| 推奨度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

**推奨: Lightning AI（切断リスクなし、コスパ良好）**

---

## 📝 詳細ガイド

- **完全ガイド**: [COLAB_AND_LIGHTNING_GUIDE.md](./COLAB_AND_LIGHTNING_GUIDE.md)
- **Colab Notebook**: [colab_setup.ipynb](./colab_setup.ipynb)
- **Lightning Setup**: [lightning_ai_setup.py](./lightning_ai_setup.py)
