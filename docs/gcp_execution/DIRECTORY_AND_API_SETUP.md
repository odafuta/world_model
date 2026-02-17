# ディレクトリ保存とAPIキー設定ガイド

## 📁 ディレクトリ保存について

### ✅ 現在の設定（このままで大丈夫）

現在のコードは**相対パスを使用**しているため、GCPでも問題なく動作します。

```python
# train_gamma_true.py の設定
save_dir='results_gamma_true'  # 相対パス（カレントディレクトリに作成）
```

### 実行時のディレクトリ構成

```
~/matwm_project/              # 作業ディレクトリ
├── train_gamma_true.py       # 実行スクリプト
├── train_gamma_false.py
├── matwm_implementation.py
├── matwm_agent.py
├── matwm_utils.py
├── curiosity_reward.py
├── requirements.txt
│
├── results_gamma_true/       # ← 自動作成される（True版）
│   └── run_20260211_100000/
│       ├── checkpoint_5000/
│       ├── checkpoint_10000/
│       └── final/
│
├── results_gamma_false/      # ← 自動作成される（False版）
│   └── run_20260211_100000/
│       ├── checkpoint_5000/
│       └── final/
│
└── llm_logs_gamma_true/      # ← LLM使用時のみ作成
    └── adversary_0.jsonl
```

### ディレクトリ作成のコード

```python
# train_gamma_true.py 内で自動的に実行される
os.makedirs(save_dir, exist_ok=True)  # results_gamma_true/ を作成
timestamp = time.strftime('%Y%m%d_%H%M%S')
run_dir = os.path.join(save_dir, f'run_{timestamp}')
os.makedirs(run_dir, exist_ok=True)  # run_YYYYMMDD_HHMMSS/ を作成
```

### ⚠️ 変更が必要な場合

もし**絶対パス**を使いたい場合のみ変更してください：

```python
# 変更例（通常は不要）
save_dir='/home/username/matwm_results/results_gamma_true'
```

**推奨:** 相対パスのまま（変更不要）

---

## 🔑 APIキー設定ガイド（LLM Curiosity使用時）

### APIキーの必要性

| 機能 | APIキー必要 | 説明 |
|------|------------|------|
| **計算型好奇心** | ❌ 不要 | Dynamics/Reward/Social Curiosity（常に有効） |
| **LLM意味的好奇心** | ✅ 必要 | OpenRouter API経由でLLMを使用 |

**重要:** APIキーなしでも実行可能です（計算型好奇心のみ使用）

---

## 🚀 ステップ1: APIキーなしで実行（推奨）

### 最も簡単な方法

**何もしない** → APIキーなしで実行

```bash
# GCP VM上で
cd ~/matwm_project
python3 train_gamma_true.py 2>&1 | tee train_true.log
```

**出力例:**
```
=== Curiosity Configuration ===
  Social Curiosity Weight: 2.0 ★
  Dynamics Curiosity Weight: 1.0
  Decay method: adaptive
  LLM enabled: False  # ← APIキーなしの場合
```

**メリット:**
- ✅ 設定不要
- ✅ 追加コストなし
- ✅ 計算型好奇心（Dynamics/Reward/Social）は完全に動作
- ✅ 論文再現に十分

**デメリット:**
- LLM意味的好奇心は使用されない（オプション機能）

---

## 🚀 ステップ2: APIキーを使用する場合（オプション）

### 2-1. OpenRouter APIキーの取得

#### ① OpenRouterアカウント作成

1. ブラウザで https://openrouter.ai/ を開く
2. **Sign Up** をクリック
3. Googleアカウントまたはメールアドレスで登録
4. メール認証を完了

#### ② APIキーの生成

1. ログイン後、右上のアカウントアイコンをクリック
2. **API Keys** を選択
3. **Create New Key** をクリック
4. キー名を入力（例: `matwm_training`）
5. **Create** をクリック
6. 表示されたAPIキーをコピー（例: `sk-or-v1-abc123...`）

⚠️ **重要:** APIキーは一度しか表示されません。必ず保存してください。

#### ③ クレジット追加（必要な場合）

1. **Credits** タブを開く
2. **Add Credits** をクリック
3. 最低$5から追加可能
4. クレジットカードで支払い

**コスト見積もり:**
- 使用モデル: `google/gemma-3-4b-it:free` → **無料**
- 有料モデルを使う場合: 約$0.01-0.10/エピソード

---

### 2-2. GCP VMでのAPIキー設定

#### 方法A: 環境変数として設定（推奨）

**SSH接続後、VM上で:**

```bash
# 環境変数を設定
export OPENROUTER_API_KEY="sk-or-v1-YOUR_API_KEY_HERE"

# 確認
echo $OPENROUTER_API_KEY
```

**永続化（再起動後も有効）:**

```bash
# ~/.bashrc に追加
echo 'export OPENROUTER_API_KEY="sk-or-v1-YOUR_API_KEY_HERE"' >> ~/.bashrc

# 設定を反映
source ~/.bashrc

# 確認
echo $OPENROUTER_API_KEY
```

**実行:**

```bash
cd ~/matwm_project
python3 train_gamma_true.py 2>&1 | tee train_true.log
```

**出力例:**
```
=== Curiosity Configuration ===
  Social Curiosity Weight: 2.0 ★
  Dynamics Curiosity Weight: 1.0
  Decay method: adaptive
  LLM enabled: True  # ← APIキーありの場合
  LLM model: google/gemma-3-4b-it:free
```

#### 方法B: 実行時に指定

```bash
# 一時的に設定して実行
OPENROUTER_API_KEY="sk-or-v1-YOUR_API_KEY_HERE" python3 train_gamma_true.py 2>&1 | tee train_true.log
```

#### 方法C: スクリプトに直接記述（非推奨）

```python
# train_gamma_true.py を編集（セキュリティリスクあり）
OPENROUTER_API_KEY = "sk-or-v1-YOUR_API_KEY_HERE"  # 直接記述
```

⚠️ **非推奨理由:**
- APIキーがコードに含まれる
- Gitにコミットすると漏洩リスク
- 環境変数を使う方が安全

---

### 2-3. APIキー設定の確認

```bash
# Python で確認
python3 -c "
import os
key = os.environ.get('OPENROUTER_API_KEY', None)
if key:
    print(f'✓ API Key set: {key[:20]}...')
else:
    print('✗ API Key not set')
"
```

**期待される出力:**
```
✓ API Key set: sk-or-v1-abc123def456...
```

---

## 🔍 APIキーの動作確認

### テストスクリプト

```bash
# test_api.py を作成
cat > test_api.py << 'EOF'
import os
import requests

api_key = os.environ.get('OPENROUTER_API_KEY', None)

if not api_key:
    print('✗ API Key not set')
    exit(1)

print(f'✓ API Key: {api_key[:20]}...')

# OpenRouter API テスト
url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
data = {
    "model": "google/gemma-3-4b-it:free",
    "messages": [{"role": "user", "content": "Hello"}]
}

try:
    response = requests.post(url, headers=headers, json=data, timeout=10)
    if response.status_code == 200:
        print('✓ API connection successful')
    else:
        print(f'✗ API error: {response.status_code}')
        print(response.text)
except Exception as e:
    print(f'✗ Connection failed: {e}')
EOF

# 実行
python3 test_api.py
```

**期待される出力:**
```
✓ API Key: sk-or-v1-abc123def456...
✓ API connection successful
```

---

## 📊 APIキー使用時のコスト

### 無料モデル使用時（推奨）

```python
MODEL_NAME = 'google/gemma-3-4b-it:free'  # デフォルト設定
```

**コスト:** $0（完全無料）

### 有料モデル使用時

| モデル | コスト/1Mトークン | 50000 steps予想 |
|--------|-----------------|----------------|
| GPT-3.5-turbo | $0.50 | 約$2-5 |
| GPT-4 | $30.00 | 約$120-300 |
| Claude-3-Haiku | $0.25 | 約$1-3 |

**推奨:** 無料モデル（`google/gemma-3-4b-it:free`）を使用

---

## 🎯 実行パターン別の設定

### パターン1: APIキーなし（最も簡単）

```bash
# 何もしない
cd ~/matwm_project
python3 train_gamma_true.py 2>&1 | tee train_true.log
```

**結果:**
- 計算型好奇心: ✅ 有効
- LLM意味的好奇心: ❌ 無効

### パターン2: APIキーあり（無料モデル）

```bash
# 環境変数設定
export OPENROUTER_API_KEY="sk-or-v1-YOUR_API_KEY_HERE"

# 実行
python3 train_gamma_true.py 2>&1 | tee train_true.log
```

**結果:**
- 計算型好奇心: ✅ 有効
- LLM意味的好奇心: ✅ 有効（無料）

### パターン3: APIキーあり（有料モデル）

```python
# train_gamma_true.py を編集
MODEL_NAME = 'openai/gpt-3.5-turbo'  # 有料モデルに変更
```

```bash
export OPENROUTER_API_KEY="sk-or-v1-YOUR_API_KEY_HERE"
python3 train_gamma_true.py 2>&1 | tee train_true.log
```

**結果:**
- 計算型好奇心: ✅ 有効
- LLM意味的好奇心: ✅ 有効（有料）

---

## 🔒 セキュリティのベストプラクティス

### ✅ 推奨

1. **環境変数を使用**
   ```bash
   export OPENROUTER_API_KEY="sk-or-v1-..."
   ```

2. **~/.bashrc に保存（VM上のみ）**
   ```bash
   echo 'export OPENROUTER_API_KEY="sk-or-v1-..."' >> ~/.bashrc
   ```

3. **APIキーをGitにコミットしない**
   ```bash
   # .gitignore に追加
   echo '*.env' >> .gitignore
   echo '.env*' >> .gitignore
   ```

### ❌ 避けるべき

1. **スクリプトに直接記述**
   ```python
   # ❌ 避ける
   OPENROUTER_API_KEY = "sk-or-v1-..."
   ```

2. **公開リポジトリにプッシュ**
   - APIキーが含まれるファイルをGitHubにアップロードしない

3. **ログファイルに出力**
   - APIキーがログに記録されないように注意

---

## 📝 チェックリスト

### APIキーなしで実行する場合
- [ ] 何もしない → そのまま実行

### APIキーありで実行する場合
- [ ] OpenRouterアカウント作成
- [ ] APIキー生成
- [ ] APIキーをコピー
- [ ] GCP VMで環境変数設定
- [ ] 動作確認（test_api.py）
- [ ] トレーニング実行

---

## 🎯 推奨設定（まとめ）

### 最も簡単で推奨: APIキーなし

```bash
# GCP VM上で
cd ~/matwm_project
python3 train_gamma_true.py 2>&1 | tee train_true.log
```

**理由:**
- ✅ 設定不要
- ✅ 追加コストなし
- ✅ 計算型好奇心で十分な性能
- ✅ 論文再現に必要な機能はすべて動作

### LLMを使いたい場合: 無料モデル

```bash
# 環境変数設定
export OPENROUTER_API_KEY="sk-or-v1-YOUR_API_KEY_HERE"

# 実行
python3 train_gamma_true.py 2>&1 | tee train_true.log
```

**理由:**
- ✅ 無料（`google/gemma-3-4b-it:free`）
- ✅ LLM意味的好奇心も使用可能
- ✅ 追加コストなし

---

## 💡 よくある質問

### Q1: APIキーなしで実行しても大丈夫？
**A:** はい、問題ありません。計算型好奇心（Dynamics/Reward/Social）は完全に動作します。

### Q2: ディレクトリは自動作成される？
**A:** はい、`results_gamma_true/` は自動的に作成されます。変更不要です。

### Q3: APIキーの料金は？
**A:** デフォルトの無料モデル（`google/gemma-3-4b-it:free`）を使えば$0です。

### Q4: 環境変数はどこに設定する？
**A:** GCP VMにSSH接続後、`export OPENROUTER_API_KEY="..."`を実行してください。

### Q5: 永続化する方法は？
**A:** `~/.bashrc`に追加してください：
```bash
echo 'export OPENROUTER_API_KEY="sk-or-v1-..."' >> ~/.bashrc
source ~/.bashrc
```

---

**ディレクトリ保存は現在の設定のままで問題なし！APIキーは不要（オプション）です。🚀**
