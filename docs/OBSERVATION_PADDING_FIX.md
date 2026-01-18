# 観測サイズ統一の実装（ゼロパディング）

## 問題

`simple_tag_v3`環境では、エージェントごとに観測サイズが異なります：
- **adversary_0, 1, 2**: 16次元
- **agent_0**: 14次元

この違いにより、共有World Modelのエンコーダーが固定サイズの入力を期待するため、そのままでは動作しません。

## 論文での言及

MATWM論文（arXiv:2506.18537）では、異なる観測サイズの処理について特に言及されていません。論文で評価されている環境（SMAC、PettingZoo、Melting Pot）では、同一環境内の全エージェントが同じ観測サイズを持つと考えられます。

## 解決策

**ゼロパディング**により、全エージェントの観測を最大サイズ（16次元）に統一します。これは以下の理由で適切です：

1. **情報の保持**: 元の観測データは保持され、追加部分のみゼロで埋められる
2. **ニューラルネットワークとの親和性**: ゼロパディングは標準的な手法で、ネットワークは学習を通じてパディング部分を無視することを学習できる
3. **実装の簡潔性**: 各エージェント用に別々のエンコーダーを用意するより効率的

## 実装の詳細

### 1. `matwm_implementation.py`

#### `MATWMConfig`に`max_obs_dim`を追加
```python
# Unified observation dimension (max across all agents, for zero-padding)
max_obs_dim: int = 16  # Max of all obs_dims values
```

#### `pad_observation()`関数を追加
```python
def pad_observation(obs, target_dim):
    """
    Zero-pad observation to target dimension.
    
    Args:
        obs: Observation array/tensor (shape: [..., obs_dim])
        target_dim: Target dimension after padding
    
    Returns:
        Padded observation (shape: [..., target_dim])
    """
    # NumPy配列とPyTorch Tensorの両方に対応
    # 元のサイズより小さい場合は、末尾にゼロを追加
    # 元のサイズ以上の場合は、切り詰め（通常は発生しない）
```

#### `WorldModel`で統一サイズを使用
```python
def __init__(self, config, agent_name):
    # Use unified max_obs_dim for all agents (with zero-padding)
    obs_dim = config.max_obs_dim  # 以前: config.obs_dims[agent_name]
    
    self.encoder = Encoder(obs_dim, ...)
    self.decoder = Decoder(..., obs_dim, ...)
```

### 2. `matwm_agent.py`

#### `pad_observation`をインポート
```python
from matwm_implementation import (
    ...,
    pad_observation
)
```

#### `select_action()`でパディング適用
```python
def select_action(self, obs, deterministic=False):
    with torch.no_grad():
        # Zero-pad observation to max_obs_dim
        obs_padded = pad_observation(obs, self.config.max_obs_dim)
        obs_tensor = torch.FloatTensor(obs_padded).unsqueeze(0).to(self.device)
        ...
```

#### `store_experience()`でパディング適用
```python
def store_experience(self, obs, action, reward, next_obs, done, other_actions):
    # Zero-pad observations to max_obs_dim
    obs_padded = pad_observation(obs, self.config.max_obs_dim)
    next_obs_padded = pad_observation(next_obs, self.config.max_obs_dim)
    
    experience = {
        'obs': obs_padded,
        ...
        'next_obs': next_obs_padded,
        ...
    }
```

### 3. `2026_MATWM_simple_tag_Implementation.ipynb`

- `pad_observation`を`matwm_implementation`からインポート
- 観測パディングのテストセルを追加（Cell 6-7）
- `MATWMAgent`クラスが内部でパディングを処理するため、訓練ループの変更は不要

## 動作確認

Notebookで以下のテストを実行：
```python
# 14次元の観測をパディング
test_obs_14 = np.random.randn(14)
padded_14 = pad_observation(test_obs_14, 16)
# → shape=(16,), 最後の2要素は0

# 16次元の観測は変更なし
test_obs_16 = np.random.randn(16)
padded_16 = pad_observation(test_obs_16, 16)
# → shape=(16,), 元の値と同一
```

## まとめ

✓ 全エージェントの観測が16次元に統一される  
✓ World Modelのエンコーダー/デコーダーが固定サイズ入力で動作  
✓ エージェント固有の観測情報は保持される  
✓ 実装はシンプルで効率的

この修正により、simple_tag環境でMATWMが正しく動作するようになります。
