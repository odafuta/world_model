#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightning AI H100 Setup Script
MATWM Training on Lightning AI Studio
"""

import os
import sys
import subprocess
from pathlib import Path

print('=' * 70)
print('Lightning AI H100 Setup for MATWM Training')
print('=' * 70)

# ============================================================
# 1. Environment Check
# ============================================================
print('\n[1/7] Checking environment...')

# Check if running on Lightning AI
if 'LIGHTNING_CLOUD_PROJECT_ID' in os.environ:
    print('✓ Running on Lightning AI')
    project_id = os.environ['LIGHTNING_CLOUD_PROJECT_ID']
    print(f'  Project ID: {project_id}')
else:
    print('⚠️  Not detected as Lightning AI environment')
    print('  This script is designed for Lightning AI Studio')

# Check GPU
import torch
print(f'\nPyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('⚠️  GPU not available')

# ============================================================
# 2. Create Working Directory
# ============================================================
print('\n[2/7] Creating working directory...')

# Lightning AI のホームディレクトリ
home_dir = Path.home()
work_dir = home_dir / 'matwm_project'
work_dir.mkdir(exist_ok=True)

print(f'✓ Working directory: {work_dir}')

# ============================================================
# 3. Upload Files Instructions
# ============================================================
print('\n[3/7] File upload instructions:')
print('''
Lightning AI Studio でファイルをアップロードする方法:

方法A: Web UI経由（推奨）
1. Lightning AI Studio の左サイドバーで "Files" をクリック
2. "Upload" ボタンをクリック
3. 以下のファイルを選択してアップロード:
   - train_gamma_true.py
   - train_gamma_false.py
   - matwm_implementation.py
   - matwm_agent.py
   - matwm_utils.py
   - curiosity_reward.py
   - requirements.txt

方法B: Git経由
1. GitHub/GitLabにプロジェクトをプッシュ
2. Lightning AI Studio のターミナルで:
   git clone YOUR_REPO_URL ~/matwm_project

方法C: Lightning AI CLI
1. ローカルPCで:
   lightning upload ~/matwm_project

現在のディレクトリ内容:
''')

# List current directory
if work_dir.exists():
    files = list(work_dir.glob('*'))
    if files:
        for f in files:
            print(f'  - {f.name}')
    else:
        print('  (empty)')

input('\nファイルのアップロードが完了したら Enter を押してください...')

# ============================================================
# 4. Verify Files
# ============================================================
print('\n[4/7] Verifying files...')

required_files = [
    'train_gamma_true.py',
    'train_gamma_false.py',
    'matwm_implementation.py',
    'matwm_agent.py',
    'matwm_utils.py',
    'curiosity_reward.py',
    'requirements.txt',
]

missing_files = []
for filename in required_files:
    filepath = work_dir / filename
    if filepath.exists():
        print(f'✓ {filename}')
    else:
        print(f'✗ {filename} (missing)')
        missing_files.append(filename)

if missing_files:
    print(f'\n⚠️  Missing files: {", ".join(missing_files)}')
    print('Please upload the missing files and run this script again.')
    sys.exit(1)
else:
    print('\n✓ All required files found')

# ============================================================
# 5. Install Dependencies
# ============================================================
print('\n[5/7] Installing dependencies...')

os.chdir(work_dir)

# Install from requirements.txt
print('Installing packages from requirements.txt...')
result = subprocess.run(
    ['pip', 'install', '-r', 'requirements.txt'],
    capture_output=True,
    text=True
)

if result.returncode == 0:
    print('✓ Dependencies installed successfully')
else:
    print('✗ Installation failed:')
    print(result.stderr)
    sys.exit(1)

# Verify installations
print('\nVerifying installations...')
result = subprocess.run(
    ['pip', 'list'],
    capture_output=True,
    text=True
)

for package in ['torch', 'pettingzoo', 'gymnasium', 'tqdm']:
    if package in result.stdout.lower():
        print(f'✓ {package}')
    else:
        print(f'✗ {package} (not found)')

# ============================================================
# 6. Test Imports
# ============================================================
print('\n[6/7] Testing imports...')

try:
    from matwm_implementation import MATWMConfig
    print('✓ matwm_implementation')
except Exception as e:
    print(f'✗ matwm_implementation: {e}')

try:
    from matwm_agent import MATWMAgent
    print('✓ matwm_agent')
except Exception as e:
    print(f'✗ matwm_agent: {e}')

try:
    from curiosity_reward import CuriosityConfig
    print('✓ curiosity_reward')
except Exception as e:
    print(f'✗ curiosity_reward: {e}')

try:
    from pettingzoo.mpe import simple_tag_v3
    print('✓ pettingzoo')
except Exception as e:
    print(f'✗ pettingzoo: {e}')

# ============================================================
# 7. Ready to Train
# ============================================================
print('\n[7/7] Setup complete!')
print('=' * 70)
print('Ready to start training')
print('=' * 70)

print(f'''
次のステップ:

1. use_gamma_progress=True で実行:
   cd {work_dir}
   python train_gamma_true.py 2>&1 | tee train_true.log

2. use_gamma_progress=False で実行:
   cd {work_dir}
   python train_gamma_false.py 2>&1 | tee train_false.log

3. バックグラウンドで実行（推奨）:
   nohup python train_gamma_true.py > train_true.log 2>&1 &
   
4. 進捗確認:
   tail -f train_true.log

5. 結果の確認:
   ls -lh results_gamma_true/

実行時間（H100使用時）:
- 約11-13時間/実験

Lightning AI のコスト:
- H100: 約$2-3/時間
- 総コスト: 約$22-39/実験
''')

print('\n✓ Setup script completed successfully!')
