# Experiments Directory

This directory contains different experimental configurations for MATWM training.

## 📁 Directory Structure

```
experiments/
├── llm_curiosity_and_γ-progress/  # LLM + γ-Progress
│   ├── train_llm_and_gamma.py              (full training: 50000 steps)
│   ├── train_llm_and_gamma_test.py         (test: 2000 steps)
│   ├── 2026_MATWM_simple_tag_Implementation_gamma_true.ipynb
│   └── 2026_MATWM_simple_tag_Implementation_gamma_true_test.ipynb
│
├── only_llm_curiosity/            # Only LLM (γ-Progress disabled)
│   ├── train_only_llm.py                    (full training: 50000 steps)
│   ├── train_only_llm_test.py               (test: 2000 steps)
│   ├── 2026_MATWM_simple_tag_Implementation_gamma_false.ipynb
│   └── 2026_MATWM_simple_tag_Implementation_gamma_false_test.ipynb
│
└── only_γ_progress/               # Only γ-Progress (LLM disabled)
    ├── train_only_gamma.py                  (full training: 50000 steps)
    ├── train_only_gamma_test.py             (test: 2000 steps)
    ├── 2026_MATWM_simple_tag_Implementation_oly_gamma_true.ipynb
    └── 2026_MATWM_simple_tag_Implementation_only_gamma_true_test.ipynb
```

## 🧪 Experiment Methods

### 1. `llm_curiosity_and_γ-progress/`
- **γ-Progress**: ✅ Enabled (`use_gamma_progress=True`)
- **LLM Curiosity**: ✅ Enabled (if API key is set)
- **Purpose**: Test the full proposed approach with both mechanisms

### 2. `only_llm_curiosity/`
- **γ-Progress**: ❌ Disabled (`use_gamma_progress=False`)
- **LLM Curiosity**: ✅ Enabled (if API key is set)
- **Purpose**: Ablation study - LLM curiosity only

### 3. `only_γ_progress/`
- **γ-Progress**: ✅ Enabled (`use_gamma_progress=True`)
- **LLM Curiosity**: ❌ Disabled
- **Purpose**: Ablation study - γ-Progress only

## 📂 Output Directory Structure

Each experiment creates organized output directories:

```
results/
├── llm_and_gamma/
│   └── {timestamp}/
│       ├── checkpoint_1000/
│       ├── checkpoint_2000/
│       ├── ...
│       └── final/
├── only_llm/
│   └── {timestamp}/
│       └── ...
└── only_gamma/
    └── {timestamp}/
        └── ...

llm_logs/
├── llm_and_gamma/
│   └── {timestamp}/
│       ├── curiosity_adversary_0_{timestamp}.jsonl
│       ├── curiosity_adversary_1_{timestamp}.jsonl
│       ├── curiosity_adversary_2_{timestamp}.jsonl
│       └── curiosity_agent_0_{timestamp}.jsonl
├── only_llm/
│   └── {timestamp}/
│       └── ...
└── only_gamma/
    └── {timestamp}/
        └── ... (empty if LLM disabled)
```

## 🚀 Running Experiments

### Method 1: Jupyter Notebook (Recommended)

**Start JupyterLab:**
```bash
jupyter lab
```

**Select and run a notebook:**
1. Navigate to `experiments/` directory
2. Choose experiment method:
   - `llm_curiosity_and_γ-progress/`
   - `only_llm_curiosity/`
   - `only_γ_progress/`
3. Open notebook:
   - Test version (2000 steps): `*_test.ipynb`
   - Full version (50000 steps): `*.ipynb` (without `_test`)
4. Run cells sequentially (Shift + Enter)

**For LLM experiments:**
- Set `OPENROUTER_API_KEY` environment variable before starting Jupyter
- Or modify the `OPENROUTER_API_KEY` variable in the notebook cell

### Method 2: Python Scripts (Legacy)

### Test Version (Quick validation: 2000 steps)

```bash
# From project root
cd "experiments/llm_curiosity_and_γ-progress"
python train_llm_and_gamma_test.py

cd "../only_llm_curiosity"
python train_only_llm_test.py

cd "../only_γ_progress"
python train_only_gamma_test.py
```

### Full Training (50000 steps)

```bash
# Set API key (if using LLM)
export OPENROUTER_API_KEY="sk-or-v1-..."  # Linux/Mac
# or
set OPENROUTER_API_KEY=sk-or-v1-...      # Windows

# Run experiments
cd "experiments/llm_curiosity_and_γ-progress"
python train_llm_and_gamma.py

cd "../only_llm_curiosity"
python train_only_llm.py

cd "../only_γ_progress"
python train_only_gamma.py
```

## 📊 Output Files

Each run creates:

1. **Checkpoints**: `results/{method}/{timestamp}/checkpoint_{step}/`
   - Full model states
   - Training metrics
   - Episode rewards

2. **Final Model**: `results/{method}/{timestamp}/final/`
   - Final trained agents
   - Complete training history

3. **LLM Logs**: `llm_logs/{method}/{timestamp}/` (if LLM enabled)
   - JSONL format (1 line per episode)
   - Semantic curiosity evaluations
   - Exploration phase recommendations

## 🔧 Configuration

All experiments share the same base configuration from `train_base.py`:
- Environment: PettingZoo simple_tag_v3
- Agents: 4 (3 adversaries + 1 good agent)
- Max cycles per episode: 25
- Social curiosity weight: 2.0 (emphasized)

## 📝 Notes

- **Test versions** run for 2000 steps (~5-10 minutes on GPU)
- **Full versions** run for 50000 steps (~2-3 hours on GPU, 11-13 hours on H100)
- Results are automatically organized by method and timestamp
- No manual directory management needed
