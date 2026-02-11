#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MATWM Training Script: Only γ-Progress (TEST VERSION)
Short training for testing: total_steps=2000
"""

import os
import sys
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from train_base import train_matwm_with_curiosity, MATWMConfig, CuriosityConfig

# Experiment configuration
EXPERIMENT_METHOD = 'only_gamma'
TIMESTAMP = time.strftime('%Y%m%d_%H%M%S')

# MATWM Configuration - TEST VERSION
config = MATWMConfig(
    total_steps=2000,       # TEST: Short training
    warmup_steps=200,       # TEST: Short warmup
    log_interval=50,
    save_interval=1000,
    use_gamma_progress=True,  # γ-Progress enabled
)

# Curiosity Configuration
# LLM is disabled for this experiment
LLM_LOG_DIR = f'llm_logs/{EXPERIMENT_METHOD}_test/{TIMESTAMP}'
RESULTS_DIR = f'results/{EXPERIMENT_METHOD}_test/{TIMESTAMP}'

curiosity_config = CuriosityConfig(
    # Computational curiosity weights
    dynamics_curiosity_weight=1.0,
    reward_curiosity_weight=0.5,
    social_curiosity_weight=2.0,
    
    # Decay method
    curiosity_decay_method='adaptive',
    curiosity_initial_weight=1.0,
    curiosity_min_weight=0.1,
    
    # LLM semantic curiosity - DISABLED
    use_llm_curiosity=False,
    llm_api_key='',
    llm_model='',
    llm_eval_every_n_episodes=1,
    semantic_curiosity_weight=0.0,
    
    # Logging
    verbose=True,
    log_dir=LLM_LOG_DIR,
)

print('=' * 70)
print(f'[TEST VERSION] Experiment: {EXPERIMENT_METHOD}')
print(f'Timestamp: {TIMESTAMP}')
print(f'Training steps: {config.total_steps} (test mode)')
print(f'γ-Progress: {config.use_gamma_progress}')
print(f'LLM Curiosity: {curiosity_config.use_llm_curiosity}')
print(f'Results dir: {RESULTS_DIR}')
print(f'LLM logs dir: {LLM_LOG_DIR}')
print('=' * 70)

if __name__ == '__main__':
    agents, episode_rewards, training_metrics, episode_curiosity, curiosity_metrics = \
        train_matwm_with_curiosity(config, curiosity_config, save_dir=RESULTS_DIR)
    
    print(f'\n[OK] TEST training completed!')
    print(f'Results saved to: {RESULTS_DIR}/')
    print(f'LLM logs saved to: {LLM_LOG_DIR}/')
