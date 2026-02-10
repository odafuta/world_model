#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MATWM Training Script with use_gamma_progress=False (TEST VERSION)
Short training for testing: total_steps=2000
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # For non-GUI environment
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Device Setup
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('=' * 70)
print('MATWM + Curiosity-Driven Training (γ-Progress=FALSE) - TEST VERSION')
print('=' * 70)
print(f'Device: {device}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

# ============================================================
# Environment Setup
# ============================================================
from pettingzoo.mpe import simple_tag_v3

def make_env(max_cycles=25, seed=None):
    env = simple_tag_v3.parallel_env(
        num_good=1,
        num_adversaries=3,
        num_obstacles=2,
        max_cycles=max_cycles,
        continuous_actions=False,
        render_mode=None,
    )
    if seed is not None:
        env.reset(seed=seed)
    return env

# ============================================================
# MATWM Implementation
# ============================================================
from matwm_implementation import MATWMConfig, pad_observation
from matwm_utils import (
    initialize_matwm_weights, init_weights,
    save_full_checkpoint, load_full_checkpoint,
    plot_training_progress,
)
from matwm_agent import MATWMAgent

# Configuration - TEST VERSION (short training)
config = MATWMConfig(
    total_steps=2000,       # TEST: Short training
    warmup_steps=200,       # TEST: Short warmup
    log_interval=50,
    save_interval=1000,
    use_gamma_progress=False,  # ★ FALSE for this script
)

print(f'\n⚠️  TEST VERSION - Short training (2000 steps)')
print(f'use_gamma_progress: {config.use_gamma_progress}')
print(f'Total steps: {config.total_steps}')
print(f'Warmup steps: {config.warmup_steps}')
print('=' * 70)

# ============================================================
# Curiosity Configuration
# ============================================================
from curiosity_reward import CuriosityConfig, CuriosityManager, create_curiosity_managers

# LLM API Key (optional, from environment variable)
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', None)
USE_LLM = OPENROUTER_API_KEY is not None and OPENROUTER_API_KEY != ''
MODEL_NAME = 'google/gemma-3-4b-it:free'
LOG_DIR = 'llm_logs_gamma_false_test'

curiosity_config = CuriosityConfig(
    # Computational curiosity weights
    dynamics_curiosity_weight=1.0,
    reward_curiosity_weight=0.5,
    social_curiosity_weight=2.0,
    
    # Decay method
    curiosity_decay_method='adaptive',
    curiosity_initial_weight=1.0,
    curiosity_min_weight=0.1,
    
    # LLM semantic curiosity (optional)
    use_llm_curiosity=USE_LLM,
    llm_api_key=OPENROUTER_API_KEY if USE_LLM else '',
    llm_model=MODEL_NAME,
    llm_eval_every_n_episodes=1,
    semantic_curiosity_weight=0.5,
    
    # Logging
    verbose=True,
    log_dir=LOG_DIR,
)

print(f'\n=== Curiosity Configuration ===')
print(f'  Social Curiosity Weight: {curiosity_config.social_curiosity_weight}')
print(f'  LLM enabled: {curiosity_config.use_llm_curiosity}')

# ============================================================
# Training Function (same as full version)
# ============================================================
def train_matwm_with_curiosity(config, curiosity_config, save_dir='results_gamma_false_test', resume_from=None):
    """MATWM + Curiosity-driven training without γ-Progress"""
    env = make_env(max_cycles=config.max_cycles, seed=42)
    agent_names = env.agents
    
    if config.use_gamma_progress:
        shared_wm, shared_wm_ema, shared_wm_opt = MATWMAgent.create_shared_world_model_with_ema(config, device)
        print(f'\nShared World Model: {sum(p.numel() for p in shared_wm.parameters())} params (γ-Progress enabled)')
    else:
        shared_wm, shared_wm_opt = MATWMAgent.create_shared_world_model(config, device)
        shared_wm_ema = None
        print(f'\nShared World Model: {sum(p.numel() for p in shared_wm.parameters())} params (γ-Progress disabled)')
    
    agents = {}
    for idx, name in enumerate(agent_names):
        agents[name] = MATWMAgent(config, name, idx, device, shared_world_model=shared_wm)
    
    if resume_from is None:
        print('\n=== Initializing Weights ===')
        initialize_matwm_weights(shared_wm,
                                 list(agents.values())[0].actor,
                                 list(agents.values())[0].critic)
        for agent in agents.values():
            agent.actor.apply(init_weights)
            agent.critic.apply(init_weights)
        print('✓ Weight initialization complete')
    
    curiosity_managers = create_curiosity_managers(
        agent_names,
        curiosity_config,
        matwm_config=config,
        world_model=shared_wm,
        world_model_ema=shared_wm_ema,
        device=device,
    )
    
    episode_rewards = {name: [] for name in agent_names}
    episode_curiosity = {name: [] for name in agent_names}
    training_metrics = defaultdict(list)
    curiosity_metrics = defaultdict(list)
    start_step = 0
    
    if resume_from is not None and os.path.exists(resume_from):
        print(f'\n=== Resuming from: {resume_from} ===')
        episode_rewards, training_metrics, start_step = load_full_checkpoint(
            agents, shared_wm, shared_wm_opt, resume_from, device
        )
        print(f'✓ Resumed from step {start_step}')
    
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(save_dir, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    print(f'\n=== Starting Training ===')
    print(f'Save directory: {run_dir}\n')
    
    global_step = start_step
    episode_count = 0
    min_data = config.wm_batch_length + 10
    pbar = tqdm(total=config.total_steps, initial=start_step, desc='Training')
    
    while global_step < config.total_steps:
        obs, info = env.reset()
        ep_reward = {name: 0.0 for name in agent_names}
        ep_intrinsic = {name: 0.0 for name in agent_names}
        
        for name in agent_names:
            curiosity_managers[name].reset_episode(episode_count)
        
        for step in range(config.max_cycles):
            actions = {}
            for name, agent in agents.items():
                if global_step < config.warmup_steps:
                    actions[name] = env.action_space(name).sample()
                else:
                    actions[name] = agent.select_action(obs[name])
            
            next_obs, rewards, terms, truncs, infos = env.step(actions)
            done = {name: terms[name] or truncs[name] for name in agent_names}
            
            for name, agent in agents.items():
                other_acts = {k: v for k, v in actions.items() if k != name}
                env_r = rewards[name]
                
                intrinsic_r = 0.0
                if global_step >= min_data:
                    obs_padded = pad_observation(obs[name], config.max_obs_dim)
                    next_obs_padded = pad_observation(next_obs[name], config.max_obs_dim)
                    intrinsic_r = curiosity_managers[name].compute_intrinsic_reward(
                        shared_wm, obs_padded, actions[name],
                        env_r, next_obs_padded, other_acts, device,
                        done=done[name],
                    )
                
                total_r = env_r + intrinsic_r
                
                agent.store_experience(
                    obs[name], actions[name], total_r,
                    next_obs[name], done[name], other_acts,
                )
                ep_reward[name] += env_r
                ep_intrinsic[name] += intrinsic_r
            
            obs = next_obs
            global_step += 1
            pbar.update(1)
            
            if global_step >= config.warmup_steps:
                wm_metrics = MATWMAgent.train_world_model_shared(
                    agents, config, device, shared_wm_opt
                )
                if wm_metrics:
                    for k, v in wm_metrics.items():
                        training_metrics[f'shared_{k}'].append(v)
                
                if config.use_gamma_progress and shared_wm_ema is not None:
                    MATWMAgent.update_shared_world_model_ema(
                        shared_wm, shared_wm_ema, config.gamma_progress
                    )
                
                for name, agent in agents.items():
                    ac_metrics = agent.train_agent()
                    for k, v in ac_metrics.items():
                        training_metrics[f'{name}_{k}'].append(v)
            
            if global_step % config.log_interval == 0 and global_step >= config.warmup_steps:
                log_str = f'Step {global_step}: '
                for name in agent_names:
                    if episode_rewards[name]:
                        log_str += f'{name}={np.mean(episode_rewards[name][-10:]):.2f} '
                pbar.set_description(log_str)
            
            if global_step % config.save_interval == 0 and global_step >= config.warmup_steps:
                ckpt_dir = os.path.join(run_dir, f'checkpoint_{global_step}')
                os.makedirs(ckpt_dir, exist_ok=True)
                for name, agent in agents.items():
                    agent.save(os.path.join(ckpt_dir, f'{name}.pt'))
                save_full_checkpoint(
                    agents, shared_wm, shared_wm_opt,
                    episode_rewards, training_metrics, global_step,
                    os.path.join(ckpt_dir, 'full_checkpoint.pt')
                )
                print(f'\n✓ Checkpoint saved at step {global_step}')
            
            if all(done.values()):
                break
        
        for name in agent_names:
            llm_result = curiosity_managers[name].end_episode()
            summary = curiosity_managers[name].get_episode_summary()
            
            episode_rewards[name].append(ep_reward[name])
            episode_curiosity[name].append(ep_intrinsic[name])
            
            for k, v in summary.items():
                if isinstance(v, (int, float)):
                    curiosity_metrics[f'{name}_{k}'].append(v)
        
        episode_count += 1
    
    pbar.close()
    env.close()
    
    final_dir = os.path.join(run_dir, 'final')
    os.makedirs(final_dir, exist_ok=True)
    for name, agent in agents.items():
        agent.save(os.path.join(final_dir, f'{name}.pt'))
    save_full_checkpoint(
        agents, shared_wm, shared_wm_opt,
        episode_rewards, training_metrics, global_step,
        os.path.join(final_dir, 'full_checkpoint.pt')
    )
    
    print(f'\n{"=" * 70}')
    print('Training Complete!')
    print(f'{"=" * 70}')
    print(f'Total episodes: {episode_count}')
    print(f'Final checkpoint: {final_dir}')
    for name in agent_names:
        if episode_rewards[name]:
            r = episode_rewards[name][-100:] if len(episode_rewards[name]) >= 100 else episode_rewards[name]
            print(f'  {name}: mean reward = {np.mean(r):.2f}')
    
    return agents, episode_rewards, training_metrics, episode_curiosity, curiosity_metrics

# ============================================================
# Main Execution
# ============================================================
if __name__ == '__main__':
    print('\nStarting TEST training (2000 steps)...\n')
    
    agents, episode_rewards, training_metrics, episode_curiosity, curiosity_metrics = \
        train_matwm_with_curiosity(config, curiosity_config, save_dir='results_gamma_false_test')
    
    print(f'\n✓ TEST training completed!')
