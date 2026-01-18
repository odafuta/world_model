"""
Utility functions for MATWM implementation
- Weight initialization
- Model inspection
- Visualization
- GPU environment info

Naming Convention:
- Public API (外部から呼び出す関数): 通常の名前
- Internal helpers (内部ヘルパー関数): _で始まる名前
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import subprocess
import platform


# ============================================================================
# 1. Weight Initialization
# ============================================================================

def init_weights(module):
    """
    Initialize weights using Xavier/Kaiming initialization.
    Based on MATWM paper recommendations.
    
    PUBLIC API: 個別モジュールの重み初期化（外部から直接使用可能）
    
    Args:
        module: PyTorch module to initialize
    """
    if isinstance(module, nn.Linear):
        # Xavier uniform for linear layers (used in Actor/Critic)
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    
    elif isinstance(module, nn.Conv2d):
        # Kaiming initialization for conv layers (if used)
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    
    elif isinstance(module, nn.Embedding):
        # Normal initialization for embeddings
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
        # Standard normalization layer initialization
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    
    elif isinstance(module, nn.MultiheadAttention):
        # Xavier for attention layers
        if hasattr(module, 'in_proj_weight') and module.in_proj_weight is not None:
            nn.init.xavier_uniform_(module.in_proj_weight)
        if hasattr(module, 'out_proj') and module.out_proj.weight is not None:
            nn.init.xavier_uniform_(module.out_proj.weight)


def initialize_matwm_weights(world_model, actor, critic):
    """
    Initialize all MATWM components with proper weight initialization.
    
    PUBLIC API: 外部から呼び出される主要関数
    
    Args:
        world_model: WorldModel instance
        actor: Actor instance
        critic: Critic instance
    """
    print("Initializing MATWM weights...")
    world_model.apply(init_weights)
    actor.apply(init_weights)
    critic.apply(init_weights)
    print("✓ Weight initialization complete")


# ============================================================================
# 2. Model Saving/Loading (Enhanced)
# ============================================================================

def save_full_checkpoint(agents, shared_world_model, shared_wm_optimizer, 
                         episode_rewards, training_metrics, global_step, path):
    """
    Save complete training state including all agents and metrics.
    
    PUBLIC API: 外部から呼び出される主要関数
    
    Args:
        agents: Dictionary of MATWMAgent instances
        shared_world_model: Shared world model instance
        shared_wm_optimizer: Optimizer for shared world model
        episode_rewards: Episode reward history
        training_metrics: Training metrics history
        global_step: Current training step
        path: Save path
    """
    checkpoint = {
        'global_step': global_step,
        'shared_world_model': shared_world_model.state_dict(),
        'shared_wm_optimizer': shared_wm_optimizer.state_dict(),
        'episode_rewards': episode_rewards,
        'training_metrics': training_metrics,
        'agents': {}
    }
    
    for name, agent in agents.items():
        checkpoint['agents'][name] = {
            'actor': agent.actor.state_dict(),
            'critic': agent.critic.state_dict(),
            'actor_optimizer': agent.actor_optimizer.state_dict(),
            'critic_optimizer': agent.critic_optimizer.state_dict(),
        }
    
    torch.save(checkpoint, path)
    print(f"✓ Full checkpoint saved to {path}")


def load_full_checkpoint(agents, shared_world_model, shared_wm_optimizer, path, device):
    """
    Load complete training state.
    
    PUBLIC API: 外部から呼び出される主要関数
    
    Returns:
        tuple: (episode_rewards, training_metrics, global_step)
    """
    checkpoint = torch.load(path, map_location=device)
    
    shared_world_model.load_state_dict(checkpoint['shared_world_model'])
    shared_wm_optimizer.load_state_dict(checkpoint['shared_wm_optimizer'])
    
    for name, agent in agents.items():
        if name in checkpoint['agents']:
            agent.actor.load_state_dict(checkpoint['agents'][name]['actor'])
            agent.critic.load_state_dict(checkpoint['agents'][name]['critic'])
            agent.actor_optimizer.load_state_dict(checkpoint['agents'][name]['actor_optimizer'])
            agent.critic_optimizer.load_state_dict(checkpoint['agents'][name]['critic_optimizer'])
    
    print(f"✓ Full checkpoint loaded from {path}")
    
    return (
        checkpoint.get('episode_rewards', {}),
        checkpoint.get('training_metrics', {}),
        checkpoint.get('global_step', 0)
    )


# ============================================================================
# 3. Visualization
# ============================================================================

def plot_training_progress(episode_rewards, training_metrics, save_path='training_curves.png'):
    """
    Plot comprehensive training progress visualization.
    
    PUBLIC API: 外部から呼び出される主要関数
    
    Args:
        episode_rewards: Dictionary of episode rewards per agent
        training_metrics: Dictionary of training metrics
        save_path: Path to save the figure
    """
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Episode Rewards (Moving Average)
    ax1 = plt.subplot(3, 3, 1)
    for name, rewards in episode_rewards.items():
        if len(rewards) > 0:
            window = min(50, len(rewards))
            if len(rewards) >= window:
                ma_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax1.plot(ma_rewards, label=name, linewidth=2)
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel(f'Reward (MA-{window})', fontsize=12)
    ax1.set_title('Episode Rewards', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. World Model Total Loss
    ax2 = plt.subplot(3, 3, 2)
    key = 'shared_wm_total_loss'
    if key in training_metrics and len(training_metrics[key]) > 0:
        ax2.plot(training_metrics[key], label='Shared WM', alpha=0.8, linewidth=1.5)
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('World Model Total Loss', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Teammate Prediction Loss (★ Social World Model)
    ax3 = plt.subplot(3, 3, 3)
    key = 'shared_wm_teammate_loss'
    if key in training_metrics and len(training_metrics[key]) > 0:
        ax3.plot(training_metrics[key], label='Teammate Pred', color='purple', alpha=0.8, linewidth=1.5)
    ax3.set_xlabel('Training Step', fontsize=12)
    ax3.set_ylabel('Loss', fontsize=12)
    ax3.set_title('★ Teammate Prediction Loss ★', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. World Model Component Losses
    ax4 = plt.subplot(3, 3, 4)
    wm_loss_keys = ['shared_wm_recon_loss', 'shared_wm_dynamics_loss', 
                    'shared_wm_reward_loss', 'shared_wm_cont_loss']
    wm_loss_labels = ['Reconstruction', 'Dynamics', 'Reward', 'Continuation']
    for key, label in zip(wm_loss_keys, wm_loss_labels):
        if key in training_metrics and len(training_metrics[key]) > 0:
            ax4.plot(training_metrics[key], label=label, alpha=0.7, linewidth=1.5)
    ax4.set_xlabel('Training Step', fontsize=12)
    ax4.set_ylabel('Loss', fontsize=12)
    ax4.set_title('World Model Components', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Actor Loss (Per Agent)
    ax5 = plt.subplot(3, 3, 5)
    agent_names = list(episode_rewards.keys())
    for name in agent_names:
        key = f'{name}_actor_loss'
        if key in training_metrics and len(training_metrics[key]) > 0:
            ax5.plot(training_metrics[key], label=name, alpha=0.7, linewidth=1.5)
    ax5.set_xlabel('Training Step', fontsize=12)
    ax5.set_ylabel('Loss', fontsize=12)
    ax5.set_title('Actor Loss (Policy)', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Critic Loss (Per Agent)
    ax6 = plt.subplot(3, 3, 6)
    for name in agent_names:
        key = f'{name}_critic_loss'
        if key in training_metrics and len(training_metrics[key]) > 0:
            ax6.plot(training_metrics[key], label=name, alpha=0.7, linewidth=1.5)
    ax6.set_xlabel('Training Step', fontsize=12)
    ax6.set_ylabel('Loss', fontsize=12)
    ax6.set_title('Critic Loss (Value)', fontsize=14, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Mean Imagined Reward
    ax7 = plt.subplot(3, 3, 7)
    for name in agent_names:
        key = f'{name}_mean_imagined_reward'
        if key in training_metrics and len(training_metrics[key]) > 0:
            ax7.plot(training_metrics[key], label=name, alpha=0.7, linewidth=1.5)
    ax7.set_xlabel('Training Step', fontsize=12)
    ax7.set_ylabel('Reward', fontsize=12)
    ax7.set_title('Mean Imagined Reward (Rollout)', fontsize=14, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Mean Value Estimate
    ax8 = plt.subplot(3, 3, 8)
    for name in agent_names:
        key = f'{name}_mean_value'
        if key in training_metrics and len(training_metrics[key]) > 0:
            ax8.plot(training_metrics[key], label=name, alpha=0.7, linewidth=1.5)
    ax8.set_xlabel('Training Step', fontsize=12)
    ax8.set_ylabel('Value', fontsize=12)
    ax8.set_title('Mean Value Estimate', fontsize=14, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Cumulative Rewards
    ax9 = plt.subplot(3, 3, 9)
    for name, rewards in episode_rewards.items():
        if len(rewards) > 0:
            cumulative = np.cumsum(rewards)
            ax9.plot(cumulative, label=name, linewidth=2)
    ax9.set_xlabel('Episode', fontsize=12)
    ax9.set_ylabel('Cumulative Reward', fontsize=12)
    ax9.set_title('Cumulative Rewards', fontsize=14, fontweight='bold')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training curves saved to {save_path}")
    plt.show()


# ============================================================================
# 4. Model Architecture Inspection
# ============================================================================

def _count_layers(model, layer_types=(nn.Linear, nn.Conv2d)):
    """
    Count number of layers in a model.
    
    INTERNAL HELPER: 内部ヘルパー関数
    
    Args:
        model: PyTorch model
        layer_types: Tuple of layer types to count
    
    Returns:
        int: Number of layers
    """
    count = 0
    for m in model.modules():
        if isinstance(m, layer_types):
            count += 1
    return count


def _count_parameters(model, trainable_only=True):
    """
    Count total parameters in a model.
    
    INTERNAL HELPER: 内部ヘルパー関数
    
    Args:
        model: PyTorch model
        trainable_only: If True, only count trainable parameters
    
    Returns:
        int: Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def inspect_matwm_architecture(world_model, actor, critic, config, device):
    """
    Comprehensive architecture inspection for MATWM.
    
    PUBLIC API: 外部から呼び出される主要関数
    
    Args:
        world_model: WorldModel instance
        actor: Actor instance
        critic: Critic instance
        config: MATWMConfig
        device: torch device
    """
    print("\n" + "="*70)
    print("MATWM ARCHITECTURE INSPECTION")
    print("="*70)
    
    # 1. Dummy input test
    print("\n[1] Dummy Input Test")
    print("-" * 70)
    
    # Test World Model (use max_obs_dim for unified observation size)
    dummy_obs = torch.randn(1, config.max_obs_dim).to(device)
    print(f"Input observation shape: {dummy_obs.shape} (max_obs_dim={config.max_obs_dim})")
    
    try:
        z, z_logits = world_model.encode(dummy_obs)
        print(f"✓ Encoder output (latent): {z.shape}")
        print(f"✓ Encoder output (logits): {z_logits.shape}")
        
        obs_recon = world_model.decode(z)
        print(f"✓ Decoder output (reconstruction): {obs_recon.shape}")
        
        dummy_action = torch.randint(0, config.action_dim, (1, 1)).to(device)
        z_next, z_next_logits = world_model.predict_next(z.unsqueeze(1), dummy_action)
        print(f"✓ Dynamics output (next latent): {z_next.shape}")
        
        reward_logits = world_model.predict_reward(z_next)
        print(f"✓ Reward predictor output: {reward_logits.shape}")
        
        cont_logits = world_model.predict_continuation(z_next)
        print(f"✓ Continuation predictor output: {cont_logits.shape}")
        
        teammate_logits = world_model.predict_teammates(z.unsqueeze(1), 0)
        print(f"✓ Teammate predictor output: {len(teammate_logits)} agents")
        for agent_idx, logits in teammate_logits.items():
            print(f"  - Agent {agent_idx}: {logits.shape}")
    
    except Exception as e:
        print(f"✗ Error during forward pass: {e}")
    
    # Test Actor
    try:
        action_logits = actor(z)
        print(f"✓ Actor output (action logits): {action_logits.shape}")
    except Exception as e:
        print(f"✗ Actor error: {e}")
    
    # Test Critic
    try:
        value = critic(z)
        print(f"✓ Critic output (value): {value.shape}")
    except Exception as e:
        print(f"✗ Critic error: {e}")
    
    # 2. Layer count
    print(f"\n[2] Layer Count")
    print("-" * 70)
    wm_layers = _count_layers(world_model)
    actor_layers = _count_layers(actor)
    critic_layers = _count_layers(critic)
    print(f"World Model layers (Linear + Conv): {wm_layers}")
    print(f"Actor layers (Linear + Conv): {actor_layers}")
    print(f"Critic layers (Linear + Conv): {critic_layers}")
    print(f"Total layers: {wm_layers + actor_layers + critic_layers}")
    
    # 3. Parameter count
    print(f"\n[3] Parameter Count")
    print("-" * 70)
    wm_params = _count_parameters(world_model)
    actor_params = _count_parameters(actor)
    critic_params = _count_parameters(critic)
    total_params = wm_params + actor_params + critic_params
    
    print(f"World Model parameters: {wm_params:,} ({wm_params/1e6:.2f}M)")
    print(f"Actor parameters: {actor_params:,} ({actor_params/1e6:.2f}M)")
    print(f"Critic parameters: {critic_params:,} ({critic_params/1e6:.2f}M)")
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Breakdown by component
    print(f"\n[3.1] World Model Component Breakdown")
    print("-" * 70)
    encoder_params = _count_parameters(world_model.encoder)
    decoder_params = _count_parameters(world_model.decoder)
    dynamics_params = _count_parameters(world_model.dynamics)
    reward_params = _count_parameters(world_model.reward_predictor)
    cont_params = _count_parameters(world_model.continuation_predictor)
    teammate_params = _count_parameters(world_model.teammate_predictor)
    
    print(f"  Encoder: {encoder_params:,} ({encoder_params/1e6:.3f}M)")
    print(f"  Decoder: {decoder_params:,} ({decoder_params/1e6:.3f}M)")
    print(f"  Dynamics (Transformer): {dynamics_params:,} ({dynamics_params/1e6:.3f}M)")
    print(f"  Reward Predictor: {reward_params:,} ({reward_params/1e6:.3f}M)")
    print(f"  Continuation Predictor: {cont_params:,} ({cont_params/1e6:.3f}M)")
    print(f"  Teammate Predictor ★: {teammate_params:,} ({teammate_params/1e6:.3f}M)")
    
    # 4. Detailed summary with torchinfo (if available)
    print(f"\n[4] Detailed Model Summary (torchinfo)")
    print("-" * 70)
    try:
        from torchinfo import summary
        print("✓ torchinfo imported successfully")
        
        print("\n--- World Model Encoder ---")
        try:
            result = summary(world_model.encoder, input_size=(1, config.max_obs_dim), 
                    device=device, depth=3, col_names=["input_size", "output_size", "num_params"])
            print(result)
        except Exception as e:
            print(f"⚠ Encoder summary failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n--- World Model Dynamics (Transformer) ---")
        try:
            # Prepare dummy inputs for Dynamics
            dummy_z = torch.randn(1, 1, config.latent_dim, config.num_classes).to(device)
            dummy_action = torch.randint(0, config.action_dim, (1, 1)).to(device)
            result = summary(world_model.dynamics, 
                    input_data=[dummy_z, dummy_action],
                    device=device, depth=3, col_names=["input_size", "output_size", "num_params"])
            print(result)
        except Exception as e:
            print(f"⚠ Dynamics summary failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n--- Actor Network ---")
        try:
            result = summary(actor, input_size=(1, config.latent_dim, config.num_classes),
                    device=device, depth=3, col_names=["input_size", "output_size", "num_params"])
            print(result)
        except Exception as e:
            print(f"⚠ Actor summary failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n--- Critic Network ---")
        try:
            result = summary(critic, input_size=(1, config.latent_dim, config.num_classes),
                    device=device, depth=3, col_names=["input_size", "output_size", "num_params"])
            print(result)
        except Exception as e:
            print(f"⚠ Critic summary failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        
    except ImportError as e:
        print(f"⚠ torchinfo not installed: {e}")
        print("   Install with: pip install torchinfo")
        print("   Skipping detailed model summary.")
    except Exception as e:
        print(f"⚠ Unexpected error with torchinfo: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("INSPECTION COMPLETE")
    print("="*70 + "\n")


# ============================================================================
# 5. GPU Environment Information
# ============================================================================

def _get_gpu_info():
    """
    Get comprehensive GPU environment information.
    
    INTERNAL HELPER: 内部ヘルパー関数
    
    Returns:
        dict: GPU information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'gpu_count': 0,
        'gpus': [],
        'current_device': None,
        'current_gpu_name': None
    }
    
    if not torch.cuda.is_available():
        return info
    
    # GPU count
    info['gpu_count'] = torch.cuda.device_count()
    
    # Current device
    info['current_device'] = torch.cuda.current_device()
    info['current_gpu_name'] = torch.cuda.get_device_name(info['current_device'])
    
    # All available GPUs
    for i in range(info['gpu_count']):
        gpu_info = {
            'id': i,
            'name': torch.cuda.get_device_name(i),
            'capability': torch.cuda.get_device_capability(i),
            'total_memory_gb': torch.cuda.get_device_properties(i).total_memory / 1e9,
        }
        info['gpus'].append(gpu_info)
    
    return info


def _identify_gpu_type(gpu_name):
    """
    Identify common GPU types from name.
    
    INTERNAL HELPER: 内部ヘルパー関数
    
    Args:
        gpu_name: GPU name string
    
    Returns:
        str: Identified GPU type
    """
    gpu_name_lower = gpu_name.lower()
    
    # Common GPU types
    if 'a100' in gpu_name_lower:
        return "NVIDIA A100 (High-end Datacenter GPU)"
    elif 'a10' in gpu_name_lower:
        return "NVIDIA A10 (Datacenter GPU)"
    elif 'v100' in gpu_name_lower:
        return "NVIDIA V100 (Datacenter GPU)"
    elif 't4' in gpu_name_lower:
        return "NVIDIA T4 (Inference/Training GPU)"
    elif 'l4' in gpu_name_lower:
        return "NVIDIA L4 (AI Inference GPU)"
    elif 'rtx 4090' in gpu_name_lower:
        return "NVIDIA RTX 4090 (Consumer High-end)"
    elif 'rtx 4080' in gpu_name_lower:
        return "NVIDIA RTX 4080 (Consumer High-end)"
    elif 'rtx 3090' in gpu_name_lower:
        return "NVIDIA RTX 3090 (Consumer High-end)"
    elif 'rtx 3080' in gpu_name_lower:
        return "NVIDIA RTX 3080 (Consumer High-end)"
    elif 'rtx 3070' in gpu_name_lower:
        return "NVIDIA RTX 3070 (Consumer Mid-range)"
    elif 'rtx 3060' in gpu_name_lower:
        return "NVIDIA RTX 3060 (Consumer Mid-range)"
    elif 'rtx' in gpu_name_lower:
        return f"NVIDIA RTX Series ({gpu_name})"
    elif 'gtx' in gpu_name_lower:
        return f"NVIDIA GTX Series ({gpu_name})"
    elif 'tesla' in gpu_name_lower:
        return f"NVIDIA Tesla Series ({gpu_name})"
    else:
        return gpu_name


def print_gpu_info():
    """
    Print comprehensive GPU environment information.
    
    PUBLIC API: 外部から呼び出される主要関数
    
    Returns:
        dict: GPU information
    """
    print("\n" + "="*70)
    print("GPU ENVIRONMENT INFORMATION")
    print("="*70)
    
    info = _get_gpu_info()
    
    print(f"\nCUDA Available: {info['cuda_available']}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Version: {torch.version.cuda if info['cuda_available'] else 'N/A'}")
    
    if info['cuda_available']:
        print(f"\n[Current Device]")
        print(f"  Device ID: {info['current_device']}")
        print(f"  GPU Name: {info['current_gpu_name']}")
        
        print(f"\n[Available GPUs: {info['gpu_count']}]")
        print("-" * 70)
        for gpu in info['gpus']:
            print(f"  GPU {gpu['id']}: {gpu['name']}")
            print(f"    - Compute Capability: {gpu['capability'][0]}.{gpu['capability'][1]}")
            print(f"    - Total Memory: {gpu['total_memory_gb']:.2f} GB")
            
            # Try to get current memory usage
            try:
                allocated = torch.cuda.memory_allocated(gpu['id']) / 1e9
                reserved = torch.cuda.memory_reserved(gpu['id']) / 1e9
                print(f"    - Memory Allocated: {allocated:.2f} GB")
                print(f"    - Memory Reserved: {reserved:.2f} GB")
            except:
                pass
            print()
        
        # GPU type identification
        print("[GPU Type Identification]")
        print("-" * 70)
        for gpu in info['gpus']:
            gpu_type = _identify_gpu_type(gpu['name'])
            print(f"  GPU {gpu['id']}: {gpu_type}")
    
    else:
        print("\n⚠ No GPU available. Training will run on CPU (very slow).")
    
    print("\n" + "="*70 + "\n")
    
    return info


# ============================================================================
# Convenience function
# ============================================================================

def setup_matwm_training(config, device):
    """
    Complete setup for MATWM training with all utilities.
    
    PUBLIC API: 外部から呼び出される主要関数
    
    Args:
        config: MATWMConfig
        device: torch device
    
    Returns:
        dict: Setup information
    """
    print("\n" + "="*70)
    print("MATWM TRAINING SETUP")
    print("="*70)
    
    # GPU info
    gpu_info = print_gpu_info()
    
    # Create models (will be done in notebook)
    print("✓ GPU environment checked")
    print("✓ Ready for model initialization")
    
    print("\nNext steps:")
    print("1. Create shared world model with: MATWMAgent.create_shared_world_model()")
    print("2. Initialize weights with: initialize_matwm_weights()")
    print("3. Inspect architecture with: inspect_matwm_architecture()")
    print("4. Start training!")
    
    print("\n" + "="*70 + "\n")
    
    return gpu_info


if __name__ == "__main__":
    # Test GPU info
    print_gpu_info()
