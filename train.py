# train.py
import numpy as np
import torch
import matplotlib.pyplot as plt
from environment import NetworkSlicingEnv
from agents import MADRLAgent
import json
import os
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import re
from utils import Configuration

class TrainingManager:
    """Manages the training process for UAV network slicing MADRL"""

    def __init__(self, env_config_path: str = None, train_config_path: str = None):
        # Load configurations using your Configuration class
        env_config = Configuration(env_config_path)
        train_config = Configuration(train_config_path)

        self.config = {
            # Environment parameters
            'num_uavs': env_config.system.num_uavs,
            'num_ues': env_config.system.num_ues,
            'service_area': tuple(env_config.system.service_area),
            'height_range': tuple(env_config.system.height_range),
            'num_das_per_slice': env_config.system.num_das_per_slice,

            # Step-based training parameters
            'total_training_steps': train_config.total_training_steps,
            'episode_length': train_config.episode_length,
            'batch_size': train_config.batch_size,
            'learning_rate_actor': train_config.learning_rate_actor,
            'learning_rate_critic': train_config.learning_rate_critic,
            'gamma': train_config.gamma,
            'tau': train_config.tau,
            'buffer_size': train_config.buffer_size,

            # Step-based intervals
            'log_interval': train_config.log_interval,
            'save_interval': train_config.save_interval,
            'evaluation_interval': train_config.evaluation_interval,
            'plot_interval': train_config.plot_interval,

            # Paths
            'save_dir': train_config.save_dir,
            'log_dir': train_config.log_dir,
            'tensorboard_dir': train_config.tensorboard_dir,
            
            # Config paths
            'env_config_path': env_config_path,
            'train_config_path': train_config_path
        }

        # Create directories
        os.makedirs(self.config['save_dir'], exist_ok=True)
        os.makedirs(self.config['log_dir'], exist_ok=True)

        # Create unique model directory
        self.model_dir = self._get_next_model_dir(self.config['save_dir'])
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, 'training_progress'), exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, 'checkpoints'), exist_ok=True)

        # Initialize tracking variables for step-based training
        self.step_metrics = {
            'steps': [],
            'rewards': [],
            'qos_satisfaction': [],
            'energy_efficiency': [],
            'interference_level': [],
            'active_ues': [],
            'actor_losses': [],
            'critic_losses': []
        }
        
        self.episode_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_numbers': [],
            'episode_steps': []  # Global step when episode ended
        }

        # Initialize environment and agent
        self.env = NetworkSlicingEnv(config_path=self.config['env_config_path'])
        
        # Get observation and action dimensions
        obs_sample = self.env.reset()
        obs_dim = len(list(obs_sample.values())[0])
        action_dim = 4 + self.config['num_das_per_slice'] * 3  # pos + power + bandwidth per DA
        
        self.agent = MADRLAgent(
            num_agents=self.config['num_uavs'],
            obs_dim=obs_dim,
            action_dim=action_dim,
            lr_actor=self.config['learning_rate_actor'],
            lr_critic=self.config['learning_rate_critic'],
            gamma=self.config['gamma'],
            tau=self.config['tau'],
            buffer_size=self.config['buffer_size'],
            batch_size=self.config['batch_size']
        )

    def _get_next_model_dir(self, save_dir):
        """Find the next available model directory name (model1, model2, ...)"""
        existing = [d for d in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, d))]
        numbers = []
        for name in existing:
            m = re.match(r'model(\d+)', name)
            if m:
                numbers.append(int(m.group(1)))
        next_num = 1
        while f"model{next_num}" in existing:
            next_num += 1
        return os.path.join(save_dir, f"model{next_num}")

    def _save_env_info(self):
        """Save environment configuration to the model directory"""
        # Get obs_dim and action_dim
        sample_obs = self.env.reset()
        obs_dim = sample_obs[0].shape[0]
        action_dim = 4 + self.config['num_das_per_slice'] * 3  # position(3) + power(1) + bandwidth allocation

        env_info = {
            'num_uavs': self.config['num_uavs'],
            'num_ues': self.config['num_ues'],
            'service_area': self.config['service_area'],
            'height_range': self.config['height_range'],
            'num_das_per_slice': self.config['num_das_per_slice'],
            'obs_dim': obs_dim,
            'action_dim': action_dim,
            'config': self.config
        }
        env_info_path = os.path.join(self.model_dir, 'env_info.json')
        with open(env_info_path, 'w') as f:
            json.dump(env_info, f, indent=2)

    def train(self):
        """Step-based training loop with comprehensive logging"""
        print(f"Starting step-based training for {self.config['total_training_steps']} steps")
        print(f"Episode length: {self.config['episode_length']} steps")
        
        # Initialize environment
        observations = self.env.reset()
        
        # Training state variables
        episode_count = 0
        current_episode_reward = 0
        current_episode_steps = 0
        best_eval_reward = float('-inf')
        
        # Main training loop
        progress_bar = tqdm(range(self.config['total_training_steps']), desc="Training Steps")
        
        for global_step in progress_bar:
            # Select actions
            actions = self.agent.select_actions(observations, explore=True)
            
            # Execute actions
            next_observations, reward, done, info = self.env.step(actions)
            
            # Store transition
            self.agent.store_transition(
                observations, actions, reward, next_observations, done
            )
            
            # Train agent if buffer has enough samples
            train_info = None
            if len(self.agent.buffer) >= self.config['batch_size']:
                train_info = self.agent.train()
            
            # Store step metrics
            self._store_step_metrics(global_step, reward, info, train_info)
            
            # Update state
            observations = next_observations
            current_episode_reward += reward
            current_episode_steps += 1
            
            # Handle episode termination
            if done or current_episode_steps >= self.config['episode_length']:
                self._handle_episode_completion(
                    episode_count, current_episode_reward, 
                    current_episode_steps, global_step
                )
                
                # Reset for new episode
                observations = self.env.reset()
                episode_count += 1
                current_episode_reward = 0
                current_episode_steps = 0
            
            # Periodic operations
            if global_step % self.config['log_interval'] == 0 and global_step > 0:
                self._log_step_progress(global_step, progress_bar)
            
            if global_step % self.config['evaluation_interval'] == 0 and global_step > 0:
                eval_reward = self._evaluate_agent(global_step)
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    self._save_best_model(global_step)
            
            if global_step % self.config['save_interval'] == 0 and global_step > 0:
                self._save_checkpoint(global_step)
            
            if global_step % self.config['plot_interval'] == 0 and global_step > 0:
                self._plot_training_progress(global_step)
            
            # Decay exploration noise
            if hasattr(self.agent, 'exploration_noise'):
                self.agent.exploration_noise = max(
                    self.agent.min_noise,
                    self.agent.exploration_noise * self.agent.noise_decay
                )
        
        # Final operations
        self._save_checkpoint(self.config['total_training_steps'])
        self._plot_final_results()
        print("Training completed!")

    def _store_step_metrics(self, step, reward, info, train_info):
        """Store metrics for this step"""
        self.step_metrics['steps'].append(step)
        self.step_metrics['rewards'].append(reward)
        self.step_metrics['qos_satisfaction'].append(info['qos_satisfaction'])
        self.step_metrics['energy_efficiency'].append(info['energy_efficiency'])
        self.step_metrics['interference_level'].append(info['interference_level'])
        self.step_metrics['active_ues'].append(info['active_ues'])
        
        # Store training metrics if available
        if train_info:
            # Store average actor loss (since train_info['actor_losses'] is [loss1, loss2, loss3])
            actor_losses = train_info.get('actor_losses', [])
            avg_actor_loss = sum(actor_losses) / len(actor_losses) if actor_losses else 0
            self.step_metrics['actor_losses'].append(avg_actor_loss)
            
            # Store critic loss
            self.step_metrics['critic_losses'].append(train_info.get('critic_loss', 0))
        else:
            self.step_metrics['actor_losses'].append(0)
            self.step_metrics['critic_losses'].append(0)

    def _handle_episode_completion(self, episode_num, episode_reward, episode_steps, global_step):
        """Handle episode completion"""
        self.episode_metrics['episode_rewards'].append(episode_reward)
        self.episode_metrics['episode_lengths'].append(episode_steps)
        self.episode_metrics['episode_numbers'].append(episode_num)
        self.episode_metrics['episode_steps'].append(global_step)
        
        print(f"\nEpisode {episode_num} completed at step {global_step}")
        print(f"Episode length: {episode_steps}, Reward: {episode_reward:.2f}")

    def _log_step_progress(self, step, progress_bar):
        """Log training progress"""
        # Get recent metrics (last 100 steps)
        recent_rewards = self.step_metrics['rewards'][-100:] if len(self.step_metrics['rewards']) >= 100 else self.step_metrics['rewards']
        recent_qos = self.step_metrics['qos_satisfaction'][-100:] if len(self.step_metrics['qos_satisfaction']) >= 100 else self.step_metrics['qos_satisfaction']
        
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0
        avg_qos = np.mean(recent_qos) if recent_qos else 0
        
        # Update progress bar
        progress_bar.set_postfix({
            'Reward': f'{avg_reward:.3f}',
            'QoS': f'{avg_qos:.3f}',
            'Noise': f'{getattr(self.agent, "exploration_noise", 0):.4f}'
        })

    def _evaluate_agent(self, step):
        """Evaluate agent performance without exploration"""
        print(f"\n--- Evaluation at step {step} ---")
        
        total_eval_reward = 0
        
        for eval_episode in range(self.config.get('evaluation', {}).get('eval_episodes', 3)):
            eval_obs = self.env.reset()
            eval_reward = 0
            eval_steps = 0
            
            for _ in range(self.config.get('evaluation', {}).get('eval_steps', 500)):
                eval_actions = self.agent.select_actions(eval_obs, explore=False)
                eval_obs, reward, done, info = self.env.step(eval_actions)
                eval_reward += reward
                eval_steps += 1
                
                if done:
                    break
            
            total_eval_reward += eval_reward
            print(f"Eval episode {eval_episode + 1}: {eval_steps} steps, Reward: {eval_reward:.2f}")
        
        avg_eval_reward = total_eval_reward / self.config.get('evaluation', {}).get('eval_episodes', 3)
        print(f"Average evaluation reward: {avg_eval_reward:.2f}")
        print("--- End Evaluation ---\n")
        
        return avg_eval_reward

    def _plot_training_progress(self, current_step):
        """Plot comprehensive training progress"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Training Progress - Step {current_step}', fontsize=16)
        
        steps = self.step_metrics['steps']
        
        # Plot 1: Reward progression
        axes[0, 0].plot(steps, self.step_metrics['rewards'], alpha=0.3, color='blue')
        if len(steps) > 100:
            smoothed_rewards = self._moving_average(self.step_metrics['rewards'], 1000)
            axes[0, 0].plot(steps[-len(smoothed_rewards):], smoothed_rewards, 'b-', linewidth=2)
        axes[0, 0].set_title('Training Reward')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # Plot 2: QoS satisfaction
        axes[0, 1].plot(steps, self.step_metrics['qos_satisfaction'], 'g-', alpha=0.6)
        if len(steps) > 100:
            smoothed_qos = self._moving_average(self.step_metrics['qos_satisfaction'], 1000)
            axes[0, 1].plot(steps[-len(smoothed_qos):], smoothed_qos, 'darkgreen', linewidth=2)
        axes[0, 1].set_title('QoS Satisfaction')
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('QoS Satisfaction')
        axes[0, 1].grid(True)
        
        # Plot 3: Energy efficiency
        axes[0, 2].plot(steps, self.step_metrics['energy_efficiency'], 'orange', alpha=0.6)
        if len(steps) > 100:
            smoothed_energy = self._moving_average(self.step_metrics['energy_efficiency'], 1000)
            axes[0, 2].plot(steps[-len(smoothed_energy):], smoothed_energy, 'darkorange', linewidth=2)
        axes[0, 2].set_title('Energy Efficiency')
        axes[0, 2].set_xlabel('Steps')
        axes[0, 2].set_ylabel('Energy Consumption Rate')
        axes[0, 2].grid(True)
        
        # Plot 4: Active UEs
        axes[1, 0].plot(steps, self.step_metrics['active_ues'], 'purple', alpha=0.6)
        if len(steps) > 100:
            smoothed_ues = self._moving_average(self.step_metrics['active_ues'], 1000)
            axes[1, 0].plot(steps[-len(smoothed_ues):], smoothed_ues, 'darkviolet', linewidth=2)
        axes[1, 0].set_title('Active UEs')
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Number of UEs')
        axes[1, 0].grid(True)
        
        # Plot 5: Training losses
        if any(loss > 0 for loss in self.step_metrics['actor_losses']):
            axes[1, 1].plot(steps, self.step_metrics['actor_losses'], 'red', alpha=0.6, label='Actor Loss')
            axes[1, 1].plot(steps, self.step_metrics['critic_losses'], 'blue', alpha=0.6, label='Critic Loss')
            axes[1, 1].set_title('Training Losses')
            axes[1, 1].set_xlabel('Steps')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        # Plot 6: Episode rewards
        if self.episode_metrics['episode_rewards']:
            axes[1, 2].plot(self.episode_metrics['episode_steps'], 
                        self.episode_metrics['episode_rewards'], 'mo-', alpha=0.7)
            axes[1, 2].set_title('Episode Rewards')
            axes[1, 2].set_xlabel('Training Step')
            axes[1, 2].set_ylabel('Episode Reward')
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.model_dir}/training_progress/training_progress_step_{current_step}.png', dpi=150, bbox_inches='tight')
        plt.close()  # Close to save memory

    def _moving_average(self, data, window_size):
        """Calculate moving average"""
        if len(data) < window_size:
            return data
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    def _save_checkpoint(self, step):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.model_dir, 'checkpoints', f'checkpoint_step_{step}.pth')
        # Assuming your agent has a save method
        if hasattr(self.agent, 'save_models'):
            self.agent.save_models(checkpoint_path)
        print(f"Checkpoint saved at step {step}")

    def _save_best_model(self, step):
        """Save best performing model"""
        best_model_path = os.path.join(self.model_dir, 'best_model.pth')
        if hasattr(self.agent, 'save_models'):
            self.agent.save_models(best_model_path)
        print(f"Best model saved at step {step}")

    def _plot_final_results(self):
        """Plot comprehensive final results"""
        import matplotlib.pyplot as plt
        
        # Create final comprehensive plot
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Final Training Results', fontsize=16)
        
        steps = self.step_metrics['steps']
        
        # All metrics with smoothing
        metrics_config = [
            ('rewards', 'Reward', 'blue', axes[0, 0]),
            ('qos_satisfaction', 'QoS Satisfaction', 'green', axes[0, 1]),
            ('energy_efficiency', 'Energy Efficiency', 'orange', axes[1, 0]),
            ('interference_level', 'Interference Level', 'red', axes[1, 1]),
            ('active_ues', 'Active UEs', 'purple', axes[2, 0])
        ]
        
        for metric_name, title, color, ax in metrics_config:
            data = self.step_metrics[metric_name]
            ax.plot(steps, data, alpha=0.3, color=color)
            if len(data) > 1000:
                smoothed = self._moving_average(data, 1000)
                ax.plot(steps[-len(smoothed):], smoothed, color=color, linewidth=2)
            ax.set_title(title)
            ax.set_xlabel('Training Steps')
            ax.grid(True)
        
        # Episode summary
        if self.episode_metrics['episode_rewards']:
            axes[2, 1].plot(self.episode_metrics['episode_steps'], 
                        self.episode_metrics['episode_rewards'], 'mo-', alpha=0.7)
            axes[2, 1].set_title('Episode Rewards')
            axes[2, 1].set_xlabel('Training Step')
            axes[2, 1].set_ylabel('Episode Reward')
            axes[2, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.model_dir}/final_training_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Save metrics to CSV for further analysis
        self._save_metrics_to_csv()

    def _save_metrics_to_csv(self):
        """Save all metrics to CSV files"""
        import pandas as pd
        
        # Step-based metrics
        step_df = pd.DataFrame(self.step_metrics)
        step_df.to_csv(f'{self.model_dir}/step_metrics.csv', index=False)
        
        # Episode-based metrics
        if self.episode_metrics['episode_rewards']:
            episode_df = pd.DataFrame(self.episode_metrics)
            episode_df.to_csv(f'{self.model_dir}/episode_metrics.csv', index=False)
        
        print(f"Metrics saved to {self.model_dir}")
def main():
    """Main function to run training"""
    # You can specify a custom config file
    env_config_file = "config/environment/default.yaml"  # or 'config.json' if you have one
    train_config_file = 'config/train/default.yaml'
    # Create training manager
    trainer = TrainingManager(env_config_path=env_config_file, 
                              train_config_path=train_config_file)

    # Run training
    trainer.train()

    # Plot results
    trainer.plot_training_results()

if __name__ == "__main__":
    main()