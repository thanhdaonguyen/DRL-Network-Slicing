# agents.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from typing import Dict, List, Tuple, Optional
from utils import Configuration

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for agent communication"""
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and split into heads
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.embed_dim
        )
        
        output = self.out_linear(context)
        return output, attention_weights

class ActorNetwork(nn.Module):
    """Actor network for individual UAV agents"""
    def __init__(self, obs_dim, action_dim, hidden_dims=[256, 256, 128]):
        super().__init__()
        
        layers = []
        input_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Separate heads for different action types
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # delta_x, delta_y, delta_h
            nn.Tanh()  # Normalized position changes
        )
        
        self.power_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # Power adjustment
            nn.Sigmoid()  # Normalized power level
        )
        
        self.bandwidth_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Linear(64, action_dim - 4),  # Bandwidth allocation for DAs
            nn.Softmax(dim=-1)  # Ensure valid probability distribution
        )
        
    def forward(self, obs):
        features = self.shared_layers(obs)
        
        position_actions = self.position_head(features)
        power_action = self.power_head(features)
        bandwidth_actions = self.bandwidth_head(features)
        
        # Concatenate all actions
        actions = torch.cat([position_actions, power_action, bandwidth_actions], dim=-1)
        return actions

class CriticNetwork(nn.Module):
    """Centralized critic network with attention mechanism"""
    def __init__(self, state_dim, action_dim, num_agents, hidden_dims=[512, 256, 128]):
        super().__init__()
        
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Embedding layers for each agent's state-action pair
        self.agent_encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Attention mechanism for agent interactions
        self.attention = MultiHeadAttention(128, num_heads=4)
        
        # Global state encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(128 * num_agents, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Value network
        layers = []
        input_dim = hidden_dims[0]
        
        for hidden_dim in hidden_dims[1:]:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        self.value_network = nn.Sequential(*layers)

    def forward(self, states, actions):
        batch_size = states.size(0)
        
        # Reshape to per-agent format
        states_per_agent = states.view(batch_size, self.num_agents, -1)
        actions_per_agent = actions.view(batch_size, self.num_agents, -1)
        
        # Process each agent's state-action pair
        agent_features_list = []
        for i in range(self.num_agents):
            agent_input = torch.cat([states_per_agent[:, i], actions_per_agent[:, i]], dim=-1)
            agent_feature = self.agent_encoder(agent_input)
            agent_features_list.append(agent_feature)
        
        # Stack agent features
        agent_features = torch.stack(agent_features_list, dim=1)  # (batch_size, num_agents, feature_dim)
        
        # Apply self-attention among agents
        attended_features, _ = self.attention(
            agent_features, agent_features, agent_features
        )
        
        # Aggregate features
        global_features = attended_features.view(batch_size, -1)
        global_encoding = self.global_encoder(global_features)
        
        # Compute Q-value
        q_value = self.value_network(global_encoding)
        return q_value

class ExperienceBuffer:
    """Experience replay buffer for MADRL"""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, experience):
        """Add experience tuple to buffer"""
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        """Sample batch of experiences"""
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor(np.array([e[0] for e in experiences]))
        actions = torch.FloatTensor(np.array([e[1] for e in experiences]))
        rewards = torch.FloatTensor(np.array([e[2] for e in experiences]))
        next_states = torch.FloatTensor(np.array([e[3] for e in experiences]))
        dones = torch.FloatTensor(np.array([e[4] for e in experiences]))
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class MADRLAgent:
    """Multi-Agent Deep Reinforcement Learning Agent for UAV Network Slicing"""
    def __init__(self, 
                 num_agents: int,
                 obs_dim: int,
                 action_dim: int,
                 lr_actor: float = 3e-4,
                 lr_critic: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 buffer_size: int = 100000,
                 batch_size: int = 256,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 config_path = "./config/agents/default.yaml"):
        
        
        self.config = Configuration(config_path)
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device(device)
        
        # Initialize actor networks for each agent
        self.actors = []
        self.actor_targets = []
        self.actor_optimizers = []
        
        for i in range(num_agents):
            actor = ActorNetwork(obs_dim, action_dim).to(self.device)
            actor_target = ActorNetwork(obs_dim, action_dim).to(self.device)
            actor_target.load_state_dict(actor.state_dict())
            
            self.actors.append(actor)
            self.actor_targets.append(actor_target)
            self.actor_optimizers.append(optim.Adam(actor.parameters(), lr=lr_actor))
        
        # Initialize centralized critic
        self.critic = CriticNetwork(obs_dim, action_dim, num_agents).to(self.device)
        self.critic_target = CriticNetwork(obs_dim, action_dim, num_agents).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Experience buffer
        self.buffer = ExperienceBuffer(buffer_size)
        
        # Exploration parameters
        self.exploration_noise = 0.1
        self.noise_decay = 0.995
        self.min_noise = 0.01
        
    def select_actions(self, observations: Dict[int, np.ndarray], 
                      explore: bool = True) -> Dict[int, np.ndarray]:
        """Select actions for all agents based on observations"""
        actions = {}
        
        for agent_id, obs in observations.items():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action = self.actors[agent_id](obs_tensor).cpu().numpy()[0]
            
            # Add exploration noise
            if explore:
                noise = np.random.normal(0, self.exploration_noise, action.shape)
                action = action + noise
                # Clip actions to valid range
                action[:3] = np.clip(action[:3], -1, 1) # Position changes
                action[3] = np.clip(action[3], 0, 1)  # Power adjustment
                
                # Bandwidth actions (5+): Apply softmax for proper distribution
                if len(action) > 4:
                    bandwidth_part = action[4:]
                    # Simple softmax re-normalization after noise
                    bandwidth_part = np.exp(bandwidth_part - np.max(bandwidth_part))
                    bandwidth_part = bandwidth_part / np.sum(bandwidth_part)
                    action[4:] = bandwidth_part
            
            actions[agent_id] = action

        return actions
    
    def store_transition(self, observations: Dict[int, np.ndarray],
                        actions: Dict[int, np.ndarray],
                        reward: float,
                        next_observations: Dict[int, np.ndarray],
                        done: bool):
        """Store transition in replay buffer"""
        # Convert dictionaries to arrays
        obs_array = np.concatenate([observations[i] for i in range(self.num_agents)])
        action_array = np.concatenate([actions[i] for i in range(self.num_agents)])
        next_obs_array = np.concatenate([next_observations[i] for i in range(self.num_agents)])
        
        self.buffer.push((obs_array, action_array, reward, next_obs_array, done))
    
    def train(self):
        """Train the MADRL agent"""
        if len(self.buffer) < self.batch_size:
            return {}
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device).unsqueeze(-1)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device).unsqueeze(-1)
        
        # Reshape for individual agents
        states_per_agent = states.view(self.batch_size, self.num_agents, self.obs_dim)
        actions_per_agent = actions.view(self.batch_size, self.num_agents, self.action_dim)
        next_states_per_agent = next_states.view(self.batch_size, self.num_agents, self.obs_dim)
        
        # Update critic
        with torch.no_grad():
            # Get target actions from target actors
            target_actions = []
            for i in range(self.num_agents):
                target_action = self.actor_targets[i](next_states_per_agent[:, i, :])
                target_actions.append(target_action)
            
            target_actions = torch.stack(target_actions, dim=1)
            target_actions = target_actions.view(self.batch_size, -1)
            
            # Compute target Q-value
            target_q = self.critic_target(next_states_per_agent.view(self.batch_size, -1), target_actions)
            target_value = rewards + (1 - dones) * self.gamma * target_q
        
        # Current Q-value
        current_q = self.critic(states, actions)
        
        # Critic loss
        critic_loss = F.mse_loss(current_q, target_value)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()   
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Update actors
        actor_losses = []
        for i in range(self.num_agents):
            # Get current actions from all agents
            current_actions = []
            for j in range(self.num_agents):
                if j == i:
                    # For agent i, use its actor network
                    action = self.actors[j](states_per_agent[:, j, :])
                else:
                    # For other agents, use their current actions (detached)
                    action = actions_per_agent[:, j, :].detach()
                current_actions.append(action)
            
            current_actions = torch.stack(current_actions, dim=1)
            current_actions = current_actions.view(self.batch_size, -1)
            
            # Actor loss (negative Q-value for gradient descent)
            actor_loss = -self.critic(states, current_actions).mean()
            
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 1.0)
            self.actor_optimizers[i].step()
            
            actor_losses.append(actor_loss.item())
        
        # Soft update target networks
        self._soft_update()
        
        # Decay exploration noise
        self.exploration_noise = max(self.min_noise, self.exploration_noise * self.noise_decay)
        # print("Actor Loss:", actor_losses, "Critic Loss:", critic_loss)
        return {
            'actor_losses': actor_losses,
            'critic_loss': critic_loss.item()
        }
        # return {
        #     'critic_loss': critic_loss.item(),
        #     'actor_losses': actor_losses,
        #     'exploration_noise': self.exploration_noise
        # }
    
    def _soft_update(self):
        """Soft update target networks"""
        # Update critic target
        for param, target_param in zip(self.critic.parameters(), 
                                      self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + 
                                  (1 - self.tau) * target_param.data)
        
        # Update actor targets
        for i in range(self.num_agents):
            for param, target_param in zip(self.actors[i].parameters(), 
                                          self.actor_targets[i].parameters()):
                target_param.data.copy_(self.tau * param.data + 
                                      (1 - self.tau) * target_param.data)
    
    def save_models(self, path: str):
        """Save all models"""
        checkpoint = {
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'actors': [actor.state_dict() for actor in self.actors],
            'actor_targets': [actor.state_dict() for actor in self.actor_targets],
            'actor_optimizers': [opt.state_dict() for opt in self.actor_optimizers],
            'exploration_noise': self.exploration_noise
        }
        torch.save(checkpoint, path)
        print(f"Models saved to {path}")
    
    def load_models(self, path: str):
        """Load all models"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        for i in range(self.num_agents):
            self.actors[i].load_state_dict(checkpoint['actors'][i])
            self.actor_targets[i].load_state_dict(checkpoint['actor_targets'][i])
            self.actor_optimizers[i].load_state_dict(checkpoint['actor_optimizers'][i])
        
        self.exploration_noise = checkpoint['exploration_noise']
        print(f"Models loaded from {path}")