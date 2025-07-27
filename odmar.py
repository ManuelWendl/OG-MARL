import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
from collections import deque
import random
import copy
from env import MultiAgentEnvironment
import os
import sys

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Constants
MAX_TRAINING_STEPS = 10000
MAX_STEPS = 100
EVAL_INTERVAL = 200
N_AGENTS = 5
ACTION_PENALTY = False  
AGENT_STATE_DIM = 2  
ACTION_DIM = 2  
GLOBAL_STATE_DIM = N_AGENTS * AGENT_STATE_DIM  
CONSENSUS_ROUNDS = 100

# Hyperparameters
BUFFER_SIZE = 1000000
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005  
ALPHA_LR = 3e-4
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
MODEL_BATCH_SIZE = 256
MODEL_TRAIN_FREQ = 250
USE_DISTRIBUTED_MODEL = True  
UPDATES_PER_STEP = 10  
REAL_RATIO = 0.0  
IMAGINARY_ROLLOUT_LENGTH = 2  
OPTIMISM_SCALE = 0.0 
SEED = 0

# Set seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

class ReplayBuffer:
    """Experience replay buffer to store and sample transitions."""
    
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(states).to(device),
            torch.FloatTensor(actions).to(device),
            torch.FloatTensor(rewards).unsqueeze(1).to(device),
            torch.FloatTensor(next_states).to(device),
            torch.FloatTensor(dones).unsqueeze(1).to(device)
        )
    
    def __len__(self):
        return len(self.buffer)

class DistributedGP:
    """Wrapper for distributed GP from distributed_gp.py"""
    
    def __init__(self, n_agents, state_dim, action_dim, graph_type="Cycle", kernel_type="rbf"):
        self.n_agents = n_agents
        self.state_dim = state_dim  # Now this is the agent's state dimension (2)
        self.action_dim = action_dim
        self.graph_type = graph_type
        self.kernel_type = kernel_type  # 'rbf', 'linear', or 'combined'
        
        # Kernel hyperparameters
        self.lengthscale = 2.0  # Hyperparameter for RBF kernel
        self.noise_var = 0.01  # Observation noise variance
        self.linear_coef = 1.0  # Coefficient for linear kernel
        self.linear_constant = 0.1  # Constant term in linear kernel
        
        # These will be initialized in initialize_model
        self.Z = None  # Inducing points
        self.m_list = []  # Mean vectors for each agent
        self.S_list = []  # Covariance matrices for each agent
        self.J_list = []  # Natural precision parameters for each agent
        self.h_list = []  # Natural mean parameters for each agent
        
        # Initialize weights based on graph type
        if graph_type == "Cycle":
            self.W = self._initialize_cycle_graph()
        else:
            raise ValueError(f"Unsupported graph type: {graph_type}")
        
        # For tracking model loss
        self.model_losses = []
    
    def _initialize_cycle_graph(self):
        """Initialize weights for a cycle graph."""
        W = np.zeros((self.n_agents, self.n_agents))
        for i in range(self.n_agents):
            W[i, i] = 0.5  # Self-weight
            W[i, (i-1) % self.n_agents] = 0.25  # Left neighbor
            W[i, (i+1) % self.n_agents] = 0.25  # Right neighbor
        return W
    
    def rbf_kernel(self, X1, X2):
        """RBF kernel function."""
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2*np.dot(X1, X2.T)
        return np.exp(-0.5/self.lengthscale**2 * sqdist)
    
    def linear_kernel(self, X1, X2):
        """Linear kernel function: k(x,y) = x^T y + c"""
        return self.linear_coef * np.dot(X1, X2.T) + self.linear_constant
    
    def combined_kernel(self, X1, X2):
        """Combined RBF and Linear kernel"""
        return self.rbf_kernel(X1, X2) + self.linear_kernel(X1, X2)
    
    def compute_kernel(self, X1, X2):
        """Compute kernel based on selected type"""
        if self.kernel_type == "rbf":
            return self.rbf_kernel(X1, X2)
        elif self.kernel_type == "linear":
            return self.linear_kernel(X1, X2)
        elif self.kernel_type == "combined":
            return self.combined_kernel(X1, X2)
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")
    
    def initialize_model(self, dataset_list, n_inducing=1000):
        """Initialize GP model for each agent using their local datasets."""
        # Combine all data to sample inducing points
        all_X = np.vstack([data[0] for data in dataset_list])
        
        # Sample inducing points
        idx = np.random.choice(all_X.shape[0], n_inducing, replace=False)
        self.Z = all_X[idx]
        
        # Compute Kzz (inducing points kernel matrix)
        self.Kzz = self.compute_kernel(self.Z, self.Z)
        
        # Initialize natural parameters for each agent
        for i, (X, Y) in enumerate(dataset_list):
            Kzx = self.compute_kernel(self.Z, X)
            Kxz = Kzx.T
            
            # Compute natural parameters
            Ji = (1/self.noise_var) * (Kzx.dot(Kxz))
            hi = (1/self.noise_var) * (Kzx.dot(Y))
            
            self.J_list.append(Ji)
            self.h_list.append(hi)
        
        # Perform consensus rounds to initialize posterior
        self._perform_consensus()
    
    def _perform_consensus(self):
        """Perform consensus rounds using the defined graph."""
        # Make copies of J and h for consensus updates
        J_cons = self.J_list.copy()
        h_cons = self.h_list.copy()
        
        # Perform consensus rounds
        for _ in range(CONSENSUS_ROUNDS):
            # Update J_cons
            J_next = []
            for i in range(self.n_agents):
                Ji_next = sum(self.W[i, j] * J_cons[j] for j in range(self.n_agents))
                J_next.append(Ji_next)
            J_cons = J_next
            
            # Update h_cons
            h_next = []
            for i in range(self.n_agents):
                hi_next = sum(self.W[i, j] * h_cons[j] for j in range(self.n_agents))
                h_next.append(hi_next)
            h_cons = h_next
        
        # Store consensus results
        self.J_cons = J_cons
        self.h_cons = h_cons
        
        # Compute posterior for each agent
        self.S_list = []
        self.m_list = []
        
        for i in range(self.n_agents):
            # Recover global parameters (all agents have same info after consensus)
            J_bar = J_cons[i]
            h_bar = h_cons[i]
            J_sum = self.n_agents * J_bar  # Scale by number of agents
            h_sum = self.n_agents * h_bar
            
            # Compute posterior
            S_i = np.linalg.inv(self.Kzz + J_sum)
            m_i = S_i.dot(h_sum)
            
            self.S_list.append(S_i)
            self.m_list.append(m_i)
    
    def update_model(self, dataset_list):
        """Update GP model with new data."""
        # Compute new natural parameters
        J_new = []
        h_new = []
        
        model_loss = 0.0
        
        for i, (X, Y) in enumerate(dataset_list):
            Kzx = self.compute_kernel(self.Z, X)
            Kxz = Kzx.T
            
            # Compute natural parameters
            Ji = (1/self.noise_var) * (Kzx.dot(Kxz))
            hi = (1/self.noise_var) * (Kzx.dot(Y))
            
            # For model loss calculation (MSE)
            Y_pred = self.predict(i, X)
            mse = np.mean(np.square(Y - Y_pred))
            model_loss += mse
            
            J_new.append(Ji)
            h_new.append(hi)
        # Update natural parameters
        self.J_list = J_new
        self.h_list = h_new
        
        # Perform consensus
        self._perform_consensus()
        
        # Record model loss
        avg_model_loss = model_loss / self.n_agents
        self.model_losses.append(avg_model_loss)
        
        return avg_model_loss
    
    def predict(self, agent_idx, X):
        """Make prediction for a given agent using its posterior."""
        K_test = self.compute_kernel(X, self.Z)
        return K_test.dot(self.m_list[agent_idx])

    def predict_with_variance(self, agent_idx, X):
        """Make prediction with variance for uncertainty."""
        K_test = self.compute_kernel(X, self.Z)
        mean = K_test.dot(self.m_list[agent_idx])
        var = np.diag(self.compute_kernel(X, X) - K_test.dot(self.S_list[agent_idx]).dot(K_test.T))
        return mean, var
    
class CentralizedGP:
    """Centralized GP model used by all agents."""
    
    def __init__(self, state_dim, action_dim, kernel_type="combined"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_dim = state_dim + action_dim
        self.output_dim = state_dim
        self.kernel_type = kernel_type
        
        # Kernel hyperparameters
        self.lengthscale = 2.0  # Hyperparameter for RBF kernel
        self.noise_var = 0.01  # Observation noise variance
        self.linear_coef = 0.5  # Coefficient for linear kernel
        self.linear_constant = 0.1  # Constant term in linear kernel
        
        # These will be initialized in initialize_model
        self.Z = None  # Inducing points
        self.m = None  # Mean vector
        self.S = None  # Covariance matrix
        self.Kzz = None  # Kernel matrix of inducing points
        self.Kzz_inv = None  # Inverse of Kzz
        
        # For tracking model loss
        self.model_losses = []
    
    def rbf_kernel(self, X1, X2):
        """RBF kernel function."""
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2*np.dot(X1, X2.T)
        return np.exp(-0.5/self.lengthscale**2 * sqdist)
    
    def linear_kernel(self, X1, X2):
        """Linear kernel function: k(x,y) = x^T y + c"""
        return self.linear_coef * np.dot(X1, X2.T) + self.linear_constant
    
    def combined_kernel(self, X1, X2):
        """Combined RBF and Linear kernel"""
        return self.rbf_kernel(X1, X2) + self.linear_kernel(X1, X2)
    
    def compute_kernel(self, X1, X2):
        """Compute kernel based on selected type"""
        if self.kernel_type == "rbf":
            return self.rbf_kernel(X1, X2)
        elif self.kernel_type == "linear":
            return self.linear_kernel(X1, X2)
        elif self.kernel_type == "combined":
            return self.combined_kernel(X1, X2)
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")
    
    def initialize_model(self, dataset_list, n_inducing=1000):
        """Initialize GP model using data from all agents."""
        # Combine all data
        all_X = np.vstack([X for X, _ in dataset_list])
        all_Y = np.vstack([Y for _, Y in dataset_list])
        
        # Sample inducing points
        if all_X.shape[0] > n_inducing:
            idx = np.random.choice(all_X.shape[0], n_inducing, replace=False)
            self.Z = all_X[idx]
        else:
            self.Z = all_X.copy()  # Use all data points if less than n_inducing
            
        # Compute Kzz (inducing points kernel matrix)
        self.Kzz = self.compute_kernel(self.Z, self.Z)
        self.Kzz_inv = np.linalg.inv(self.Kzz + 1e-6 * np.eye(self.Kzz.shape[0]))
        
        # Compute posterior
        Kzx = self.compute_kernel(self.Z, all_X)
        Kxz = Kzx.T
        
        # Compute posterior parameters
        A = Kzx.dot(Kxz) / self.noise_var + self.Kzz
        b = Kzx.dot(all_Y) / self.noise_var
        
        try:
            self.S = np.linalg.inv(A)
            self.m = self.S.dot(b)
        except np.linalg.LinAlgError:
            # Fall back to more stable but slower approach
            self.S = np.linalg.inv(A + 1e-6 * np.eye(A.shape[0]))
            self.m = self.S.dot(b)
        
        # Calculate initial loss
        Y_pred = self.predict(all_X)
        mse = np.mean(np.square(all_Y - Y_pred))
        self.model_losses.append(mse)
        
        print(f"Model initialized with {len(self.Z)} inducing points, initial MSE: {mse:.4f}")
        return mse
    
    def predict(self, X):
        """Make prediction using posterior."""
        K_test = self.compute_kernel(X, self.Z)
        return K_test.dot(self.m)
    
    def predict_with_variance(self, X):
        """Make prediction with variance for uncertainty estimation."""
        K_test = self.compute_kernel(X, self.Z)
        mean = K_test.dot(self.m)
        
        # Calculate variance
        var_diag = np.diag(self.compute_kernel(X, X)) - np.sum(K_test.dot(self.S) * K_test, axis=1)
        
        # Ensure positive variance
        var_diag = np.maximum(var_diag, 1e-6)
        
        return mean, var_diag
    
    def update_model(self, dataset_list):
        """Update GP model with new data."""
        # Combine all data
        all_X = np.vstack([X for X, _ in dataset_list])
        all_Y = np.vstack([Y for _, Y in dataset_list])
        
        # Recompute posterior with new data
        Kzx = self.compute_kernel(self.Z, all_X)
        Kxz = Kzx.T
        
        # Compute posterior parameters
        A = Kzx.dot(Kxz) / self.noise_var + self.Kzz
        b = Kzx.dot(all_Y) / self.noise_var
        
        try:
            self.S = np.linalg.inv(A)
            self.m = self.S.dot(b)
        except np.linalg.LinAlgError:
            # Fall back to more stable approach
            self.S = np.linalg.inv(A + 1e-6 * np.eye(A.shape[0]))
            self.m = self.S.dot(b)
        
        # Calculate model loss
        Y_pred = self.predict(all_X)
        mse = np.mean(np.square(all_Y - Y_pred))
        self.model_losses.append(mse)
        
        return mse

class PolicyNetwork(nn.Module):
    """Actor network for SAC."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        
        # Squash to [-1, 1]
        action = torch.tanh(x_t)
        
        # Calculate log probability, adding correction for tanh squashing
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob

class QNetwork(nn.Module):
    """Critic network for SAC."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class SACAgent:
    """Soft Actor-Critic agent with model-based policy optimization."""
    
    def __init__(self, agent_idx, state_dim, action_dim):
        self.agent_idx = agent_idx
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Actor network
        self.actor = PolicyNetwork(state_dim, action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        
        # Critic networks (two for double Q-learning)
        self.critic1 = QNetwork(state_dim, action_dim).to(device)
        self.critic2 = QNetwork(state_dim, action_dim).to(device)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=CRITIC_LR)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=CRITIC_LR)
        
        # Target networks
        self.critic1_target = QNetwork(state_dim, action_dim).to(device)
        self.critic2_target = QNetwork(state_dim, action_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Automatic entropy tuning
        self.target_entropy = -action_dim  # Heuristic
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=ALPHA_LR)
        self.alpha = self.log_alpha.exp()
        
        # Loss tracking
        self.actor_losses = []
        self.critic_losses = []
        self.alpha_losses = []
    
    def select_action(self, state, evaluate=False):
        """Select action based on current policy."""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        if evaluate:
            # Deterministic action for evaluation
            with torch.no_grad():
                mean, _ = self.actor(state)
                return torch.tanh(mean).cpu().numpy().flatten()
        else:
            # Sample action for training
            with torch.no_grad():
                action, _ = self.actor.sample(state)
                return action.cpu().numpy().flatten()
    
    def update_parameters(self, batch):
        """Update policy and value parameters using batch of experience tuples."""
        states, actions, rewards, next_states, dones = batch
        
        # Update critics
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next = self.critic1_target(next_states, next_actions)
            q2_next = self.critic2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + (1 - dones) * GAMMA * q_next
        
        # Critic loss
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic1_loss = F.mse_loss(q1, q_target)
        critic2_loss = F.mse_loss(q2, q_target)
        critic_loss = critic1_loss + critic2_loss
        
        # Update critic networks
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()
        
        # Update actor
        new_actions, log_probs = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp()
        
        # Soft update target networks
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
            
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        
        # Monitor gradient norms
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        critic1_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 10.0)
        critic2_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 10.0)
        
        # Print every 1000 updates
        if random.random() < 0.001:  # Sample 0.1% of updates for logging
            print(f"Agent {self.agent_idx} - Grad norms: Actor {actor_grad_norm:.4f}, Critic1 {critic1_grad_norm:.4f}, Critic2 {critic2_grad_norm:.4f}")
            print(f"Alpha: {self.alpha.item():.4f}")
        
        # Record losses
        self.critic_losses.append(critic_loss.item())
        self.actor_losses.append(actor_loss.item())
        self.alpha_losses.append(alpha_loss.item())
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha.item()
        }

class DistributedMBPO:
    """Distributed Model-Based Policy Optimization algorithm with GP comparison."""
    
    def __init__(self):
        # Create environment
        self.env = MultiAgentEnvironment(n_agents=N_AGENTS, n_rewards=10, max_steps=MAX_STEPS, action_penalty=ACTION_PENALTY)
        
        # Create agents
        self.agents = [SACAgent(i, AGENT_STATE_DIM, ACTION_DIM) for i in range(N_AGENTS)]
        
        # Create replay buffer for environment data (one per agent)
        self.replay_buffers = [ReplayBuffer(BUFFER_SIZE, BATCH_SIZE) for _ in range(N_AGENTS)]
        
        # Create model buffers for each agent (imaginary data)
        self.model_buffers = [ReplayBuffer(BUFFER_SIZE, BATCH_SIZE) for _ in range(N_AGENTS)]
        
        # Initialize GP models
        self.centralized_model = None
        self.distributed_model = None
        
        # For collecting initial data
        self.data_collection_steps = 5000
        
        # For tracking rewards
        self.episode_rewards = []
        self.steps_per_episode = []
        
        # For comparing GP models
        self.prediction_errors = {
            'centralized': [],
            'distributed': [],
            'difference': []
        }
    
    def collect_initial_data(self):
        """Collect initial data with random actions."""
        print("Collecting initial data...")
        
        global_state = self.env.reset()
        agent_states = self._extract_agent_states(global_state)
        
        for step in range(self.data_collection_steps):
            # Random actions for all agents
            actions = np.random.uniform(-1, 1, (N_AGENTS, ACTION_DIM))
            
            next_global_state, rewards, done, _ = self.env.step(actions)
            next_agent_states = self._extract_agent_states(next_global_state)
            
            # Store transition for each agent
            for i in range(N_AGENTS):
                self.replay_buffers[i].add(agent_states[i], actions[i], rewards, next_agent_states[i], float(done))
            
            agent_states = next_agent_states
            
            if done:
                global_state = self.env.reset()
                agent_states = self._extract_agent_states(global_state)
                
            if (step + 1) % 1000 == 0:
                print(f"Collected {step + 1} initial steps")
        
        print("Initial data collection complete")
    
    def _extract_agent_states(self, global_state):
        """Extract individual agent states from the global state."""
        agent_states = []
        for i in range(N_AGENTS):
            start_idx = i * AGENT_STATE_DIM
            end_idx = (i + 1) * AGENT_STATE_DIM
            agent_states.append(global_state[start_idx:end_idx])
        return agent_states
    
    def initialize_model(self):
        """Initialize both centralized and distributed GP models with collected data."""
        print("Initializing GP models...")
        
        # Prepare data for each agent
        dataset_list = []
        
        for i in range(N_AGENTS):
            # Sample data from replay buffer
            if len(self.replay_buffers[i]) >= MODEL_BATCH_SIZE:
                indices = np.random.choice(len(self.replay_buffers[i].buffer), MODEL_BATCH_SIZE, replace=False)
                batch = [self.replay_buffers[i].buffer[idx] for idx in indices]
                states, actions, _, next_states, _ = zip(*batch)
                
                states = np.array(states)
                actions = np.array(actions)
                next_states = np.array(next_states)
                
                # Compute delta states
                delta_states = next_states - states
                
                # Input is state + action
                X = np.concatenate([states, actions], axis=1)
                Y = delta_states
                
                dataset_list.append((X, Y))
            else:
                raise ValueError(f"Not enough data in replay buffer for agent {i} to initialize model")
        
        # Initialize centralized model
        self.centralized_model = CentralizedGP(
            AGENT_STATE_DIM, 
            ACTION_DIM, 
            kernel_type="combined"  # Use combined RBF + Linear kernel
        )
        centralized_loss = self.centralized_model.initialize_model(dataset_list)
        print(f"Centralized GP initialized with MSE: {centralized_loss:.4f}")
        
        # Initialize distributed model
        self.distributed_model = DistributedGP(
            N_AGENTS, 
            AGENT_STATE_DIM, 
            ACTION_DIM,
            kernel_type="combined"
        )
        self.distributed_model.initialize_model(dataset_list)
        
        # Initial comparison
        self.compare_gp_models(dataset_list)
    
    def generate_imaginary_data(self, use_distributed=True):
        """Generate imaginary data using the selected GP model."""
        model_name = "distributed" if use_distributed else "centralized"
        print(f"Generating imaginary data using {model_name} GP model...")
        
        # Track buffer sizes before generating data
        buffer_sizes_before = [len(buffer) for buffer in self.model_buffers]
    
        # Generate imaginary trajectories for each agent
        for agent_idx, agent in enumerate(self.agents):
            # Sample real states from agent's replay buffer as starting points
            if len(self.replay_buffers[agent_idx]) < BATCH_SIZE:
                continue
            
            indices = np.random.choice(len(self.replay_buffers[agent_idx].buffer), BATCH_SIZE, replace=False)
            initial_transitions = [self.replay_buffers[agent_idx].buffer[idx] for idx in indices]
            
            for initial_transition in initial_transitions:
                # Use real state as starting point
                state = initial_transition[0]  # Initial state
                
                for _ in range(IMAGINARY_ROLLOUT_LENGTH):
                    # Select action using agent's policy
                    action = agent.select_action(state)
                    
                    # Prepare input for GP model
                    model_input = np.concatenate([state, action]).reshape(1, -1)
                    
                    # Predict next state using the selected model
                    if use_distributed:
                        delta_state_mean, delta_state_var = self.distributed_model.predict_with_variance(agent_idx, model_input)
                    else:
                        delta_state_mean, delta_state_var = self.centralized_model.predict_with_variance(model_input)
                    
                    # Add noise proportional to uncertainty
                    delta_state = delta_state_mean.flatten()
                    
                    # Compute next state
                    next_state = state + delta_state
                    
                    # Use environment reward function instead of approximate reward
                    env_copy = copy.deepcopy(self.env)
                    # Set the environment to the current state
                    all_agent_positions = np.zeros((N_AGENTS, AGENT_STATE_DIM))
                    all_agent_positions[agent_idx] = state
                    env_copy.agent_positions = all_agent_positions
                    
                    # Step environment with just this agent's action
                    all_actions = np.zeros((N_AGENTS, ACTION_DIM))
                    all_actions[agent_idx] = action
                    _, reward, _, _ = env_copy.step(all_actions)

                    # Add optimism to reward
                    reward += OPTIMISM_SCALE * np.sqrt(np.sum(delta_state_var))
                    
                    # Store transition
                    self.model_buffers[agent_idx].add(state, action, reward, next_state, False)
                    
                    # Update state for next step in rollout
                    state = next_state
    
        # Print buffer sizes for debugging
        buffer_sizes_after = [len(buffer) for buffer in self.model_buffers]
        print(f"Model buffer sizes before: {buffer_sizes_before}")
        print(f"Model buffer sizes after: {buffer_sizes_after}")
    
    def update_model(self):
        """Update both GP models with new data."""
        print("Updating GP models...")
        
        # Prepare data for each agent
        dataset_list = []
        
        for i in range(N_AGENTS):
            # Sample data from replay buffer
            if len(self.replay_buffers[i]) >= MODEL_BATCH_SIZE:
                indices = np.random.choice(len(self.replay_buffers[i].buffer), MODEL_BATCH_SIZE, replace=False)
                batch = [self.replay_buffers[i].buffer[idx] for idx in indices]
                states, actions, _, next_states, _ = zip(*batch)
                
                states = np.array(states)
                actions = np.array(actions)
                next_states = np.array(next_states)
                
                # Compute delta states
                delta_states = next_states - states
                
                # Input is state + action
                X = np.concatenate([states, actions], axis=1)
                Y = delta_states
                
                dataset_list.append((X, Y))
            else:
                raise ValueError(f"Not enough data in replay buffer for agent {i}")
        
        # Update centralized model
        centralized_loss = self.centralized_model.update_model(dataset_list)
        print(f"Centralized model loss: {centralized_loss:.4f}")
        
        # Update distributed model
        distributed_loss = self.distributed_model.update_model(dataset_list)
        print(f"Distributed model average loss: {distributed_loss:.4f}")
        
        # Compare models after update
        self.compare_gp_models(dataset_list)
        
        return centralized_loss  
    
    def compare_gp_models(self, dataset_list):
        """Compare predictions of centralized and distributed GP models."""
        total_centralized_mse = 0
        total_distributed_mse = 0
        total_difference_mse = 0
        total_samples = 0
        
        for agent_idx, (X, Y) in enumerate(dataset_list):
            # Get predictions from both models
            centralized_preds = self.centralized_model.predict(X)
            distributed_preds = self.distributed_model.predict(agent_idx, X)
            
            # Compute MSE against ground truth for both models
            centralized_mse = np.mean(np.square(Y - centralized_preds))
            distributed_mse = np.mean(np.square(Y - distributed_preds))
            
            # Compute MSE between model predictions
            difference_mse = np.mean(np.square(centralized_preds - distributed_preds))
            
            # Weight by number of samples
            n_samples = X.shape[0]
            total_samples += n_samples
            total_centralized_mse += centralized_mse * n_samples
            total_distributed_mse += distributed_mse * n_samples
            total_difference_mse += difference_mse * n_samples
        
        # Compute weighted averages
        avg_centralized_mse = total_centralized_mse / total_samples
        avg_distributed_mse = total_distributed_mse / total_samples
        avg_difference_mse = total_difference_mse / total_samples
        
        # Store results for plotting
        self.prediction_errors['centralized'].append(avg_centralized_mse)
        self.prediction_errors['distributed'].append(avg_distributed_mse)
        self.prediction_errors['difference'].append(avg_difference_mse)
        
        print(f"Model Comparison - Centralized MSE: {avg_centralized_mse:.4f}, "
            f"Distributed MSE: {avg_distributed_mse:.4f}, Difference: {avg_difference_mse:.4f}")
        
        # Visualize comparison every few updates
        if len(self.prediction_errors['centralized']) % 5 == 0:
            self.plot_gp_comparison()

    def plot_gp_comparison(self):
        """Plot comparison of centralized and distributed GP models."""
        updates = range(1, len(self.prediction_errors['centralized']) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(updates, self.prediction_errors['centralized'], 'b-', label='Centralized GP MSE')
        plt.plot(updates, self.prediction_errors['distributed'], 'r-', label='Distributed GP MSE')
        plt.plot(updates, self.prediction_errors['difference'], 'g--', label='Prediction Difference')
        plt.title('GP Model Comparison')
        plt.xlabel('Model Updates')
        plt.ylabel('Mean Squared Error')
        plt.legend()
        plt.grid(True)
        
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        
        plt.savefig(f'results/gp_comparison_update_{len(updates)}.png')
        plt.close()
    
    def train_step(self, total_steps):
        """Perform one training step using both real and model data."""
        losses = {
            'critic_loss': [],
            'actor_loss': [],
            'alpha_loss': []
        }
        
        # Update models and generate imaginary data periodically
        if total_steps % MODEL_TRAIN_FREQ == 0:
            self.update_model()
            # Alternate between centralized and distributed model
            self.generate_imaginary_data(use_distributed=USE_DISTRIBUTED_MODEL)

        for _ in range(UPDATES_PER_STEP):
        
            # Update each agent using combined real and model data
            for agent_idx, agent in enumerate(self.agents):
                real_buffer = self.replay_buffers[agent_idx]
                model_buffer = self.model_buffers[agent_idx]
                
                # Skip if real buffer doesn't have enough data
                if len(real_buffer) < BATCH_SIZE:
                    continue
                
                # Case 1: If we have model data, mix it according to REAL_RATIO
                if len(model_buffer) >= BATCH_SIZE:
                    # Calculate number of samples from each buffer
                    real_samples = max(int(BATCH_SIZE * REAL_RATIO), 1)  # At least 1 real sample
                    model_samples = BATCH_SIZE - real_samples
                    
                    # Sample from real buffer
                    real_indices = np.random.choice(len(real_buffer.buffer), real_samples, replace=False)
                    real_batch = [real_buffer.buffer[idx] for idx in real_indices]
                    
                    # Sample from model buffer
                    model_indices = np.random.choice(len(model_buffer.buffer), model_samples, replace=False)
                    model_batch = [model_buffer.buffer[idx] for idx in model_indices]
                    
                    # Combine batches and shuffle
                    combined_batch = real_batch + model_batch
                    random.shuffle(combined_batch)  # Mix real and model data
                    
                    # Convert to tensors
                    states, actions, rewards, next_states, dones = zip(*combined_batch)
                    batch = (
                        torch.FloatTensor(states).to(device),
                        torch.FloatTensor(actions).to(device),
                        torch.FloatTensor(rewards).unsqueeze(1).to(device),
                        torch.FloatTensor(next_states).to(device),
                        torch.FloatTensor(dones).unsqueeze(1).to(device)
                    )
                
                # Case 2: If no model data yet, just use real data
                else:
                    batch = real_buffer.sample()
                
                # Update agent parameters with the batch
                update_info = agent.update_parameters(batch)
                
                # Record losses
                for k, v in update_info.items():
                    if k in losses:
                        losses[k].append(v)
        
        # Average losses across agents
        avg_losses = {}
        for k, v in losses.items():
            if v:
                avg_losses[k] = sum(v) / len(v)
        
        return avg_losses
    
    def train(self, max_steps=MAX_TRAINING_STEPS, eval_interval=EVAL_INTERVAL, run_id=None, plot_dir=None):
        """Train the distributed MBPO algorithm."""
        print("Starting training...")
        
        # Collect initial data
        self.collect_initial_data()
        
        # Initialize model
        self.initialize_model()
        
        # Generate initial imaginary data
        self.generate_imaginary_data()
        
        # Training loop
        global_state = self.env.reset()
        agent_states = self._extract_agent_states(global_state)
        episode_reward = 0
        episode_steps = 0
        episode_num = 0
        
        # For visualization
        critic_losses = []
        actor_losses = []
        alpha_losses = []
        model_losses = []
        eval_rewards = []
        steps_list = []
        
        for total_steps in range(1, max_steps + 1):
            # Select actions for each agent using its own state
            actions = np.zeros((N_AGENTS, ACTION_DIM))
            for i in range(N_AGENTS):
                actions[i] = self.agents[i].select_action(agent_states[i])
            
            # Environment step
            next_global_state, reward, done, _ = self.env.step(actions)
            next_agent_states = self._extract_agent_states(next_global_state)
            
            # Store transition for each agent
            for i in range(N_AGENTS):
                self.replay_buffers[i].add(agent_states[i], actions[i], reward, next_agent_states[i], float(done))
            
            # Update state and rewards
            agent_states = next_agent_states
            episode_reward += reward
            episode_steps += 1
            
            # Train agents
            if all(len(buffer) >= BATCH_SIZE for buffer in self.replay_buffers):
                losses = self.train_step(total_steps)
                
                if losses:
                    critic_losses.append(losses.get('critic_loss', 0))
                    actor_losses.append(losses.get('actor_loss', 0))
                    alpha_losses.append(losses.get('alpha_loss', 0))
                
                if self.centralized_model and self.centralized_model.model_losses:
                    model_losses = self.centralized_model.model_losses
            
            # Reset if episode is done
            if done:
                print(f"Episode {episode_num + 1}: Reward = {episode_reward:.2f}, Steps = {episode_steps}")
                self.episode_rewards.append(episode_reward)
                self.steps_per_episode.append(episode_steps)
                
                # Reset environment
                global_state = self.env.reset()
                agent_states = self._extract_agent_states(global_state)
                episode_reward = 0
                episode_steps = 0
                episode_num += 1
            
            # Evaluate and log
            if total_steps % eval_interval == 0:
                eval_reward = self.evaluate()
                eval_rewards.append(eval_reward)
                steps_list.append(total_steps)
                
                # Plot losses and rewards
                self.plot_training_curves(
                    steps_list, 
                    critic_losses, 
                    actor_losses, 
                    alpha_losses, 
                    model_losses, 
                    eval_rewards,
                    total_steps,
                    run_id=run_id,
                    plot_dir=plot_dir
                )
                
                print(f"Step {total_steps}: Eval reward = {eval_reward:.2f}")
        
        print("Training complete!")
        return self.episode_rewards, self.steps_per_episode
    
    def evaluate(self, n_episodes=5):
        """Evaluate current policies."""
        total_reward = 0
        
        for _ in range(n_episodes):
            global_state = self.env.reset()
            agent_states = self._extract_agent_states(global_state)
            episode_reward = 0
            done = False
            
            while not done:
                actions = np.zeros((N_AGENTS, ACTION_DIM))
                for i in range(N_AGENTS):
                    actions[i] = self.agents[i].select_action(agent_states[i], evaluate=True)
                
                next_global_state, reward, done, _ = self.env.step(actions)
                next_agent_states = self._extract_agent_states(next_global_state)
                
                episode_reward += reward
                agent_states = next_agent_states
            
            total_reward += episode_reward
        
        return total_reward / n_episodes

    def trained_policy(self, env):
        """Policy function for evaluation with the environment."""
        # Extract global state and convert to individual agent states
        global_state = env.get_global_state()
        agent_states = self._extract_agent_states(global_state)
        
        # Get actions from each agent
        actions = np.zeros((N_AGENTS, ACTION_DIM))
        for i in range(N_AGENTS):
            actions[i] = self.agents[i].select_action(agent_states[i], evaluate=True)
            
        return actions

    def plot_training_curves(self, steps, critic_losses, actor_losses, alpha_losses, model_losses, eval_rewards, step, run_id=None, plot_dir=None):
        """Plot and save training curves."""
        # Original plots for training curves (keeping as is)
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Critic loss
        if critic_losses:
            axs[0, 0].plot(critic_losses[-100:], label='Critic Loss')
            axs[0, 0].set_title('Critic Loss (last 100 updates)')
            axs[0, 0].set_xlabel('Update Step')
            axs[0, 0].set_ylabel('Loss')
            axs[0, 0].grid(True)
        
        # Actor loss
        if actor_losses:
            axs[0, 1].plot(actor_losses[-100:], label='Actor Loss')
            axs[0, 1].set_title('Actor Loss (last 100 updates)')
            axs[0, 1].set_xlabel('Update Step')
            axs[0, 1].set_ylabel('Loss')
            axs[0, 1].grid(True)
        
        # Alpha loss and model loss
        if alpha_losses and model_losses:
            ax1 = axs[1, 0]
            ax2 = ax1.twinx()
            
            ax1.plot(alpha_losses[-100:], 'b-', label='Alpha Loss')
            ax1.set_xlabel('Update Step')
            ax1.set_ylabel('Alpha Loss', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            
            ax2.plot(model_losses, 'r-', label='Model Loss')
            ax2.set_ylabel('Model Loss', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            axs[1, 0].set_title('Alpha and Model Losses')
        
        # Evaluation rewards
        if steps and eval_rewards:
            axs[1, 1].plot(steps, eval_rewards, 'g-', marker='o')
            axs[1, 1].set_title('Evaluation Rewards')
            axs[1, 1].set_xlabel('Environment Steps')
            axs[1, 1].set_ylabel('Average Reward')
            axs[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        if run_id:
            plt.savefig(f'{plot_dir}/training_curves_step_{step}.png')
        else:
            plt.savefig(f'results/training_curves_step_{step}.png')
        plt.close()
        
        # Plot GP model comparison if we have data
        if self.prediction_errors['centralized'] and self.prediction_errors['distributed']:
            self.plot_gp_comparison()

def main(seed=None):
    """Main function to run the distributed MBPO algorithm with GP comparison."""
    # Set seed if provided
    if seed is not None:
        global SEED
        SEED = seed
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
    
    # Create timestamp for unique run identification
    timestamp = int(time.time())
    run_id = f"seed_{SEED}_time_{timestamp}"
    
    # Create output directories
    results_dir = "results"
    data_dir = f"{results_dir}/data"
    plot_dir = f"{results_dir}/{run_id}"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    print(f"Starting training run with seed {SEED}")
    
    # Initialize and train the algorithm
    dist_mbpo = DistributedMBPO()
    episode_rewards, steps_per_episode = dist_mbpo.train(
        max_steps=MAX_TRAINING_STEPS,  # Adjust based on available computational resources
        eval_interval=EVAL_INTERVAL,
        run_id=run_id,
        plot_dir=plot_dir
    )
    
    # Collect all training history data
    training_history = {
        'seed': SEED,
        'timestamp': timestamp,
        'episode_rewards': episode_rewards,
        'steps_per_episode': steps_per_episode,
        'prediction_errors': {
            'centralized': dist_mbpo.prediction_errors['centralized'],
            'distributed': dist_mbpo.prediction_errors['distributed'],
            'difference': dist_mbpo.prediction_errors['difference']
        },
        'agent_losses': {
            'critic_losses': [agent.critic_losses for agent in dist_mbpo.agents],
            'actor_losses': [agent.actor_losses for agent in dist_mbpo.agents],
            'alpha_losses': [agent.alpha_losses for agent in dist_mbpo.agents]
        },
        'model_losses': dist_mbpo.centralized_model.model_losses if dist_mbpo.centralized_model else [],
    }
    
    # Save to JSON file
    filename = f"{data_dir}/history_seed_{SEED}.json"
    with open(filename, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_data = {k: v if not isinstance(v, (np.ndarray, list)) or not isinstance(v[0], np.ndarray)
                     else [x.tolist() if isinstance(x, np.ndarray) else x for x in v]
                     for k, v in training_history.items()}
        
        # Handle nested dictionaries
        for k, v in json_data.items():
            if isinstance(v, dict):
                json_data[k] = {k2: v2 if not isinstance(v2, (np.ndarray, list)) or not isinstance(v2[0], np.ndarray)
                              else [x.tolist() if isinstance(x, np.ndarray) else x for x in v2]
                              for k2, v2 in v.items()}
                
        json.dump(json_data, f, indent=2)
    
    print(f"Training history saved to {filename}")
    
    # Plot final training results in the run-specific directory
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig(f'{plot_dir}/final_rewards.png')
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps_per_episode)
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.grid(True)
    plt.savefig(f'{plot_dir}/steps_per_episode.png')
    
    # Final GP model comparison
    plt.figure(figsize=(10, 6))
    updates = range(1, len(dist_mbpo.prediction_errors['centralized']) + 1)
    plt.plot(updates, dist_mbpo.prediction_errors['centralized'], 'b-', label='Centralized GP MSE')
    plt.plot(updates, dist_mbpo.prediction_errors['distributed'], 'r-', label='Distributed GP MSE')
    plt.plot(updates, dist_mbpo.prediction_errors['difference'], 'g--', label='Prediction Difference')
    plt.title('Final GP Model Comparison')
    plt.xlabel('Model Updates')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{plot_dir}/final_gp_comparison.png')
    
    # Show a demo of trained agents
    env = MultiAgentEnvironment(n_agents=N_AGENTS, n_rewards=10, max_steps=MAX_STEPS, action_penalty=ACTION_PENALTY)
    
    # Run episode with video
    env.run_episode_with_policy(
        policy_func=dist_mbpo.trained_policy,
        render=True,
        save_video=True,
        video_name=f"{run_id}_trained_distributed_mbpo"
    )

def run_multiple_seeds(seeds=[0, 1, 2, 3, 4]):
    """Run multiple training instances with different random seeds."""
    for seed in seeds:
        print("===============================================")
        print(f"Starting training with seed {seed}")
        print("===============================================")
        main(seed=seed)

def plot_aggregated_results(seeds=[0, 1, 2, 3, 4]):
    """Load results from multiple seeds and plot aggregated statistics."""
    data_dir = "results/data"
    
    all_histories = []
    for seed in seeds:
        filename = f"{data_dir}/history_seed_{seed}.json"
        try:
            with open(filename, 'r') as f:
                history = json.load(f)
                all_histories.append(history)
            print(f"Loaded data for seed {seed}")
        except FileNotFoundError:
            print(f"No data found for seed {seed}")
    
    if not all_histories:
        print("No data found for plotting")
        return
    
    # Create output directory for aggregated plots
    agg_plot_dir = "results/aggregated"
    os.makedirs(agg_plot_dir, exist_ok=True)
    
    # Plot episode rewards across seeds
    plt.figure(figsize=(10, 6))
    
    # Find shortest history length to align data
    min_episodes = min(len(h['episode_rewards']) for h in all_histories)
    
    # Plot individual seed data
    for i, h in enumerate(all_histories):
        plt.plot(h['episode_rewards'][:min_episodes], alpha=0.3, 
                 label=f"Seed {h['seed']}")
    
    # Plot mean and std
    rewards_array = np.array([h['episode_rewards'][:min_episodes] for h in all_histories])
    mean_rewards = np.mean(rewards_array, axis=0)
    std_rewards = np.std(rewards_array, axis=0)
    
    plt.plot(mean_rewards, 'k-', linewidth=2, label="Mean")
    plt.fill_between(range(len(mean_rewards)), 
                     mean_rewards - std_rewards, 
                     mean_rewards + std_rewards, 
                     alpha=0.2, color='k')
    
    plt.title('Episode Rewards Across Seeds')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{agg_plot_dir}/aggregated_rewards.png')
    
    # Add similar plots for model losses, prediction errors, etc.
    # ...
    
    print(f"Aggregated plots saved to {agg_plot_dir}/")

if __name__ == "__main__":
    # Default seeds
    default_seeds = [0, 1, 2, 3, 4]
    selected_seeds = None

    # Parse command-line arguments
    for arg in sys.argv[1:]:
        if arg.startswith("seed="):
            try:
                seed_val = int(arg.split("=")[1])
                selected_seeds = [seed_val]
            except ValueError:
                print("Invalid seed value. Using default seeds.")
                selected_seeds = default_seeds

    if selected_seeds is None:
        selected_seeds = default_seeds

    run_multiple_seeds(seeds=selected_seeds)
    plot_aggregated_results(seeds=selected_seeds)