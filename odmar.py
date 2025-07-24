import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
from collections import deque
import random
import time
from env import MultiAgentEnvironment
import os

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
N_AGENTS = 5
AGENT_STATE_DIM = 2  # Each agent has 2D position (x, y)
ACTION_DIM = 2  # Each agent has 2D action (dx, dy)
GLOBAL_STATE_DIM = N_AGENTS * AGENT_STATE_DIM  # Used for environment interaction
CONSENSUS_ROUNDS = 10

# Hyperparameters
BUFFER_SIZE = 1000000
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005  # For soft update
ALPHA_LR = 3e-4
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
MODEL_BATCH_SIZE = 256
MODEL_TRAIN_FREQ = 250
REAL_RATIO = 0.05  # Ratio of real to model data for policy update
IMAGINARY_ROLLOUT_LENGTH = 5  # Length of model rollouts
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
        self.lengthscale = 10.0  # Hyperparameter for RBF kernel
        self.noise_var = 1.0  # Observation noise variance
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
    
    def initialize_model(self, dataset_list, n_inducing=100):
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
    """Distributed Model-Based Policy Optimization algorithm."""
    
    def __init__(self):
        # Create environment
        self.env = MultiAgentEnvironment(n_agents=N_AGENTS, n_rewards=10, max_steps=200)
        
        # Create agents
        self.agents = [SACAgent(i, AGENT_STATE_DIM, ACTION_DIM) for i in range(N_AGENTS)]
        
        # Create replay buffer for environment data (one per agent)
        self.replay_buffers = [ReplayBuffer(BUFFER_SIZE, BATCH_SIZE) for _ in range(N_AGENTS)]
        
        # Create model buffers for each agent (imaginary data)
        self.model_buffers = [ReplayBuffer(BUFFER_SIZE, BATCH_SIZE) for _ in range(N_AGENTS)]
        
        # Initialize distributed GP model
        self.model = None
        
        # For collecting initial data
        self.data_collection_steps = 5000
        
        # For tracking rewards
        self.episode_rewards = []
        self.steps_per_episode = []
    
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
        """Initialize the distributed GP model with collected data."""
        print("Initializing distributed GP model...")
        
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
                
                # Compute delta states (for the agent's own state)
                delta_states = next_states - states
                
                # Input is state + action
                X = np.concatenate([states, actions], axis=1)
                Y = delta_states
                
                dataset_list.append((X, Y))
            else:
                raise ValueError(f"Not enough data in replay buffer for agent {i} to initialize model")
        
        # Initialize model
        self.model = DistributedGP(
            N_AGENTS, 
            AGENT_STATE_DIM, 
            ACTION_DIM, 
            graph_type="Cycle", 
            kernel_type="combined"  # Use combined RBF + Linear kernel
        )
        self.model.initialize_model(dataset_list)
    
    def generate_imaginary_data(self):
        """Generate imaginary data using the model for MBPO."""
        print("Generating imaginary data...")
        
        # Clear previous imaginary data
        for buffer in self.model_buffers:
            buffer.buffer.clear()
        
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
                    model_input = np.concatenate([state, action])
                    
                    # Predict next state using the model
                    delta_state_mean, delta_state_var = self.model.predict_with_variance(
                        agent_idx, model_input.reshape(1, -1))
                    
                    # Add noise proportional to uncertainty
                    noise = np.random.normal(0, np.sqrt(delta_state_var))
                    delta_state = delta_state_mean.flatten() + noise
                    
                    # Compute next state
                    next_state = state + delta_state
                    
                    # Simple reward estimation (can be improved)
                    reward = -0.01 * np.sum(next_state**2)  # Simple penalty for distance from origin
                    
                    # Add to agent's model buffer
                    self.model_buffers[agent_idx].add(state, action, reward, next_state, False)
                    
                    # Update state for next step in rollout
                    state = next_state
    
    def update_model(self):
        """Update the distributed GP model with new data."""
        print("Updating model...")
        
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
        
        # Update model
        model_loss = self.model.update_model(dataset_list)
        print(f"Model loss: {model_loss:.4f}")
    
    def train_step(self, total_steps):
        """Perform one training step."""
        losses = {
            'critic_loss': [],
            'actor_loss': [],
            'alpha_loss': []
        }
        
        # Update each agent using only real environment data
        for agent_idx, agent in enumerate(self.agents):
            # Use only real data from environment
            if len(self.replay_buffers[agent_idx]) >= BATCH_SIZE:
                # Sample batch from replay buffer
                batch = self.replay_buffers[agent_idx].sample()
                
                # Update agent parameters
                update_info = agent.update_parameters(batch)
                
                # Record losses
                for k, v in update_info.items():
                    if k in losses:
                        losses[k].append(v)
        
        # Skip model updates for debugging
        if False and total_steps % MODEL_TRAIN_FREQ == 0:
            self.update_model()
            self.generate_imaginary_data()
        
        # Average losses across agents
        avg_losses = {}
        for k, v in losses.items():
            if v:
                avg_losses[k] = sum(v) / len(v)
        
        return avg_losses
    
    def train(self, max_steps=1000000, eval_interval=5000):
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
                
                if self.model and self.model.model_losses:
                    model_losses = self.model.model_losses
            
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
                    total_steps
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

    def plot_training_curves(self, steps, critic_losses, actor_losses, alpha_losses, model_losses, eval_rewards, step):
        """Plot and save training curves."""
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
        
        plt.savefig(f'results/training_curves_step_{step}.png')
        plt.close()

def main():
    """Main function to run the distributed MBPO algorithm."""
    # Create output directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Initialize and train the algorithm
    dist_mbpo = DistributedMBPO()
    episode_rewards, steps_per_episode = dist_mbpo.train(
        max_steps=500000,  # Adjust based on available computational resources
        eval_interval=5000
    )
    
    # Plot final training results
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig('results/final_rewards.png')
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps_per_episode)
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.grid(True)
    plt.savefig('results/steps_per_episode.png')
    
    # Show a demo of trained agents
    env = MultiAgentEnvironment(n_agents=N_AGENTS, n_rewards=5, max_steps=100)
    
    # Run episode with video
    env.run_episode_with_policy(
        policy_func=dist_mbpo.trained_policy,
        render=True,
        save_video=True,
        video_name="trained_distributed_mbpo"
    )

if __name__ == "__main__":
    main()