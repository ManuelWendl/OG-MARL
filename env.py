import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, List, Dict


class MultiAgentEnvironment:
    def __init__(self, 
                 n_agents: int = 5,
                 world_bounds: Tuple[float, float, float, float] = (-10, 10, -10, 10),
                 reward_sizes: List[float] = [1, 5, 10],
                 n_rewards: int = 6,
                 noise_scale: float = 0.1,
                 max_steps: int = 100):
        """
        Multi-agent environment with sparse rewards.
        
        Args:
            n_agents: Number of agents
            world_bounds: (x_min, x_max, y_min, y_max)
            reward_sizes: Available reward values
            n_rewards: Number of rewards to place
            noise_scale: Scale of noise added to agent dynamics
            max_steps: Maximum steps per episode
        """
        self.n_agents = n_agents
        self.world_bounds = world_bounds
        self.reward_sizes = reward_sizes
        self.n_rewards = n_rewards
        self.noise_scale = noise_scale
        self.max_steps = max_steps
        
        # Normalize rewards to be in [0, 1]
        if self.reward_sizes and max(self.reward_sizes) > 0:
            max_reward = max(self.reward_sizes)
            self.reward_sizes = [r / max_reward for r in self.reward_sizes]

        # World dimensions
        self.x_min, self.x_max, self.y_min, self.y_max = world_bounds
        
        # Initialize agent-specific noise parameters
        self.agent_noise_params = self._initialize_agent_noise()
        
        # Environment state
        self.agent_positions = np.zeros((n_agents, 2))  # [x, y] for each agent
        self.rewards = []  # List of (x, y, value) tuples
        self.collected_rewards = set()  # Indices of collected rewards
        self.current_step = 0
        
        # Initialize rewards
        self.reset()
    
    def _initialize_agent_noise(self) -> np.ndarray:
        """Initialize individual noise parameters for each agent."""
        # Each agent has different noise characteristics for x and y movements
        noise_params = np.random.normal(1.0, self.noise_scale, (self.n_agents, 2))
        return noise_params
    
    def _generate_rewards(self):
        """Generate evenly distributed rewards on the map."""
        self.rewards = []
        
        # Calculate grid dimensions for even distribution
        n_cols = int(np.ceil(np.sqrt(self.n_rewards)))
        n_rows = int(np.ceil(self.n_rewards / n_cols))
        
        # Calculate spacing
        x_spacing = (self.x_max - self.x_min) / (n_cols + 1)
        y_spacing = (self.y_max - self.y_min) / (n_rows + 1)
        
        # Place rewards in a grid pattern
        reward_idx = 0
        for row in range(n_rows):
            for col in range(n_cols):
                if reward_idx >= self.n_rewards:
                    break
                    
                x = self.x_min + (col + 1) * x_spacing
                y = self.y_min + (row + 1) * y_spacing
                
                # Cycle through reward values to ensure variety
                value = self.reward_sizes[reward_idx % len(self.reward_sizes)]
                self.rewards.append((x, y, value))
                reward_idx += 1
    
    def _clip_positions(self, positions: np.ndarray) -> np.ndarray:
        """Clip agent positions to world bounds."""
        positions[:, 0] = np.clip(positions[:, 0], self.x_min, self.x_max)
        positions[:, 1] = np.clip(positions[:, 1], self.y_min, self.y_max)
        return positions
    
    def reset(self) -> np.ndarray:
        """Reset the environment and return initial state."""
        # Fixed evenly distributed initial positions for all agents
        if self.n_agents == 1:
            # Single agent at center
            self.agent_positions[0] = [0.0, 0.0]
        else:
            # Distribute agents evenly in a circle around the center
            angles = np.linspace(0, 2 * np.pi, self.n_agents, endpoint=False)
            radius = min(self.x_max - self.x_min, self.y_max - self.y_min) * 0.3  # 30% of world size
            
            for i in range(self.n_agents):
                self.agent_positions[i, 0] = radius * np.cos(angles[i])
                self.agent_positions[i, 1] = radius * np.sin(angles[i])
        
        # Generate fixed rewards (will be the same every reset)
        self._generate_rewards()
        self.collected_rewards = set()
        self.current_step = 0
        
        return self.get_global_state()
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            actions: Array of shape (n_agents, 2) with [dx, dy] for each agent
            
        Returns:
            state: Global state (stacked agent positions)
            reward: Total reward collected this step
            done: Whether episode is finished
            info: Additional information
        """
        assert actions.shape == (self.n_agents, 2), f"Expected actions shape {(self.n_agents, 2)}, got {actions.shape}"
        
        # Apply actions with individual agent noise
        noisy_actions = actions * self.agent_noise_params
        self.agent_positions += noisy_actions
        
        # Clip positions to world bounds
        self.agent_positions = self._clip_positions(self.agent_positions)
        
        # Check for reward collection
        total_reward = self._collect_rewards()
        
        # Update step counter
        self.current_step += 1
        
        # Check if episode is done (only based on max_steps now, since rewards are continuous)
        done = (self.current_step >= self.max_steps)
        
        # Prepare info
        info = {
            'visited_rewards': len(self.collected_rewards),  # Number of rewards visited at least once
            'total_rewards': len(self.rewards),
            'step': self.current_step
        }
        
        return self.get_global_state(), total_reward, done, info
    
    def _collect_rewards(self) -> float:
        """
        Check if any agent is on a reward location and return the total reward.
        A reward is collected if an agent is within a certain threshold of it.
        Rewards can be collected at every timestep an agent is on the location.
        """
        total_reward = 0.0
        
        # Iterate through each agent
        for agent_pos in self.agent_positions:
            # Iterate through all rewards
            for i, reward in enumerate(self.rewards):
                reward_pos = np.array(reward[:2])
                distance = np.linalg.norm(agent_pos - reward_pos)
                
                # If agent is close enough, collect the reward
                if distance < 0.5:  # Collection threshold
                    total_reward += reward[2]  # Add the reward's value
                    self.collected_rewards.add(i) # Mark as visited (for logging/info)
                    
        return total_reward

    def get_global_state(self) -> np.ndarray:
        """Return the global state (flattened agent positions)."""
        return self.agent_positions.flatten()
    
    def render(self, ax=None, title: str = "Multi-Agent Environment"):
        """Visualize the current state of the environment."""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Clear the axes
        ax.clear()
        
        # Set world bounds
        ax.set_xlim(self.x_min - 1, self.x_max + 1)
        ax.set_ylim(self.y_min - 1, self.y_max + 1)
        ax.set_aspect('equal')
        
        # Draw world boundaries
        rect = patches.Rectangle(
            (self.x_min, self.y_min), 
            self.x_max - self.x_min, 
            self.y_max - self.y_min,
            linewidth=2, edgecolor='black', facecolor='none'
        )
        ax.add_patch(rect)
        
        # Draw rewards
        currently_collecting = set()
        for i, (rx, ry, value) in enumerate(self.rewards):
            # Check if any agent is currently on this reward
            is_currently_collecting = False
            for agent_pos in self.agent_positions:
                distance = np.linalg.norm(agent_pos - np.array([rx, ry]))
                if distance < 0.5:  # Collection radius
                    is_currently_collecting = True
                    currently_collecting.add(i)
                    break
            
            if is_currently_collecting:
                # Currently being collected - bright with pulsing effect
                color = 'yellow' if value == 1 else 'orange' if value == 5 else 'red'
                size = 80 + value * 30  # Larger size for active collection
                ax.scatter(rx, ry, c=color, s=size, marker='*', 
                          edgecolors='white', linewidth=3, alpha=1.0)
                # Add collection radius indicator
                circle = plt.Circle((rx, ry), 0.5, fill=False, color=color, 
                                  linestyle='--', linewidth=2, alpha=0.6)
                ax.add_patch(circle)
            elif i in self.collected_rewards:
                # Previously visited but not currently collecting
                color = 'lightgray'
                size = 40 + value * 15
                ax.scatter(rx, ry, c=color, s=size, marker='*', 
                          edgecolors='gray', linewidth=1, alpha=0.5)
            else:
                # Never visited - full brightness
                color = 'gold' if value == 1 else 'orange' if value == 5 else 'red'
                size = 50 + value * 20
                ax.scatter(rx, ry, c=color, s=size, marker='*', 
                          edgecolors='black', linewidth=1, alpha=0.8)
        
        # Draw agents
        colors = plt.cm.tab10(np.linspace(0, 1, self.n_agents))
        for i, (ax_pos, color) in enumerate(zip(self.agent_positions, colors)):
            ax.scatter(ax_pos[0], ax_pos[1], c=[color], s=100, marker='o', 
                      edgecolors='black', linewidth=2, label=f'Agent {i}')
            
            # Add agent ID text
            ax.text(ax_pos[0] + 0.3, ax_pos[1] + 0.3, str(i), fontsize=8, fontweight='bold')
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f'{title} - Step {self.current_step}')
        ax.grid(True, alpha=0.3)
        
        # Add legend for rewards
        reward_handles = []
        # Available rewards
        for value in self.reward_sizes:
            color = 'gold' if value == 1 else 'orange' if value == 5 else 'red'
            size = 50 + value * 20
            handle = ax.scatter([], [], c=color, s=size, marker='*', 
                              edgecolors='black', linewidth=1, alpha=0.8, label=f'Reward {value}')
            reward_handles.append(handle)
        
        # Currently collecting
        if len(currently_collecting) > 0:
            handle = ax.scatter([], [], c='yellow', s=80, marker='*', 
                              edgecolors='white', linewidth=3, alpha=1.0, label='Collecting Now!')
            reward_handles.append(handle)
        
        # Previously visited
        if len(self.collected_rewards) > 0:
            handle = ax.scatter([], [], c='lightgray', s=40, marker='*', 
                              edgecolors='gray', linewidth=1, alpha=0.5, label='Previously Visited')
            reward_handles.append(handle)
        
        ax.legend(handles=reward_handles, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        return ax
    
    def get_info(self) -> Dict:
        """Get current environment information."""
        return {
            'n_agents': self.n_agents,
            'agent_positions': self.agent_positions.copy(),
            'total_rewards': len(self.rewards),
            'visited_rewards': len(self.collected_rewards),
            'current_step': self.current_step,
            'world_bounds': self.world_bounds,
            'agent_noise_params': self.agent_noise_params.copy()
        }
    
    def run_episode_with_policy(self, policy_func, render=True, save_video=True, 
                               video_name="environment_video", max_steps=None):
        """
        Run a complete episode with a given policy and optionally render/save video.
        
        Args:
            policy_func: Function that takes env and returns actions array
            render: Whether to create and show animation
            save_video: Whether to save video to file
            video_name: Name for the video file (without extension)
            max_steps: Override max_steps for this episode
            
        Returns:
            total_reward: Total reward collected
            episode_length: Number of steps taken
            states_history: List of states for each step (if render=True)
        """
        if max_steps is not None:
            original_max_steps = self.max_steps
            self.max_steps = max_steps
        
        # Reset environment
        state = self.reset()
        
        # Storage for animation
        states_history = [] if render else None
        step_rewards = []
        total_reward = 0
        
        print(f"Running episode with render={render}, save_video={save_video}")
        
        # Run episode
        for step in range(self.max_steps):
            # Store state for animation
            if render:
                states_history.append({
                    'agent_positions': self.agent_positions.copy(),
                    'rewards': self.rewards.copy(),
                    'collected_rewards': self.collected_rewards.copy(),
                    'step': step,
                    'total_reward': total_reward
                })
            
            # Get actions from policy
            actions = policy_func(self)
            
            # Take step
            state, reward, done, info = self.step(actions)
            total_reward += reward
            step_rewards.append(reward)
            
            if step % 10 == 0 or reward > 0:
                print(f"Step {step+1}: Reward = {reward:.1f}, Total = {total_reward:.1f}, "
                      f"Visited = {info['visited_rewards']}/{info['total_rewards']}")
            
            if done:
                print(f"Episode finished at step {step+1}")
                break
        
        # Add final state
        if render:
            states_history.append({
                'agent_positions': self.agent_positions.copy(),
                'rewards': self.rewards.copy(),
                'collected_rewards': self.collected_rewards.copy(),
                'step': step + 1,
                'total_reward': total_reward
            })
        
        # Create animation if requested
        if render and states_history:
            self._create_animation(states_history, step_rewards, save_video, video_name)
        
        # Restore original max_steps if it was overridden
        if max_steps is not None:
            self.max_steps = original_max_steps
        
        return total_reward, step + 1, states_history
    
    def _create_animation(self, states_history, step_rewards, save_video, video_name):
        """Create and optionally save animation."""
        import matplotlib.animation as animation
        
        print(f"Creating animation with {len(states_history)} frames...")
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        def animate_frame(frame_idx):
            """Animation function for each frame."""
            ax1.clear()
            ax2.clear()
            
            # Get state for this frame
            state_data = states_history[frame_idx]
            
            # Temporarily set environment state for rendering
            original_positions = self.agent_positions.copy()
            original_collected = self.collected_rewards.copy()
            original_step = self.current_step
            
            self.agent_positions = state_data['agent_positions']
            self.collected_rewards = state_data['collected_rewards']
            self.current_step = state_data['step']
            
            # Render environment state
            self.render(ax1, f"Step {state_data['step']} - Total Reward: {state_data['total_reward']:.1f}")
            
            # Plot reward history
            if frame_idx > 0:
                steps_so_far = list(range(1, min(frame_idx + 1, len(step_rewards) + 1)))
                rewards_so_far = step_rewards[:frame_idx]
                cumulative_so_far = np.cumsum(rewards_so_far)
                
                if len(steps_so_far) > 0 and len(rewards_so_far) > 0:
                    ax2.plot(steps_so_far, rewards_so_far, 'b-', 
                            linewidth=2, marker='o', label='Step Reward', markersize=4)
                    ax2.plot(steps_so_far, cumulative_so_far, 'r-', 
                            linewidth=2, marker='s', label='Cumulative Reward', markersize=4)
            
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Reward')
            ax2.set_title(f'Reward Progress (Frame {frame_idx+1}/{len(states_history)})')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, max(len(step_rewards), 10))
            if step_rewards:
                max_step_reward = max(step_rewards) if step_rewards else 1
                max_cum_reward = max(np.cumsum(step_rewards)) if step_rewards else 1
                ax2.set_ylim(0, max(max_step_reward, max_cum_reward) * 1.1)
            
            # Restore original state
            self.agent_positions = original_positions
            self.collected_rewards = original_collected
            self.current_step = original_step
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate_frame, frames=len(states_history),
            interval=300, repeat=True, blit=False
        )
        
        # Save video if requested
        if save_video:
            self._save_animation(anim, video_name)
        
        # Show animation
        plt.tight_layout()
        plt.show()
        
        return anim
    
    def _save_animation(self, anim, video_name):
        """Save animation to file."""
        print(f"Saving animation as {video_name}...")
        
        # Try MP4 first
        try:
            anim.save(f'{video_name}.mp4', writer='ffmpeg', fps=3, bitrate=1800)
            print(f"✓ Saved as {video_name}.mp4")
            return
        except Exception as e:
            print(f"Could not save MP4 (ffmpeg might not be available): {e}")
        
        # Try GIF as fallback
        try:
            anim.save(f'{video_name}.gif', writer='pillow', fps=2)
            print(f"✓ Saved as {video_name}.gif")
            return
        except Exception as e:
            print(f"Could not save GIF either: {e}")
            print("Animation will only be shown in window.")

def demo_environment():
    """Demonstrate the environment with random actions."""
    env = MultiAgentEnvironment(n_agents=3, n_rewards=10)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Initial state
    state = env.reset()
    env.render(ax1, "Initial State")
    
    # Run a few random steps
    total_reward = 0
    rewards_over_time = []
    
    for step in range(20):
        # Random actions for all agents
        actions = np.random.uniform(-1, 1, (env.n_agents, 2))
        
        state, reward, done, info = env.step(actions)
        total_reward += reward
        rewards_over_time.append(total_reward)
        
        if done:
            break
    
    # Final state
    env.render(ax2, f"After {step+1} Steps")
    
    plt.tight_layout()
    plt.show()
    
    # Plot rewards over time
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_over_time, 'b-', linewidth=2, marker='o')
    plt.xlabel('Step')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Rewards Over Time')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"Final Info: {env.get_info()}")
    print(f"Total Reward Collected: {total_reward}")


if __name__ == "__main__":
    demo_environment()