#!/usr/bin/env python3
"""
Example usage of the MultiAgentEnvironment with integrated video rendering
"""

import numpy as np
import matplotlib.pyplot as plt
from env import MultiAgentEnvironment


def center_seeking_policy(env):
    """Policy that moves agents towards center if far, random exploration if close."""
    actions = np.zeros((env.n_agents, 2))
    
    for i, pos in enumerate(env.agent_positions):
        # If agent is far from center, move towards center
        center = np.array([0.0, 0.0])
        to_center = center - pos
        distance_to_center = np.linalg.norm(to_center)
        
        if distance_to_center > 5.0:
            # Move towards center
            actions[i] = to_center / distance_to_center * 0.5
        else:
            # Random exploration
            actions[i] = np.random.uniform(-0.8, 0.8, 2)
    
    return actions


def reward_seeking_policy(env):
    """Policy that seeks the highest value rewards first."""
    actions = np.zeros((env.n_agents, 2))
    
    # Get uncollected rewards
    uncollected_rewards = [
        (rx, ry, value, i) for i, (rx, ry, value) in enumerate(env.rewards)
        if i not in env.collected_rewards
    ]
    
    for i, pos in enumerate(env.agent_positions):
        if uncollected_rewards:
            # Find closest high-value reward
            reward_scores = []
            for rx, ry, value, _ in uncollected_rewards:
                distance = np.linalg.norm(pos - np.array([rx, ry]))
                score = value / (distance + 0.1)  # Higher score for closer, higher-value rewards
                reward_scores.append(score)
            
            best_reward_idx = np.argmax(reward_scores)
            rx, ry, _, _ = uncollected_rewards[best_reward_idx]
            
            # Move towards best reward
            to_reward = np.array([rx, ry]) - pos
            distance = np.linalg.norm(to_reward)
            if distance > 0.1:
                actions[i] = to_reward / distance * 0.7
                # Add some noise for exploration
                actions[i] += np.random.normal(0, 0.1, 2)
        else:
            # Random exploration if no rewards left
            actions[i] = np.random.uniform(-0.5, 0.5, 2)
    
    return actions


def random_policy(env):
    """Random exploration policy."""
    return np.random.uniform(-1, 1, (env.n_agents, 2))


def demo_with_rendering():
    """Demonstrate environment with video rendering."""
    print("=== Multi-Agent Environment Demo with Video ===")
    
    # Create environment
    env = MultiAgentEnvironment(
        n_agents=4,
        n_rewards=10,
        max_steps=50
    )
    
    print(f"Environment: {env.n_agents} agents, {len(env.rewards)} rewards")
    print(f"World bounds: {env.world_bounds}")
    
    # Run episode with center-seeking policy and video rendering
    total_reward, episode_length, _ = env.run_episode_with_policy(
        policy_func=center_seeking_policy,
        render=True,
        save_video=True,
        video_name="center_seeking_demo",
        max_steps=40
    )
    
    print("\nCenter-seeking policy results:")
    print(f"Total reward: {total_reward:.1f}")
    print(f"Episode length: {episode_length} steps")
    
    return env


def demo_without_rendering():
    """Demonstrate environment without video rendering for faster execution."""
    print("\n=== Fast Demo without Rendering ===")
    
    env = MultiAgentEnvironment(n_agents=3, n_rewards=8, max_steps=100)
    
    # Run episode without rendering (much faster)
    total_reward, episode_length, _ = env.run_episode_with_policy(
        policy_func=reward_seeking_policy,
        render=False,
        save_video=False
    )
    
    print("Reward-seeking policy results:")
    print(f"Total reward: {total_reward:.1f}")
    print(f"Episode length: {episode_length} steps")
    
    # Show final state
    fig, ax = plt.subplots(figsize=(8, 8))
    env.render(ax, "Final State - Reward Seeking Policy")
    plt.show()
    
    return env


def compare_policies():
    """Compare different policies with and without rendering."""
    print("\n=== Policy Comparison ===")
    
    policies = {
        "Random": random_policy,
        "Center-seeking": center_seeking_policy,
        "Reward-seeking": reward_seeking_policy
    }
    
    results = {}
    
    for name, policy in policies.items():
        print(f"\nTesting {name} policy...")
        env = MultiAgentEnvironment(n_agents=3, n_rewards=8, max_steps=50)
        
        # Run without rendering for comparison
        total_reward, episode_length, _ = env.run_episode_with_policy(
            policy_func=policy,
            render=False,
            save_video=False
        )
        
        results[name] = {
            'reward': total_reward,
            'length': episode_length,
            'efficiency': total_reward / episode_length if episode_length > 0 else 0
        }
        
        print(f"{name}: Reward={total_reward:.1f}, Steps={episode_length}, Efficiency={results[name]['efficiency']:.3f}")
    
    # Show best policy with video
    best_policy_name = max(results.keys(), key=lambda x: results[x]['reward'])
    print(f"\nBest policy: {best_policy_name}")
    print("Running best policy with video...")
    
    env = MultiAgentEnvironment(n_agents=3, n_rewards=8, max_steps=50)
    env.run_episode_with_policy(
        policy_func=policies[best_policy_name],
        render=True,
        save_video=True,
        video_name=f"best_policy_{best_policy_name.lower()}",
        max_steps=50
    )


def interactive_demo():
    """Interactive demo where user can choose options."""
    print("=== Interactive Multi-Agent Environment Demo ===")
    print("\nChoose a demo:")
    print("1. Single demo with video rendering")
    print("2. Fast demo without rendering")
    print("3. Compare all policies")
    print("4. Custom policy with video")
    
    try:
        choice = input("Enter choice (1-4) or press Enter for option 1: ").strip()
        
        if choice == "2":
            demo_without_rendering()
        elif choice == "3":
            compare_policies()
        elif choice == "4":
            print("Running custom reward-seeking policy with video...")
            env = MultiAgentEnvironment(n_agents=5, n_rewards=12, max_steps=60)
            env.run_episode_with_policy(
                policy_func=reward_seeking_policy,
                render=True,
                save_video=True,
                video_name="custom_demo"
            )
        else:
            demo_with_rendering()
            
    except KeyboardInterrupt:
        print("\nDemo interrupted.")
    except Exception as e:
        print(f"Error: {e}")
        print("Running default demo...")
        demo_with_rendering()


if __name__ == "__main__":
    interactive_demo()
