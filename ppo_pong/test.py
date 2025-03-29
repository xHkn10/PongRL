import gymnasium
from ppo import Agent
import numpy as np
from collections import deque
import torch

# Assuming your Agent, Actor, Critic, and PPOMemory classes are defined as provided

# Initialize the environment
env = gymnasium.make('CartPole-v1')

# Initialize the agent with parameters suitable for CartPole
agent = Agent(
    n_actions=env.action_space.n,          # 2 actions
    input_dims=env.observation_space.shape[0],  # 4 state dimensions
    gamma=0.99,                            # Discount factor
    lr=3e-4,                               # Learning rate
    gae_lambda=0.95,                       # GAE lambda
    policy_clip=0.2,                       # PPO clipping parameter
    batch_size=64,                         # Mini-batch size
    n_epochs=10                            # Number of epochs (though not fully used in your learn())
)

# Training loop parameters
max_timesteps_per_batch = 2048  # Collect this many timesteps before learning
total_timesteps = 0             # Total timesteps across all episodes
timesteps_since_learning = 0    # Timesteps since last learn() call
recent_rewards = deque(maxlen=100)  # Store rewards of last 100 episodes
episode_rewards = []            # Store all episode rewards
n_episodes = 1000               # Maximum number of episodes

# Training loop
for episode in range(n_episodes):
    state, _ = env.reset()
    done = False
    episode_reward = 0

    while not done:
        # Choose action
        action, prob, val = agent.choose_action(state)

        # Step the environment
        next_state, reward, done, _, _ = env.step(action)

        # Store experience
        agent.remember(state, action, prob, val, reward, done)

        # Update state and counters
        state = next_state
        episode_reward += reward
        total_timesteps += 1
        timesteps_since_learning += 1

        # Learn when enough timesteps are collected
        if timesteps_since_learning >= max_timesteps_per_batch:
            if not done:  # Batch ends mid-episode
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=agent.actor.device).unsqueeze(0)
                bootstrap_value = agent.critic(next_state_tensor).item()
                agent.learn(bootstrap_value=bootstrap_value)  # Pass to learn()
            else:
                agent.learn()  # No bootstrap needed if done=True
            timesteps_since_learning = 0

    # After episode ends, record reward
    episode_rewards.append(episode_reward)
    recent_rewards.append(episode_reward)

    # Monitor progress every 10 episodes
    if episode % 10 == 0 and episode > 0:
        avg_reward = np.mean(recent_rewards)
        print(f"Episode {episode}, Avg Reward (last {min(len(recent_rewards), 100)} episodes): {avg_reward:.2f}")

        # Check for convergence (solved if avg reward over 100 episodes >= 475)
        if len(recent_rewards) == 100 and avg_reward >= 475:
            print(f"Solved after {episode} episodes!")
            break

# Close the environment
env.close()