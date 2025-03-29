import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np
from pong_env import PongGame
from DQN_PONG_features.model import DQN


class PolicyValueNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(hidden_dim, output_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.constant_(module.bias, 0)

    def zero_out_6th_neuron(self):
        self.shared[0].weight.data[:, -1] = .0

    def forward(self, x):
        x = self.shared(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value

    @torch.no_grad()
    def act(self, state, temperature=1):
        action_logits, _ = self(torch.tensor(np.array(state), dtype=torch.float))
        action_probs = torch.softmax(action_logits / temperature, -1)
        action = torch.multinomial(action_probs, 1).item()
        return action


# Compute returns and advantages using Generalized Advantage Estimation (GAE)
def compute_gae(rewards, values, dones, next_value, gamma, lambda_):
    T = len(rewards)
    advantages = np.zeros(T)
    advantage = 0
    for t in reversed(range(T)):
        if t == T - 1:
            next_v = next_value if not dones[t] else 0
        else:
            next_v = values[t + 1]
        delta = rewards[t] + gamma * next_v - values[t]
        advantage = delta if dones[t] else delta + gamma * lambda_ * advantage
        advantages[t] = advantage
    returns = advantages + values
    return returns, advantages


def ppo_update(network, optimizer, states, actions, log_probs_old, returns, advantages, epsilon, c1, c2, epochs,
               batch_size, device):
    states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.long, device=device)
    log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32, device=device)
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    advantages = torch.tensor(advantages, dtype=torch.float32, device=device)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(epochs):
        indices = np.random.permutation(len(states))
        for start in range(0, len(states), batch_size):
            end = start + batch_size
            idx = indices[start:end]
            states_mb = states[idx]
            actions_mb = actions[idx]
            log_probs_old_mb = log_probs_old[idx]
            returns_mb = returns[idx]
            advantages_mb = advantages[idx]

            policy_logits, values = network(states_mb)
            dist = torch.distributions.Categorical(logits=policy_logits)
            log_probs_new = dist.log_prob(actions_mb)
            entropy = dist.entropy().mean()

            ratio = torch.exp(log_probs_new - log_probs_old_mb)
            surr1 = ratio * advantages_mb
            surr2 = torch.clip(ratio, 1 - epsilon, 1 + epsilon) * advantages_mb
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (values.squeeze() - returns_mb).pow(2).mean()
            loss = policy_loss + c1 * value_loss - c2 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


@torch.no_grad()
def collect_trajectories(env, network, opponent, num_timesteps, device):
    states = []
    actions = []
    rewards = []
    dones = []
    log_probs = []
    values = []
    state = env.reset()

    for _ in range(num_timesteps):
        states.append(state)
        state_tensor = torch.tensor(state, dtype=torch.float, device=device)
        mirrored_state_tensor = torch.tensor(env.mirror_state(state), dtype=torch.float, device=device)

        policy_logits, value = network(state_tensor)
        dist = torch.distributions.Categorical(logits=policy_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        #opp_policy_logits, _ = opponent(mirrored_state_tensor)
        #dist = torch.distributions.Categorical(logits=opp_policy_logits)
        #opp_action = dist.sample().item()
        action = action.item()
        opp_action = opponent.act(mirrored_state_tensor[:-1])
        value = value.item()
        log_prob = log_prob.item()

        env._take_action(opp_action, racket="Right")
        next_state, reward, done = env.step(action, wind=True, right_bot=False)

        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob)
        values.append(value)

        if done:
            state = env.reset()
        else:
            state = next_state

    # Compute next_value for bootstrapping
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
    _, next_value = network(state_tensor)
    next_value = next_value.item() if not done else 0

    return states, actions, rewards, dones, log_probs, values, next_value


def train_ppo(env, num_iterations, num_timesteps, hidden_dim, lr, epochs, batch_size, gamma, lambda_, epsilon, c1, c2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 6
    output_dim = 3

    network = PolicyValueNet(input_dim, hidden_dim, output_dim).to(device)
    opponent = DQN(output_dim).to(device)

    network.load_state_dict(torch.load("models/model2000_competitive.pth"))
    #network.zero_out_6th_neuron()
    opponent.load_state_dict(torch.load("../DQN_PONG_features/models/model_5ft8000000.pth"))
    #opponent.zero_out_6th_neuron()

    optimizer = AdamW(network.parameters(), lr=lr, weight_decay=1e-5)

    for iteration in range(num_iterations):
        states, actions, rewards, dones, log_probs, values, next_value = collect_trajectories(env, network, opponent,
                                                                                              num_timesteps,
                                                                                              device)

        returns, advantages = compute_gae(rewards, values, dones, next_value, gamma, lambda_)
        ppo_update(network, optimizer, states, actions, log_probs, returns, advantages, epsilon, c1, c2, epochs,
                   batch_size, device)

        # log performance
        episode_rewards = []
        episode_reward = 0
        for reward, done in zip(rewards, dones):
            episode_reward += reward
            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0

        if episode_rewards:
            avg_episode_reward = np.mean(episode_rewards)
        else:
            avg_episode_reward = np.sum(rewards)  # Fallback in case no episodes finished

        print(f"Iteration {iteration + 1}/{num_iterations}, Average Episode Reward: {avg_episode_reward:.2f}")
        if iteration % 50 == 0:
            torch.save(network.state_dict(), f"models/model{iteration}_dest.pth")
            #opponent.load_state_dict(network.state_dict())


# Example usage
if __name__ == "__main__":
    env = PongGame()

    # Hyperparameters
    num_iterations = 100_000
    num_timesteps = 2048
    hidden_dim = 64
    lr = 3e-4
    epochs = 10
    batch_size = 64
    gamma = 0.99
    lambda_ = 0.95
    epsilon = 0.2
    c1 = 0.5
    c2 = 0.01

    train_ppo(env, num_iterations, num_timesteps, hidden_dim, lr, epochs, batch_size, gamma, lambda_, epsilon, c1, c2)
