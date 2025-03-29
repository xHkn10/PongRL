import torch
from torch.optim import AdamW
import torch.nn.functional as F
from model import DQN
import numpy as np
from dataclasses import dataclass
from collections import deque
import random
import time
from pong_env import PongGame

torch.set_float32_matmul_precision("high")

GAMMA = .99
BATCH_SIZE = 32
REPLAY_SIZE = 10_000
LR = 1e-4
SYNC = 1_000
REPLAY_START = 10_000

EPSILON_START = 1.
EPSILON_FINAL = .01
DECAY_LEN = 120_000


@dataclass
class Experience:
    old_state: torch.Tensor
    action: int
    reward: float
    is_done: bool
    new_state: torch.Tensor


class ExperienceBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def __add__(self, experience: Experience):
        self.buffer.append(experience)
        return self

    def sample(self, batch_size: int) -> list[Experience]:
        indices = np.random.choice(len(self), batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]


class Agent:
    def __init__(self, env, exp_buffer: ExperienceBuffer):
        self.env: PongGame = env
        self.exp_buffer: ExperienceBuffer = exp_buffer
        self.total_reward = .0
        self._reset()

    def _reset(self):
        self.state = self.env.reset()

    @torch.no_grad()
    def play(self, net: DQN, device, epsilon):
        net.eval()
        if random.random() < epsilon:
            action = random.choice((0, 1, 2))
        else:
            state = torch.tensor(self.state, device=device, dtype=torch.float32)
            state.unsqueeze_(0)
            q_vals = net(state)
            action = torch.max(q_vals, dim=-1).indices
            action = action.item()

        total_reward = 0
        new_state, reward, is_done = self.env.step(action)
        total_reward += reward

        exp = Experience(old_state=torch.tensor(self.state, device=device, dtype=torch.float32),
                         action=action,
                         reward=reward,
                         is_done=is_done,
                         new_state=torch.tensor(new_state, device=device, dtype=torch.float32))

        self.exp_buffer += exp
        self.state = new_state
        if is_done:
            self._reset()

        net.train()
        return total_reward


def calculate_loss(batch, net, target_net, device):
    old_states, actions, rewards, is_dones, new_states = convert_buffer_to_tensor(batch, device)
    q_vals_pred = net(old_states).gather(1, actions.unsqueeze(-1)).squeeze()
    with torch.no_grad():
        next_q_actions = net(new_states).max(1).indices.unsqueeze(-1)
        next_q_vals = target_net(new_states).gather(1, next_q_actions).squeeze()
        next_q_vals[is_dones] = .0
        expected_q_vals = rewards + GAMMA * next_q_vals.detach()
    return F.mse_loss(q_vals_pred, expected_q_vals)


def convert_buffer_to_tensor(batch, device):
    old_states, actions, rewards, is_dones, new_states = [], [], [], [], []
    for exp in batch:
        old_states.append(exp.old_state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        is_dones.append(exp.is_done)
        new_states.append(exp.new_state)
    old_states = torch.stack(old_states)
    actions = torch.tensor(actions, device=device)
    rewards = torch.tensor(rewards, device=device)
    is_dones = torch.tensor(is_dones, device=device)
    new_states = torch.stack(new_states)
    return old_states, actions, rewards, is_dones, new_states


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    env = PongGame()

    net = DQN(3).to(device)
    net.train()
    target_net = DQN(3).to(device)

    exp_buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, exp_buffer)

    optimizer = AdamW(net.parameters(), lr=LR, weight_decay=1e-4)

    recent_rewards = deque(maxlen=100)
    frame_idx = 0
    while True:
        start = time.time()
        epsilon = max(EPSILON_START - frame_idx * (EPSILON_START - EPSILON_FINAL) / DECAY_LEN, EPSILON_FINAL)
        reward = agent.play(net, device, epsilon)
        recent_rewards.append(reward)
        if frame_idx % 100 == 0:
            print(f"Frame {frame_idx} | average reward: {sum(recent_rewards)}")
        if frame_idx % 10000 == 0:
            torch.save(net.state_dict(), f"models/model_5ft{frame_idx}.pth")
        frame_idx += 1
        if len(exp_buffer) < REPLAY_START:
            continue

        if frame_idx % SYNC == 0:
            target_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = exp_buffer.sample(BATCH_SIZE)
        loss = calculate_loss(batch, net, target_net, device)
        loss.backward()
        optimizer.step()

        if frame_idx % 1000 == 0:
            print(
                f"Frame {frame_idx} | Speed {1 / (time.time() - start):.2f} | Average Reward {sum(recent_rewards)} | Epsilon {epsilon}")


if __name__ == '__main__':
    main()
