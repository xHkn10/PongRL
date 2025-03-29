import torch
import numpy as np
from model import DQN
from pong_env import PongGame
import matplotlib.pyplot as plt


def main():
    net = DQN(3)

    net.load_state_dict(torch.load("models/model_5ft8000000.pth"))
    net.eval()
    env = PongGame()
    total_reward = .0
    state = env.reset()

    actions_taken = [0, 0, 0]

    plt.ion()

    frame_idx = 0


    while True:
        frame_idx += 1
        action = net.act(state)
        actions_taken[action] += 1

        state, reward, is_done = env.step(action, right_bot=True)
        total_reward += reward
        plt.imshow(env.grid)
        plt.draw()
        plt.pause(.005)
        plt.clf()
        plt.text(15, .5, f"{env.racket_l.score, env.racket_r.score}", ha="center", va="center",
                 fontsize=20, weight="bold")
        plt.axis("off")


   # print("Total frames", frame_idx)
   # print("Taken actions:", actions_taken)
   # print(f"Total reward: {total_reward}")


if __name__ == '__main__':
    main()
