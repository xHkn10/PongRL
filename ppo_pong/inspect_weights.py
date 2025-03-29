import matplotlib.pyplot as plt
import numpy as np
import torch
from ppo import PolicyValueNet
from DQN_PONG_features.model import DQN
import os

net = PolicyValueNet(6, 64, 3)
#net = DQN(3)

ppo_directory = r"C:\Users\hknak\DL\ppo_pong\models"
dqn_directory = r"C:\Users\hknak\DL\DQN_PONG_features\models"

plt.ion()

column_labels = ["r_l", "b_y", "b_x", "v_y", "v_x", "r_r"]

for file in os.listdir(ppo_directory):
    model_path = os.path.join(ppo_directory, file)
    net.load_state_dict(torch.load(model_path))

    for module in net.modules():
        if isinstance(module, torch.nn.Linear):
            plt.clf()
            weight_matrix = module.weight.detach().numpy()

            plt.imshow(weight_matrix, cmap="gray", aspect="auto")
            plt.title(f"Model: {file}")
            plt.colorbar()

            plt.xticks(ticks=np.arange(len(column_labels)), labels=column_labels, rotation=45, ha="right")

            plt.pause(0.01)
            break

plt.ioff()
plt.show()
