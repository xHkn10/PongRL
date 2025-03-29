from torch import nn
import numpy as np
import torch

class DQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        # (y_l, x, y, v_y, v_x)
        # (B, 5)

        self.model = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, 128, bias=False),
            nn.LeakyReLU(.01),
            nn.LayerNorm(128),
            nn.Linear(128, n_actions, bias=False)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu", mode="fan_out")

    def forward(self, x):
        return self.model(x)

    def act(self, state, temperature=1):
        if isinstance(state, torch.Tensor):
            state.unsqueeze_(0)
        else:
            state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0)
        q_vals = self(state).detach().cpu().numpy()[0]
        action = np.argmax(q_vals)
        return action
