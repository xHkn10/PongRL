import torch.nn as nn

layer = nn.Linear(6, 64)

print(layer.weight.shape)