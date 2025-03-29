import torch
from model import DQN
import matplotlib.pyplot as plt

net = DQN(3)
version = 0
plt.ion()
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Increase figure size

for i in range(800):
    net.load_state_dict(torch.load(f"models/model_5ft{version}.pth"))
    version += 10_000

    # Don't clear the entire figure, just the axes
    for ax in axes:
        ax.clear()

    j = 0
    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            im = axes[j].imshow(m.weight.detach().cpu().numpy(), cmap="gray", aspect="auto")
            axes[j].set_title(f"Layer {j} - Version {version}")
            axes[j].grid(True, color='white', linestyle='-', linewidth=0.5, alpha=0.3)  # Add grid lines
            #fig.colorbar(im, ax=axes[j])
            j += 1

    plt.tight_layout()  # Improve spacing
    plt.draw()
    plt.pause(0.1)  # Slightly longer pause for better visualization

for name, param in net.named_parameters():
    if "weight" in name:
        plt.figure(figsize=(6, 4))
        plt.hist(param.detach().cpu().numpy().flatten(), bins=50, alpha=0.75, color='blue')
        plt.xlabel("Weight Value")
        plt.ylabel("Frequency")
        plt.title(f"Weight Distribution: {name}")
        plt.grid(True)
        plt.show()
