"""A minimal CONDITIONAL FLOW MATCHING demo (not CosyVoice).

Train a velocity field that transforms x0 ~ N(0, 1) toward either
x1 ~ N(-2, 0.1**2) (condition 0) or x1 ~ N(+2, 0.1**2) (condition 1).
Run: python conditional_flow_1d.py
Requires: torch
"""
import torch
from torch import nn


torch.manual_seed(7)
torch.set_num_threads(2)

class VelocityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x, t, cond):
        # x_t, flow time t and class condition are all scalar here.
        return self.net(torch.cat([x, t, cond], dim=1))

model = VelocityNet()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

for step in range(1001):
    batch_size = 256
    cond = torch.randint(0, 2, (batch_size, 1)).float()
    x0 = torch.randn(batch_size, 1)
    x1 = torch.where(cond == 0, -2.0, 2.0) + 0.1 * torch.randn_like(x0)
    t = torch.rand(batch_size, 1)

    x_t = (1.0 - t) * x0 + t * x1
    target_velocity = x1 - x0
    pred_velocity = model(x_t, t, cond)
    loss = ((pred_velocity - target_velocity) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step in (0, 200, 500, 1000):
        print(f'train step={step:4d}, velocity MSE={loss.item():.4f}')

@torch.no_grad()
def generate(cond_value, initial_noise=0.5, n_steps=80):
    x = torch.tensor([[initial_noise]], dtype=torch.float32)
    c = torch.tensor([[float(cond_value)]])
    trajectory = [x.item()]
    for i in range(n_steps):
        t = torch.tensor([[i / n_steps]])
        v = model(x, t, c)
        x = x + v / n_steps  # Euler integration
        trajectory.append(x.item())
    return trajectory

print('\nInference with the SAME initial noise, but different conditions:')
for c, name in [(0, '低目标 -2'), (1, '高目标 +2')]:
    path = generate(c)
    print(f'{name:12s}: start={path[0]:+.3f}, halfway={path[40]:+.3f}, end={path[-1]:+.3f}')

# Optional inspect the full path with your plotting library of choice.

try:
    import matplotlib.pyplot as plt
    from pathlib import Path
    destination = Path(__file__).with_name('inference_trajectories.png')
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for c, label in [(0, 'condition 0: target near -2'),
                     (1, 'condition 1: target near +2')]:
        path = generate(c)
        ax.plot([i / (len(path) - 1) for i in range(len(path))], path, label=label)
    ax.set_xlabel('Flow time t (not physical audio time)')
    ax.set_ylabel('Generated scalar x(t)')
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    print(f'\nSaved: {destination}')
except ImportError:
    print('matplotlib not installed; skipping trajectory plot')
