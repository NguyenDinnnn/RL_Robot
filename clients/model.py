import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, in_channels, height, width, n_actions):
        super(ActorCritic, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.flatten_size = 64 * height * width
        self.fc_shared = nn.Linear(self.flatten_size, 128)
        self.actor = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.flatten_size)
        x = torch.relu(self.fc_shared(x))
        policy_logits = self.actor(x)
        value = self.critic(x)
        return policy_logits, value
