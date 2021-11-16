import torch
from torch import nn
import numpy as np


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.actor_fc = nn.Sequential(
            nn.Linear(in_features=24, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=4),
            nn.Tanh()
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float).to(self.device)
        if len(state.shape) == 1 or len(state.shape) == 3:
            state = state.unsqueeze(0)
        if len(state.shape) == 4:
            state = state.permute(0, 3, 1, 2)

        action = self.actor_fc(state)
        return action

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.critic_fc = nn.Sequential(
            nn.Linear(in_features=24, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1)
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float).to(self.device)
        if len(state.shape) == 1 or len(state.shape) == 3:
            state = state.unsqueeze(0)
        if len(state.shape) == 4:
            state = state.permute(0, 3, 1, 2)

        value = self.critic_fc(state)
        return value
