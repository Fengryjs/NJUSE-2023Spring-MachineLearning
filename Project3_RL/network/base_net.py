import torch
import torch.nn as nn
import torch.nn.functional as f

class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(args.state_shape, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, args.n_actions)

    def forward(self, state):
        x1 = f.relu(self.fc1(state))
        x2 = f.relu(self.fc2(x1))
        x3 = self.fc3(x2)
        return x3


class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(args.state_shape, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, args.n_actions)

    def forward(self, state):
        x1 = f.relu(self.fc1(state))
        x2 = f.relu(self.fc2(x1))
        out = self.fc3(x2)
        probs = f.softmax(out, dim=-1)
        return probs


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear((args.state_shape + args.n_actions), 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        concat_inps = torch.cat([state, action], dim=-1)
        x1 = f.relu(self.fc1(concat_inps))
        x2 = f.relu(self.fc2(x1))
        out = self.fc3(x2)
        return out