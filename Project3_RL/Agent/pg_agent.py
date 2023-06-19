import numpy as np
import torch
from torch.distributions import Categorical
from algorithm.pg import PolicyGradient

class PG_Agent:
    def __init__(self, args):
        self.args = args
        self.policy = PolicyGradient(args)

    def select_action(self, states, evaluate=False):
        inputs = torch.tensor(states, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        probs = self.policy.actor(inputs).squeeze(0)
        if not evaluate:
            act_sample = Categorical(probs).sample().long()
        else:
            act_sample = torch.argmax(probs)
        action = act_sample.cpu().numpy()
        return action.copy()

    def learn(self, transitions):
        self.policy.train(transitions)