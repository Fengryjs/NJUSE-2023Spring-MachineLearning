import gym
import numpy as np
import torch
from algorithm.dqn import DQN
import gym.envs.classic_control.mountain_car
class DQN_Agent:
    def __init__(self, args):
        self.args = args
        self.policy = DQN(args)

    def select_action(self, states, epsilon):
        # TODO: 补全epsilon_greedy代码实现
        # 如果小于 epsilon
        if np.random.uniform() < epsilon:
            # {(): 1, (276,): 0, (275,): 2, (275, 276): 1}
            action = np.random.randint(self.args.n_actions)
        else:
            inputs = torch.tensor(states, dtype=torch.float32).unsqueeze(0)
            if self.args.cuda:
                inputs = inputs.cuda()
            q_value = self.policy.q_network.forward(inputs)
            action = np.argmax(q_value)
            action = action.cpu().numpy()
        return action

    def learn(self, transitions, logger):
        self.policy.train(transitions, logger)