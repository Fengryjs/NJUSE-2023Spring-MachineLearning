import torch
import os
import numpy as np
from network.base_net import MLP
from common.utils import hard_update, soft_update


class DQN:
    def __init__(self, args):
        self.args = args
        self.train_step = 0

        # create the network
        self.q_network = MLP(args)
        self.target_q_network = MLP(args)

        if args.cuda:
            self.q_network.cuda()
            self.target_q_network.cuda()

        # Only done in evaluation
        if self.args.load_model:
            if os.path.exists(self.args.model_save_dir + '/evaluate_model/q_params.pkl'):
                path_q = self.args.model_save_dir + '/evaluate_model/q_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.q_network.load_state_dict(torch.load(path_q, map_location=map_location))
                print('Successfully load the network: {}'.format(path_q))
            else:
                raise Exception("No pre-learned network!")

        # load the weights into the target networks
        hard_update(self.target_q_network, self.q_network)

        # create the optimizer
        self.q_optim = torch.optim.Adam(self.q_network.parameters(), lr=self.args.lr_q)


    def train(self, transitions, logger):
        states = transitions['state']
        actions = transitions['action']
        rewards = transitions['reward']
        next_states = transitions['next_state']
        dones = transitions['done']

        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions).long()
        rewards = torch.from_numpy(rewards).float()
        next_states = torch.from_numpy(next_states).float()
        dones = torch.from_numpy(dones).float()
        if self.args.cuda:
            states = states.cuda()
            actions = actions.cuda()
            rewards = rewards.cuda()
            next_states = next_states.cuda()
            dones = dones.cuda()

        dones = dones.unsqueeze(dim=1)
        done_multiplier = - (dones - 1)

        # calculate the target Q value function
        with torch.no_grad():
            q_next_all = self.target_q_network(next_states)
            q_next = torch.max(q_next_all, dim=1, keepdim=True)[0]
            # TODO: 补全target_q的计算相关代码实现
            target_q = rewards + done_multiplier * q_next  #

        q_eval_all = self.q_network(states)
        actions = actions.unsqueeze(dim=1)
        q_eval = torch.gather(q_eval_all, dim=1, index=actions)
        td_loss = (target_q - q_eval).pow(2).mean()

        if logger is not None:
            if self.train_step % 1000 == 0:
                logger.add_scalar('td_loss', td_loss, self.train_step // 1000)

        self.q_optim.zero_grad()
        td_loss.backward()
        self.q_optim.step()

        self.train_step += 1
        if self.train_step > 0 and (self.train_step % self.args.target_update_cycle) == 0:
            hard_update(self.target_q_network, self.q_network)


    def save_model(self, model_save_path, train_step):
        if train_step is not None:
            num = str(train_step // self.args.save_rate)
        else:
            num = 'final'
        model_path = os.path.join(model_save_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.q_network.state_dict(), model_path + '/' + num + '_q_params.pkl')