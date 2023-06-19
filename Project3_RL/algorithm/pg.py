import torch
import os
import numpy as np
from torch.distributions import Categorical
from network.base_net import Actor, Critic
from common.utils import hard_update, soft_update

class PolicyGradient:
    def __init__(self, args):
        self.args = args

        # create the network
        self.actor = Actor(args)
        self.critic = Critic(args)

        # build up the target network
        self.target_actor = Actor(args)
        self.target_critic = Critic(args)

        if args.cuda:
            self.actor.cuda()
            self.critic.cuda()
            self.target_actor.cuda()
            self.target_critic.cuda()

        if self.args.load_model:
            if os.path.exists(self.args.model_save_dir + '/evaluate_model/actor_params.pkl'):
                path_actor = self.args.model_save_dir + '/evaluate_model/actor_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.actor.load_state_dict(torch.load(path_actor, map_location=map_location))
                print('Successfully load the network: {}'.format(path_actor))
            else:
                raise Exception("No network!")

        # load the weights into the target networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr_critic)

    # update the network
    def train(self, transitions):
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
            target_act_probs = self.target_actor(next_states)
            target_act = Categorical(target_act_probs).sample().long()
            next_action_onehot = torch.zeros([self.args.batch_size, self.args.n_actions])
            index = np.indices((self.args.batch_size, ))
            next_action_onehot[index, target_act] = 1
            if self.args.cuda:
                next_action_onehot = next_action_onehot.cuda()
            q_next = self.target_critic(next_states, next_action_onehot)
            target_q = (rewards.unsqueeze(1) + self.args.gamma * q_next * done_multiplier)

        # the q loss
        curr_action_onehot = torch.zeros([self.args.batch_size, self.args.n_actions])
        curr_action_onehot[index, actions] = 1
        q_eval = self.critic(states, curr_action_onehot)
        critic_loss = (target_q - q_eval).pow(2).mean()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # the actor loss
        act_probs = self.actor(states)
        states_rep = states.unsqueeze(dim=1).expand(-1, self.args.n_actions, -1).reshape(-1, self.args.state_shape)
        act_rep = torch.eye(self.args.n_actions).unsqueeze(dim=0).expand(self.args.batch_size, -1, -1)
        act_rep = act_rep.reshape(-1, self.args.n_actions)      # (bs*n_actions, n_actions)
        if self.args.cuda:
            states_rep = states_rep.cuda()
            act_rep = act_rep.cuda()
        q_cf_all = self.critic(states_rep, act_rep).reshape(self.args.batch_size, self.args.n_actions)

        probs = self.actor(states)
        log_probs = torch.log(torch.sum(probs * curr_action_onehot, dim=1, keepdim=True)+1e-15)

        bias = torch.sum(act_probs * q_cf_all, dim=1, keepdim=True).detach()
        advantage = (q_eval - bias).detach()
        actor_loss = - (log_probs * advantage).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        soft_update(self.target_actor, self.actor, self.args.tau)
        soft_update(self.target_critic, self.critic, self.args.tau)

    def save_model(self, model_save_path, train_step):
        if train_step is not None:
            num = str(train_step // self.args.save_rate)
        else:
            num = 'final'
        model_path = os.path.join(model_save_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.actor.state_dict(), model_path + '/' + num + '_actor_params.pkl')
        torch.save(self.critic.state_dict(),  model_path + '/' + num + '_critic_params.pkl')