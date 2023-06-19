import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Agent.pg_agent import PG_Agent
from common.replay_buffer import Buffer
from torch.utils.tensorboard import SummaryWriter


class Runner:
    def __init__(self, args, env):
        self.args = args
        self.epsilon = args.epsilon
        self.min_epsilon = args.min_epsilon

        self.env = env
        self.episode_limit = args.episode_len
        self.agent = self._init_agents()
        self.buffer = Buffer(args)

        self.save_path = self.args.save_dir + '/' + self.args.env_name + '/' + self.args.algorithm + '/' + ('order_%d' % self.args.order)
        self.model_save_path = self.args.model_save_dir + '/' + self.args.env_name + '/' + self.args.algorithm + '/' + ('order_%d' % self.args.order)
        self.log_path = 'runs/' + self.args.env_name + '/' + self.args.algorithm + '/' + ('order%d' % self.args.order)

        if not os.path.exists(self.save_path) and not self.args.evaluate:
            os.makedirs(self.save_path)

        if self.args.log and (not self.args.evaluate):
            if not os.path.exists(self.log_path):
                os.makedirs(self.log_path)
            self.logger = SummaryWriter(self.log_path)
        else:
            self.logger = None

    def _init_agents(self):
        agent = PG_Agent(self.args)
        return agent

    def run(self):
        reward_eval = []
        for episode in tqdm(range(1, self.args.max_episodes + 1)):
            state = self.env.reset()
            done = False
            time_step = 1

            while not done:
                with torch.no_grad:
                    action = self.agent.select_action(state, self.epsilon)
                next_state, reward, done, info = self.env.step(action)
                if time_step + 1 > self.episode_limit:
                    done = True
                self.buffer.store_episode(state, action, reward, next_state, done)

                state = next_state
                time_step += 1

            if episode % self.args.episodes_per_train == 0:
                for train_epoch in range(self.args.epoches):
                    transitions = self.buffer.sample(self.args.batch_size)
                    self.agent.learn(transitions)

                self.buffer = Buffer(self.args)

            if episode % self.args.evaluate_period == 0:
                reward_eval.append(self.evaluate())
                plt.figure()
                plt.plot(range(len(reward_eval)), reward_eval)
                plt.xlabel('Episode * %d' % self.args.evaluate_period)
                plt.ylabel('Average returns')
                plt.savefig(self.save_path + '/plt.png', format='png')

        self.agent.policy.save_model(self.model_save_path, train_step=None)
        # saves final episode reward for plotting training curve later
        np.save(self.save_path + '/reward_eval', reward_eval)

        if self.logger is not None:
            self.logger.close()
        print('...Finished training.')

    def evaluate(self):
        returns = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            state = self.env.reset()
            episode_rewards = 0
            done = False
            time_step = 1
            while not done:
                # print('current_episode', episode, time_step)
                # self.env.render()
                action = self.agent.select_action(state, 0)
                next_state, reward, done, info = self.env.step(action)
                episode_rewards += reward
                state = next_state
                time_step += 1
                if time_step > self.args.evaluate_episode_len:
                    done = True
            returns.append(episode_rewards)
            # print('Returns is', rewards)
        return sum(returns) / self.args.evaluate_episodes