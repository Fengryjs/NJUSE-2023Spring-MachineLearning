import torch
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Agent.dqn_agent import DQN_Agent
from common.replay_buffer import Buffer
from torch.utils.tensorboard import SummaryWriter


class Runner:
    def __init__(self, args, env):
        self.args = args
        self.epsilon = args.epsilon
        self.min_epsilon = args.min_epsilon
        self.anneal_epsilon = float((self.epsilon - self.min_epsilon) / self.args.epsilon_anneal_time)

        self.env = env
        self.episode_limit = args.episode_len
        self.agent = self._init_agent()
        self.buffer = Buffer(args)

        self.save_path = self.args.save_dir + '/' + self.args.env_name + '/' + self.args.algorithm + '/' + ('order_%d' % self.args.order)
        self.model_save_path = self.args.model_save_dir + '/' + self.args.env_name + '/' + self.args.algorithm + '/' + ('order_%d' % self.args.order)
        self.model_load_path = self.args.model_save_dir + '/' + self.args.env_name + '/' + self.args.algorithm
        self.log_path = 'runs/' + self.args.env_name + '/' + self.args.algorithm + '/' + ('order%d' % self.args.order)

        if not os.path.exists(self.save_path) and not self.args.evaluate:
            os.makedirs(self.save_path)

        if self.args.log and (not self.args.evaluate):
            if not os.path.exists(self.log_path):
                os.makedirs(self.log_path)
            self.logger = SummaryWriter(self.log_path)
        else:
            self.logger = None

    def _init_agent(self):
        agent = DQN_Agent(self.args)
        return agent

    def run(self):
        reward_eval = []
        state, info = self.env.reset()

        done = False
        # Function render() is used to visualize the environment, which may makes the learning slow.
        # self.env.render()

        # from episode = 1 to M(Max time step) 最大时间长限制
        for time_step in tqdm(range(1, self.args.max_time_steps + 1)):
            # 如果到达时间限制，或者结束了 done = True Reset 环境
            if time_step % self.episode_limit == 0 or done:
                state, info = self.env.reset()
                done = False
            # 进行 epsilon-greedy policy 贪婪搜索
            with torch.no_grad():
                action = self.agent.select_action(state, self.epsilon)
            # new version API
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated | truncated
            # Introduce extra bonus rewards.
            # position, velocity = next_state
            # reward = abs(position - (-0.5))  # r in [0, 1]

            self.buffer.store_episode(state, action, reward, next_state, done)

            state = next_state
            if self.buffer.current_size >= self.args.batch_size:
                transitions = self.buffer.sample(self.args.batch_size)
                self.agent.learn(transitions, self.logger)

            if time_step % self.args.evaluate_rate == 0:
                reward_eval.append(self.evaluate())
                plt.figure()
                plt.plot(range(len(reward_eval)), reward_eval)
                plt.xlabel('T * %d' % self.args.evaluate_rate)
                plt.ylabel('Average returns')
                plt.savefig(self.save_path + '/plt.png', format='png')
                plt.show()
            self.epsilon = max(self.min_epsilon, self.epsilon - self.anneal_epsilon)

        self.agent.policy.save_model(self.model_save_path, train_step=None)
        # Saves final episode reward for plotting training curve later
        np.save(self.save_path + '/reward_eval', reward_eval)

        if self.logger is not None:
            self.logger.close()
        print('...Finished training.')

    def evaluate(self):
        returns = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            state, info = self.env.reset()
            episode_rewards = 0
            done = False
            time_step = 1
            while not done:
                # print('current_episode', episode, time_step)
                # self.env.render()
                with torch.no_grad():
                    action = self.agent.select_action(state, epsilon=0)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated | truncated
                # 解除这边的注释
                self.env.render()
                episode_rewards += reward
                state = next_state
                time_step += 1
                if time_step > self.args.evaluate_episode_len:
                    done = True
            returns.append(episode_rewards)
        return sum(returns) / self.args.evaluate_episodes