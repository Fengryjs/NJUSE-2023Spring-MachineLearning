import os
import gym
import torch
import numpy as np
from common.arguments import get_args


if __name__ == '__main__':
    # get the params
    args = get_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    # 'MountainCar-V0', 'CartPole-v1' serve as the benchmark to validate the effectiveness of your algorithms.
    env = gym.make(args.env_name, render_mode="human")

    print('observation_space', env.observation_space, env.observation_space.high, env.observation_space.low)
    print('action_space', env.action_space, env.action_space.n)
    print(args)
    args.state_shape = env.observation_space.shape[0]
    args.n_actions = env.action_space.n

    if args.algorithm == 'DQN':
        from runner.runner_dqn import Runner
    elif args.algorithm =='PG':
        from runner.runner_pg import Runner
    else:
        raise Exception('Extra algorithms can be achieved yourselves.')

    runner = Runner(args, env)
    if args.evaluate:
        returns = runner.evaluate()
        print('Average returns is', returns)
    else:
        runner.run()
