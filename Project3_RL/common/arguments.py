import argparse

"""
Here are the param for the training

"""

def get_args():
    parser = argparse.ArgumentParser("Parameters of RL Algorithms.")

    parser.add_argument("--seed", type=int, default=678, help="random seed")
    parser.add_argument("--order", type=int, default=1, help="record the result under different parametric settings")
    parser.add_argument("--hidden_dim", type=int, default=64, help="hidden dimension of all actor and critic networks")

    parser.add_argument("--algorithm", type=str, default="DQN")
    parser.add_argument("--env_name", type=str, default="MountainCar-v0", choices=["MountainCar-v0", "CartPole-v1"])
    parser.add_argument("--episode-len", type=int, default=200, help="maximum episode length")
    parser.add_argument("--max_time_steps", type=int, default=1000000, help="number of time steps")

    parser.add_argument("--epsilon", type=float, default=1, help="epsilon greedy")
    parser.add_argument("--min_epsilon", type=float, default=0.1)
    parser.add_argument("--epsilon_anneal_time", type=int, default=100000)

    parser.add_argument("--lr_q", type=float, default=1e-3, help="learning rate of q network")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--buffer_size", type=int, default=int(2000), help="number of transitions can be stored in buffer")
    parser.add_argument("--batch_size", type=int, default=32, help="number of episodes to optimize at the same time")
    parser.add_argument("--target_update_cycle", type=int, default=200)

    parser.add_argument("--save_dir", type=str, default="./log", help="directory in which experimental results are saved")
    parser.add_argument("--model_save_dir", type=str, default="./model_log", help="directory in which models are saved")
    parser.add_argument("--save_rate", type=int, default=500000, help="model_save_interval (episode)")

    parser.add_argument("--log", type=bool, default=True, help="whether record the change of loss in the training process")
    parser.add_argument("--log_interval", type=int, default=2000, help="log interval for each logged data (episode)")

    parser.add_argument("--evaluate_episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate_episode_len", type=int, default=200, help="length of episodes for evaluating")
    parser.add_argument("--evaluate_rate", type=int, default=2000, help="how often to evaluate network (step)")

    parser.add_argument("--evaluate", type=bool, default=True, help="whether to evaluate the network")
    parser.add_argument("-load_model", type=bool, default=True, help="must keep track with the evaluate option")

    parser.add_argument("--cuda", type=bool, default=False)
    parser.add_argument("--device", type=str, default='0', help="which GPU is used")

    # Parameters that are only used for policy gradient algorithms
    parser.add_argument("--max_episodes", type=int, default=5000, help="number of training episodes, only for policy-based methods")        # original: 2000
    parser.add_argument("--lr_actor", type=float, default=1e-4, help="learning rate of policy")
    parser.add_argument("--lr_critic", type=float, default=1e-3, help="learning rate of critic")
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--epoches", type=int, default=24, help="update times")                             # 24
    parser.add_argument("--episodes_per_train", type=int, default=5, help="update interval (episode)")     # 10, this setting only for policy-based methods
    parser.add_argument("--steps_per_train", type=int, default=5, help="update interval (step)")           #  original: 5, this setting only for value-based methods
    parser.add_argument("--evaluate_period", type=int, default=10, help="evaluation interval (episode)")

    args = parser.parse_args()

    return args
