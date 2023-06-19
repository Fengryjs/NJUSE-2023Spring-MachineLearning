# 2023Spring-机器学习强化学习大作业

[TOC]

## 作业分工

201250068 陈骏 独立完成

## MountainCar环境

![mountain_car.gif (600×400) (gymlibrary.dev)](https://www.gymlibrary.dev/_images/mountain_car.gif)

[OpenAI-Mountain_car](https://www.gymlibrary.dev/environments/classic_control/mountain_car/)问题官方问题描述如下：

* The Mountain Car MDP is a deterministic MDP that consists of a car placed stochastically at the bottom of a sinusoidal valley, with the only possible actions being the accelerations that can be applied to the car in either direction. The goal of the MDP is to strategically accelerate the car to reach the goal state on top of the right hill. There are two versions of the mountain car domain in gym: one with discrete actions and one with continuous. This version is the one with discrete actions.

小车在推力和重力的作用下发生二维移动，但是仅靠推力的作用无法到达右侧旗帜处，因此需要爬到左侧山上，积攒动力势能，在加上向右的推力，到达右侧旗帜。与基本的强化学习模型类似，小车通过试错的方式，在不同的点如果采用不同的策略（mountain_car问题中为三种情况，向左施加力，不施加，向右施加力），为怎样影响回馈函数。

## DQN

Deep Q-learNing是通过神经网络近似值函数的方法，通过NetWork来代替Table，通过两个网络Q和Q_target来计算损失函数loss，相比于Q-learning算法进行了优化。

### 算法步骤

1. 初始化 $Q$ 和 $Q_{target}$ ，使得$Q=Q_{target}$，以同样的args初始化MLP网络模型

   ~~~python
   class DQN:
       def __init__(self, args):
           self.args = args
           self.train_step = 0
   
           # create the network
           self.q_network = MLP(args)
           self.target_q_network = MLP(args)
   ~~~

2. 在每个 $eposide$，通过 $eposide -greed$ 贪婪算法，从Q上寻找当前$state$下应采取的策略$action$

   ~~~python
   for time_step in tqdm(range(1, self.args.max_time_steps + 1)):
               # 如果到达时间限制，或者结束了 done = True Reset 环境
               if time_step % self.episode_limit == 0 or done:
                   state, info = self.env.reset()
                   done = False
               # 进行 epsilon-greedy policy 贪婪搜索
               with torch.no_grad():
                   action = self.agent.select_action(state, self.epsilon)
   ~~~

3. 根据选择的策略$action$，在环境中进行迭代，更新$state$以及$replay\_buffer$

   ~~~python
   next_state, reward, terminated, truncated, info = self.env.step(action)
   self.buffer.store_episode(state, action, reward, next_state, done)
   ~~~

4. 通过对buffer抽样，进行神经网络$Q$的更新

   ~~~python
   transitions = self.buffer.sample(self.args.batch_size)
   self.agent.learn(transitions, self.logger)
   ~~~

5. 按照固定间隔，更新$Q_{target}$

   ~~~python
   if self.train_step > 0 and (self.train_step % self.args.target_update_cycle) == 0:
   	hard_update(self.target_q_network, self.q_network)
   ~~~

### 代码补全

#### epsilon_greedy

```python
def select_action(self, states, epsilon):
    # TODO: 补全epsilon_greedy代码实现
    # 如果小于 epsilon
    if np.random.uniform() < epsilon:
        # {(): 1, (276,): 0, (275,): 2, (275, 276): 1}
        action = np.random.randint(self.args.n_actions)  # added
    else:
        inputs = torch.tensor(states, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        q_value = self.policy.q_network.forward(inputs)  # added
        action = np.argmax(q_value)  # added
        action = action.cpu().numpy()
    return action
```

#### target_q

```python
with torch.no_grad():
    q_next_all = self.target_q_network(next_states)
    q_next = torch.max(q_next_all, dim=1, keepdim=True)[0]
    # TODO: 补全target_q的计算相关代码实现
    target_q = rewards + done_multiplier * q_next  # added
```

## 实验结果

### Average Return曲线

![plt](.\log\MountainCar-v0\DQN\order_1\plt.png)

### 验证视频

<video src=".\RL_result.mp4"></video>
## 框架Bug

1. step函数返回值在新版gym中变为5个，见gym.core，修改runner_dpn.py line65为如下，其余step调用类似处理

   ~~~python
               next_state, reward, terminated, truncated, info = self.env.step(action)
               done = terminated | truncated
   ~~~

   ~~~python
       def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
           """Run one timestep of the environment's dynamics.
   
           When end of episode is reached, you are responsible for calling :meth:`reset` to reset this environment's state.
           Accepts an action and returns either a tuple `(observation, reward, terminated, truncated, info)`.
   
           Args:
               action (ActType): an action provided by the agent
   
           Returns:
               observation (object): this will be an element of the environment's :attr:`observation_space`.
                   This may, for instance, be a numpy array containing the positions and velocities of certain objects.
               reward (float): The amount of reward returned as a result of taking the action.
               terminated (bool): whether a `terminal state` (as defined under the MDP of the task) is reached.
                   In this case further step() calls could return undefined results.
               truncated (bool): whether a truncation condition outside the scope of the MDP is satisfied.
                   Typically a timelimit, but could also be used to indicate agent physically going out of bounds.
                   Can be used to end the episode prematurely before a `terminal state` is reached.
               info (dictionary): `info` contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                   This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                   hidden from observations, or individual reward terms that are combined to produce the total reward.
                   It also can contain information that distinguishes truncation and termination, however this is deprecated in favour
                   of returning two booleans, and will be removed in a future version.
   
               (deprecated)
               done (bool): A boolean value for if the episode has ended, in which case further :meth:`step` calls will return undefined results.
                   A done signal may be emitted for different reasons: Maybe the task underlying the environment was solved successfully,
                   a certain timelimit was exceeded, or the physics simulation has entered an invalid state.
           """
           raise NotImplementedError
   ~~~

2. reset函数返回值为一个ObsType加一个字典的元组，修改runner_dpn.py line45为如下，其余reset调用类似处理

   ~~~python
           state, info = self.env.reset()
   ~~~

   ~~~python
       def reset(
           self,
           *,
           seed: Optional[int] = None,
           options: Optional[dict] = None,
       ) -> Tuple[ObsType, dict]:
           """Resets the environment to an initial state and returns the initial observation.
   
           This method can reset the environment's random number generator(s) if ``seed`` is an integer or
           if the environment has not yet initialized a random number generator.
           If the environment already has a random number generator and :meth:`reset` is called with ``seed=None``,
           the RNG should not be reset. Moreover, :meth:`reset` should (in the typical use case) be called with an
           integer seed right after initialization and then never again.
   
           Args:
               seed (optional int): The seed that is used to initialize the environment's PRNG.
                   If the environment does not already have a PRNG and ``seed=None`` (the default option) is passed,
                   a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom).
                   However, if the environment already has a PRNG and ``seed=None`` is passed, the PRNG will *not* be reset.
                   If you pass an integer, the PRNG will be reset even if it already exists.
                   Usually, you want to pass an integer *right after the environment has been initialized and then never again*.
                   Please refer to the minimal example above to see this paradigm in action.
               options (optional dict): Additional information to specify how the environment is reset (optional,
                   depending on the specific environment)
   
   
           Returns:
               observation (object): Observation of the initial state. This will be an element of :attr:`observation_space`
                   (typically a numpy array) and is analogous to the observation returned by :meth:`step`.
               info (dictionary):  This dictionary contains auxiliary information complementing ``observation``. It should be analogous to
                   the ``info`` returned by :meth:`step`.
           """
           # Initialize the RNG if the seed is manually passed
           if seed is not None:
               self._np_random, seed = seeding.np_random(seed)
   ~~~

   