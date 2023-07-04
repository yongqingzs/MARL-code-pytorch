import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from envs.mpe_from_openai.make_env import make_env  # mpe_from_openai为注释后的版本
import argparse
from algorithms.mappo_mpe.normalization import Normalization, RewardScaling
from algorithms.mappo_mpe.replay_buffer import ReplayBuffer
from algorithms.mappo_mpe.mappo_mpe import MAPPO_MPE

"""
Only for homogenous agents environments like Spread in MPE, 
all agents have the same dimension of observation space and action space.
Only for discrete action space.

NOTE:
1. 在maddpgs中并没有类似的obs限制，obs限制是因为这里的mappo实现只只有一个
actor和critic
2. 改造:
- 添加save_path、load_path
- load_path默认是save_path，但可以进行修改
"""

class Runner:
    """
    atr:
    1. evalute_env: 和maddpgs不同，没有该atr
    2. env.action_space:
    - .n: 只有离散动作空间有
    - .space[0]: 只有连续动作空间有
    """
    def __init__(self, args, env, number=1, seed=0):
        self.args = args
        self.number = number
        self.seed = seed

        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Create env
        self.env = env  # default: Discrete action space
        self.env_name = self.args.env_name  # +
        self.args.N = self.env.num_agents  # The number of agents
        # obs_dim_n, action_dim_n并没有参与实际的训练中
        self.args.obs_dim_n = [self.env.observation_space[i].shape[0] for i in range(self.args.N)]  # obs dimensions of N agents
        self.args.action_dim_n = [self.env.action_space[i].n for i in range(self.args.N)]  # actions dimensions of N agents, TODO: 将n改为shape[0]
        
        # Only for homogenous agents environments like Spread in MPE, all agents have the same dimension of observation space and action space
        self.args.obs_dim = self.args.obs_dim_n[0]  # The dimensions of an agent's observation space
        self.args.action_dim = self.args.action_dim_n[0]  # The dimensions of an agent's action space
        self.args.state_dim = np.sum(self.args.obs_dim_n)  # The dimensions of global state space（Sum of the dimensions of the local observation space of all agents）
        print("observation_space=", self.env.observation_space)
        print("obs_dim_n={}".format(self.args.obs_dim_n))
        print("action_space=", self.env.action_space)
        print("action_dim_n={}".format(self.args.action_dim_n))

        # Create N agents，只有一个actor、critic
        self.agent_n = MAPPO_MPE(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        # Create a tensorboard
        self.writer = SummaryWriter(log_dir=self.args.tensorboard_log)

        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0

        # Reward norm and scaling, TODO: 这一块还不是很清楚
        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=self.args.N)
        elif self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=self.args.N, gamma=self.args.gamma)

        # + 
        self.reward_array = []

    def run(self, ):
        """
        atr:
        1. episode_steps: 完成一个episode的步数
        2. batch_size: the number of episodes，区别于ppo中的batch_size
        """
        evaluate_num = -1  # Record the number of evaluations
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:  # 只是为了evaluate_freq
                self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1

            _, episode_steps = self.run_episode_mpe(evaluate=False)  # Run an episode
            self.total_steps += episode_steps

            if self.replay_buffer.episode_num == self.args.batch_size:
                self.agent_n.train(self.replay_buffer, self.total_steps)  # Training
                self.replay_buffer.reset_buffer()

        self.evaluate_policy()
        self.env.close()

    def evaluate_policy(self, ):
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            episode_reward, _ = self.run_episode_mpe(evaluate=True)
            evaluate_reward += episode_reward

        evaluate_reward = evaluate_reward / self.args.evaluate_times
        self.evaluate_rewards.append(evaluate_reward)
        print("total_steps:{} \t evaluate_reward:{}".format(self.total_steps, evaluate_reward))
        self.writer.add_scalar('evaluate_step_rewards_{}'.format(self.env_name), evaluate_reward, global_step=self.total_steps)
        
        # Save the rewards and models
        np.savez(self.args.save_path + '.npz', np.array(self.reward_array))
        # np.save('./data_train/MAPPO_env_{}_number_{}_seed_{}.npy'.format(self.env_name, self.number, self.seed), np.array(self.evaluate_rewards))
        self.agent_n.save_model()

    def run_episode_mpe(self, evaluate=False):
        """
        进行环境的运行和交互，但不涉及agents的更新

        atr:
        1. evaluate: 控制action的随机性，run: Flase，evaluate: True
        2. episode_step: 指的是一个episode的steps
        3. episode_limit: 因为mpe中没有done_callback，所以必须给episode加上限制
        4. store_last_value: 区别于store_transition，增加episode_num += 1
        - 在每个episode结束的时候，对def进行调用
        - episode_num指的是记录的episode数目(但每次训练后会进行清空) -> 见run()
        5. episode_reward: 只会添加r_n[0]

        NOTE:
        1. 因为evaluate_policy同样要对def进行调用，
        所以reward_array.append位置和maddpgs不相同
        """
        episode_reward = 0
        obs_n = self.env.reset()

        # Reward scaling and rnn
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        if self.args.use_rnn:  # If use RNN, before the beginning of each episode，reset the rnn_hidden of the Q network.
            self.agent_n.actor.rnn_hidden = None
            self.agent_n.critic.rnn_hidden = None

        for episode_step in range(self.args.episode_limit):
            a_n, a_logprob_n = self.agent_n.choose_action(obs_n, evaluate=evaluate)  # Get actions and the corresponding log probabilities of N agents
            s = np.array(obs_n).flatten()  # In MPE, global state is the concatenation of all agents' local obs.
            v_n = self.agent_n.get_value(s)  # Get the state values (V(s)) of N agents
            obs_next_n, r_n, done_n, _ = self.env.step(a_n)
            episode_reward += r_n[0]

            if not evaluate:
                if self.args.use_reward_norm:
                    r_n = self.reward_norm(r_n)
                elif args.use_reward_scaling:
                    r_n = self.reward_scaling(r_n)

                # Store the transition
                self.replay_buffer.store_transition(episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n)

            obs_n = obs_next_n

            if all(done_n):
                break

        if not evaluate:
            # An episode is over, store v_n in the last step
            s = np.array(obs_n).flatten()
            v_n = self.agent_n.get_value(s)
            self.replay_buffer.store_last_value(episode_step + 1, v_n)
            # +
            self.reward_array.append(r_n)

        return episode_reward, episode_step + 1

    # +
    def load_models(self, path=None):
        """
        该方法不能通过argparse使用，而是通过configs

        atr:
        1. load_path(default): './model/{algorithm}/{env_name}_{algorithm}'
        2. agent_n: 不同于maddpgs，消去了[agent_id]
        """
        if path is not None:
            self.args.load_path = path
        self.agent_n.load_model()

    # +
    def actions_by_models(self, obs_n):
        """
        obs_n由外部输入
        """
        a_n, _ = self.agent_n.choose_action(obs_n, evaluate=True)  # Get actions and the corresponding log probabilities of N agents
        return a_n


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in MPE environment")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=25, help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (the number of episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the rnn")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the mlp")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GAE parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="GAE parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling. Here, we do not use it.")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_relu", type=float, default=False, help="Whether to use relu, if False, we will use tanh")
    parser.add_argument("--use_rnn", type=bool, default=False, help="Whether to use RNN")
    parser.add_argument("--add_agent_id", type=float, default=False, help="Whether to add agent_id. Here, we do not use it.")
    parser.add_argument("--use_value_clip", type=float, default=False, help="Whether to use value clip.")

    args = parser.parse_args()
    env = make_env('spread')
    runner = Runner(args, env, number=1, seed=0)
    runner.run()
