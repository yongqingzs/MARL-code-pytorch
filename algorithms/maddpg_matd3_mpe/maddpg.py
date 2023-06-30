import torch
import torch.nn.functional as F
import numpy as np
import copy
from algorithms.maddpg_matd3_mpe.networks import Actor, Critic_MADDPG


class MADDPG(object):
    def __init__(self, args, agent_id):
        self.N = args.N  # num_agents
        self.agent_id = agent_id  # agent序号
        # +
        self.save_path = args.save_path
        self.load_path = args.load_path
        self.max_action = args.max_action  # 动作范围
        self.action_dim = args.action_dim_n[agent_id]  # 每个agent动作空间dim
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.gamma = args.gamma  # Discount factor
        self.tau = args.tau  # 更新目标网络的soft参数
        self.use_grad_clip = args.use_grad_clip  # 梯度裁剪系数?
        # Create an individual actor and critic for each agent according to the 'agent_id'
        # 这是非合作关系下的多智能体AC架构，也就是每个agent都有一个中心化的critic
        self.actor = Actor(args, agent_id)
        self.critic = Critic_MADDPG(args)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    def choose_action(self, obs, noise_std):
        """
        Each agent selects actions based on its own local observations(add noise for exploration)
        
        NOTE:
        1. 这个动作添加了噪声
        2. TODO: MADDPG是如何处理离散动作的，MADDPG似乎无法处理离散动作，
        而mae中的连续动作和离散动作之间比较特殊，连续动作的维数和离散动作的维数相一致，
        并且代表的意义也一致，所以可以将输入的连续动作(one-hot)处理成离散动作
        3. TODO: 对于普通环境，maddpg应该无法处理离散动作
        """
        obs = torch.unsqueeze(torch.tensor(obs, dtype=torch.float), 0)
        a = self.actor(obs).data.numpy().flatten()
        a = (a + np.random.normal(0, noise_std, size=self.action_dim)).clip(-self.max_action, self.max_action)
        return a

    def train(self, replay_buffer, agent_n):
        """
        atr:
        1. critic_loss: 来自得到的batch(obs, a)
        2. actor_loss: 
        - 对该actor对应的agent得到的batch_a进行重新action选择
        - batch_obs_n与critic_loss一致
        - critic(batch_obs_n, batch_a_n).mean()
        - TODO: mean()带来的结果需要进行测试
        """
        batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = replay_buffer.sample()

        # Compute target_Q
        with torch.no_grad():  # target_Q has no gradient
            # Select next actions according to the actor_target
            batch_a_next_n = [agent.actor_target(batch_obs_next) for agent, batch_obs_next in zip(agent_n, batch_obs_next_n)]
            Q_next = self.critic_target(batch_obs_next_n, batch_a_next_n)
            target_Q = batch_r_n[self.agent_id] + self.gamma * (1 - batch_done_n[self.agent_id]) * Q_next  # shape:(batch_size,1)

        # batch_obs_n、 batch_a_n都是所有agents的总体batch
        current_Q = self.critic(batch_obs_n, batch_a_n)  # shape:(batch_size,1)
        critic_loss = F.mse_loss(target_Q, current_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()

        # Reselect the actions of the agent corresponding to 'agent_id'，the actions of other agents remain unchanged
        batch_a_n[self.agent_id] = self.actor(batch_obs_n[self.agent_id])
        actor_loss = -self.critic(batch_obs_n, batch_a_n).mean()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.use_grad_clip:  # 梯度裁剪系数?
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optimizer.step()

        # Softly update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_model(self):
        """
        save_path: './model/{algorithm}/{env_name}_{algorithm}'
        path: save_path + '_{agent_id}.pkl'
        """
        torch.save(self.actor.state_dict(), self.save_path + '_{}.pkl'.format(self.agent_id))

    def load_model(self):
        """
        使用模型时进行加载，
        只加载actor_network，没有添加device

        atr:
        1. path: load_path + '_{agent_id}.pkl'
        2. load_path: ?

        NOTE:
        1. 需要和actor序号相对应，而actor序号由外部输入path控制
        2. TODO: 需要测试load_state_dict的效果
        3. load_path不等于save_path
        """
        path = self.load_path + '_{}.pkl'.format(self.agent_id)
        self.actor.load_state_dict(torch.load(path))
        print('Agent {} successfully loaded actor_network'.format(self.agent_id))
