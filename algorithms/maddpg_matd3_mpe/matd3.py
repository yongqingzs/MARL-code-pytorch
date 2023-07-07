import torch
import torch.nn.functional as F
import numpy as np
import copy
from algorithms.maddpg_matd3_mpe.networks import Actor, Critic_MATD3


class MATD3(object):
    def __init__(self, args, agent_id):
        self.N = args.N
        self.agent_id = agent_id
        # +
        self.save_path = args.save_path
        self.load_path = args.load_path
        self.max_action = args.max_action
        self.action_dim = args.action_dim_n[agent_id]
        self.lr_a = args.lr_a
        self.lr_c = args.lr_c
        self.gamma = args.gamma
        self.tau = args.tau
        self.use_grad_clip = args.use_grad_clip
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.policy_update_freq = args.policy_update_freq
        self.actor_pointer = 0
        # Create an individual actor and critic for each agent according to the 'agent_id'
        self.actor = Actor(args, agent_id)
        self.critic = Critic_MATD3(args)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    # Each agent selects actions based on its own local observations(add noise for exploration)
    def choose_action(self, obs, noise_std):
        obs = torch.unsqueeze(torch.tensor(obs, dtype=torch.float), 0)
        a = self.actor(obs).data.numpy().flatten()
        a = (a + np.random.normal(0, noise_std, size=self.action_dim)).clip(-self.max_action, self.max_action)
        return a

    def train(self, replay_buffer, agent_n):
        self.actor_pointer += 1
        batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = replay_buffer.sample()

        # Compute target_Q
        with torch.no_grad():  # target_Q has no gradient
            # Trick 1:target policy smoothing
            batch_a_next_n = []
            for i in range(self.N):
                batch_a_next = agent_n[i].actor_target(batch_obs_next_n[i])
                noise = (torch.randn_like(batch_a_next) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                batch_a_next = (batch_a_next + noise).clamp(-self.max_action, self.max_action)
                batch_a_next_n.append(batch_a_next)

            # Trick 2:clipped double Q-learning
            Q1_next, Q2_next = self.critic_target(batch_obs_next_n, batch_a_next_n)
            target_Q = batch_r_n[self.agent_id] + self.gamma * (1 - batch_done_n[self.agent_id]) * torch.min(Q1_next, Q2_next)  # shape:(batch_size,1)

        # Compute current_Q
        current_Q1, current_Q2 = self.critic(batch_obs_n, batch_a_n)  # shape:(batch_size,1)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()

        # Trick 3:delayed policy updates
        if self.actor_pointer % self.policy_update_freq == 0:
            # Reselect the actions of the agent corresponding to 'agent_id', the actions of other agents remain unchanged
            batch_a_n[self.agent_id] = self.actor(batch_obs_n[self.agent_id])
            actor_loss = -self.critic.Q1(batch_obs_n, batch_a_n).mean()  # Only use Q1
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.actor_optimizer.step()

            # Softly update the target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    # +
    def save_model(self):
        """
        save_path: './model/{algorithm}/{env_name}_{algorithm}'
        path: save_path + '_{agent_id}.pkl'
        """
        path = self.save_path + '_{}.pkl'.format(self.agent_id)
        torch.save(self.actor.state_dict(), path)
        
    # +
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
