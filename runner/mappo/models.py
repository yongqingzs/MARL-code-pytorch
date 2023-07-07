import numpy as np

import os
import sys
os.chdir(sys.path[0])
from algorithms.mappo_mpe.mappo_mpe_main import Runner
from runner.mappo.configs.mpe_config import Args
from envs.mpe_from_openai.make_env import make_env

ENV_ID = 'simple_spread'
STEPS = '50w'

if __name__ == '__main__':
    args = Args(ENV_ID)
    env = make_env(ENV_ID, discrete=False)
    
    args.load_path = './models/{}/{}_{}'.format(ENV_ID, ENV_ID, STEPS)
    runner = Runner(args, env, number=1, seed=0)
    # load_path
    runner.load_models()
    total_reward_n = np.zeros(env.num_agents)
    # actions = [None] * env.num_agents
    done_n = [False] * env.num_agents
    obs_n = env.reset()
    # while False in done_n:
    for _ in range(25):
        actions = runner.actions_by_models(obs_n)
        obs_n_, reward_n, done_n, info_n = env.step(actions)
        for i, done in enumerate(done_n):
            if done is True:
                actions[i] = None
        total_reward_n += np.array(reward_n)
        obs_n = obs_n_
        env.render()
    print(total_reward_n)