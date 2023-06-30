class Args:
    def __init__(self, env_name, algorithm) -> None:
        self.max_train_steps = int(1e6)
        self.episode_limit = 25
        self.evaluate_freq = 5000
        self.evaluate_times = 3
        self.max_action = 1.0

        self.algorithm = algorithm  
        self.buffer_size = int(1e6)
        self.batch_size = 1024
        self.hidden_dim = 64
        self.noise_std_init = 0.2
        self.noise_std_min = 0.05
        self.noise_decay_steps = 3e5
        self.use_noise_decay = True
        self.lr_a = 5e-4
        self.lr_c = 5e-4
        self.gamma = 0.95
        self.tau = 0.01
        self.use_orthogonal_init = True
        self.use_grad_clip = True
        # --------------------------------------MATD3--------------------------------------------------------------------
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_update_freq = 2
        self.noise_std_decay = (self.noise_std_init - self.noise_std_min) / self.noise_decay_steps
        # +
        self.env_name = env_name
        self.tensorboard_log = 'tensorboard/{}/{}_{}'.format(self.algorithm, self.env_name, self.algorithm)
        self.save_path = './model/{}/{}_{}'.format(self.algorithm, self.env_name, self.algorithm)
        self.load_path = self.save_path  # default
