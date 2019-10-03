from components.episode_buffer import EpisodeBatch
import torch as th


class ESLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.n_agents = args.n_agents
        self.scheme = scheme
        self.logger = logger
        self.input_shape = self._get_input_shape(scheme)
        self.params = list(mac.parameters())
        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        extrap = batch["epsilons"][0, 0]
        epsilons = [i.item() for i in extrap]

        # Prepare structures
        n = self.args.batch_size_run
        agent = self.mac.agent

        # Compute gradient ascent step
        reward_epsilons_sum = 0
        chunk_size = rewards.shape[1] // n
        for i in range(n):
            start = chunk_size * i
            if i == n and rewards.shape[1]-(chunk_size*i) == 1:     # odd reward shape (cannot divide by even n)
                chunk_size += 1
            curr_rewards = rewards.narrow(1, start, chunk_size).sum().item()
            reward_epsilons_sum += curr_rewards * epsilons[i]

        fraction = self.args.alpha * reward_epsilons_sum / (n * self.args.sigma)

        # Perform gradient ascent step
        with th.no_grad():
            for param in agent.parameters():
                param.__add__(fraction)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("reward_mean", rewards.sum().item()/self.args.n_agents, t_env)
            self.log_stats_t = t_env

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape

    def cuda(self):
        self.mac.cuda()
