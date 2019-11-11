from components.episode_buffer import EpisodeBatch
import torch as th


class ESLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.n_agents = args.n_agents
        self.scheme = scheme
        self.logger = logger
        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        # terminated = batch["terminated"][:, :-1].float()
        # mask = batch["filled"][:, :-1].float()
        # mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        epsilons = batch["epsilons"]

        # Prepare structures
        n = self.args.batch_size_run

        # Compute fraction * sum
        fraction = self.args.alpha / (n * self.args.sigma)
        for i in range(len(epsilons)):
            curr_eps = epsilons[i]
            f_i = rewards[i].sum().item()
            for j in curr_eps:
                j *= f_i * fraction

        # Compute sum of each epsilon
        summed = epsilons[0]
        for i in range(1, len(epsilons)):
            k = 0
            for j in epsilons[i]:
                summed[k] += j
                k += 1

        # TODO: self.mac must be of type es_mac. Insert implementation error if not provided

        # Perform a gradient ascent step
        self.mac.gradient_ascent_step(summed)

        # Reset the controller for the next episode
        self.mac.reset()

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape

    def cuda(self):
        self.mac.cuda()
