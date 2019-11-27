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
        rews = th.sum(rewards, dim=1)
        adv = (rews - rews.mean()) / rews.std()

        # Compute fraction * sum
        fraction = self.args.alpha / (n * self.args.sigma)
        for i in range(len(epsilons)):
            curr_eps = epsilons[i]
            f_i = adv[i].item()
            for j in curr_eps:
                j *= f_i

        # Compute sum of each epsilon
        summed = epsilons[0]
        for i in range(len(epsilons)):
            k = 0
            for j in epsilons[i]:
                summed[k] += j
                k += 1

        # Multiply times fraction
        for e in summed:
            e *= fraction

        # TODO: self.mac must be of type es_mac. Insert implementation error if not provided
        for e in epsilons:
            for j in e:
                if False not in th.isnan(j):
                    self.logger.console_logger.warning("Skipping NaN update (after frac comp) for network at iteration {}".format(t_env))
                    return

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

    def save_models(self, path):
        self.mac.save_models(path)

    def load_models(self, path):
        self.mac.load_models(path)
